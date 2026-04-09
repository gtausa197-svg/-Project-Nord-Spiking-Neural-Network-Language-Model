"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v5.0 / v4.2 — Training Script (700M)             ║
║                                                                        ║
║  Usage:                                                                ║
║      CUDA:   python train_nord_tpu_700m.py                             ║
║      TPU:    python train_nord_tpu_700m.py --tpu                         ║
║      Genesis: python train_nord_tpu_700m.py --genesis-v5               ║
║      1B:      python train_nord_tpu_700m.py --preset 1b                ║
║      1.1B:    python train_nord_tpu_700m.py --preset 1.1b              ║
║      KD/EWC:  --teacher-ckpt … --kd-weight … --ewc-path … --ewc-lambda ║
║                                                                        ║
║  ФІКСИ v5.1:                                                           ║
║    FIX 1: Токенайзер — fallback GPT2 якщо Llama недоступний           ║
║    FIX 2: persistent_mem=False під час тренування (GC сумісність)      ║
║    FIX 3: gradient_checkpointing вимикається автоматично на >24GB      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse, json, math, os, shutil, struct, sys, time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# ── Backend globals ──
USE_TPU = False
xm = None

def detect_backend(force_tpu=False):
    global USE_TPU, xm
    if force_tpu:
        try:
            import torch_xla.core.xla_model as _xm
            xm = _xm; USE_TPU = True
            print("  [✓] TPU backend: torch_xla loaded", flush=True); return
        except ImportError:
            print("  [!] --tpu but torch_xla not found, fallback CUDA", flush=True)
    if torch.cuda.is_available():
        print(f"  [✓] CUDA backend: {torch.cuda.get_device_name()}", flush=True)
    else:
        try:
            import torch_xla.core.xla_model as _xm
            xm = _xm; USE_TPU = True
            print("  [✓] TPU backend (auto-detected)", flush=True)
        except ImportError:
            print("  [!] CPU mode (very slow!)", flush=True)

def get_device():
    if USE_TPU: return xm.xla_device()
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nord_core_700m import NordConfig, NordModel, apply_nord_scale_preset


# ── KD + EWC ─────────────────────────────────────────────────────────────────
def _knowledge_distillation_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
    ignore_index: int = -100,
    labels: Optional[Tensor] = None,
) -> Tensor:
    T = max(temperature, 1e-3)
    s = student_logits.float() / T
    t = teacher_logits.float() / T
    log_p = F.log_softmax(s, dim=-1)
    q = F.softmax(t, dim=-1)
    if labels is not None:
        mask = labels != ignore_index
        if not mask.any():
            return student_logits.new_tensor(0.0)
        log_p = log_p[mask]
        q = q[mask]
    return F.kl_div(log_p, q, reduction="batchmean", log_target=False) * (T * T)


def lm_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    *,
    temperature: float = 2.0,
) -> Tensor:
    sl = student_logits[:, :-1, :].contiguous().reshape(-1, student_logits.size(-1))
    tl = teacher_logits[:, :-1, :].contiguous().reshape(-1, teacher_logits.size(-1))
    return _knowledge_distillation_loss(sl, tl, temperature=temperature)


class EWCHelper:
    def __init__(self, theta_star, fisher, device):
        self.theta_star = {k: v.to(device, non_blocking=True) for k, v in theta_star.items()}
        self.fisher = {k: v.to(device, non_blocking=True) for k, v in fisher.items()}

    @classmethod
    def load(cls, path, device):
        d = torch.load(path, map_location="cpu", weights_only=False)
        ts = d.get("theta_star") or d.get("optimal_weights")
        fi = d.get("fisher")
        if ts is None or fi is None:
            raise KeyError("EWC file must contain 'theta_star' and 'fisher' dicts")
        return cls(ts, fi, device)

    def penalty(self, model, lam):
        if lam <= 0:
            return next(model.parameters()).new_tensor(0.0)
        total = None
        for name, p in model.named_parameters():
            if not p.requires_grad: continue
            if name not in self.fisher or name not in self.theta_star: continue
            f = self.fisher[name]; t0 = self.theta_star[name]
            if f.shape != p.shape or t0.shape != p.shape: continue
            term = (f * (p - t0).pow(2)).sum()
            total = term if total is None else total + term
        if total is None:
            return next(model.parameters()).new_tensor(0.0)
        return lam * total


@torch.no_grad()
def estimate_diagonal_fisher(model, batches, *, vocab_size, pad_id, device, n_batches=50, dtype=torch.float32):
    model.eval()
    fisher = {}; n = 0
    for i, input_ids in enumerate(batches):
        if i >= n_batches: break
        input_ids = input_ids.to(device)
        model.zero_grad(set_to_none=True)
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().reshape(-1, vocab_size),
            input_ids[:, 1:].contiguous().reshape(-1),
            ignore_index=pad_id,
        )
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is None: continue
            g = p.grad.detach().float().pow(2)
            if name not in fisher: fisher[name] = g.clone()
            else: fisher[name].add_(g)
        n += 1
    if n == 0: return {}
    for name in fisher: fisher[name].div_(n)
    return fisher


def save_ewc_state(path, model, fisher):
    sd = model.state_dict()
    theta_star = {k: v.detach().cpu().clone() for k, v in sd.items() if k in fisher}
    fisher_cpu = {k: v.detach().cpu().clone() for k, v in fisher.items()}
    torch.save({"theta_star": theta_star, "fisher": fisher_cpu}, path)


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


def _nord_arch_key(cfg):
    return (
        cfg.d_model, cfg.n_heads, cfg.d_ff,
        cfg.sensory_layers, cfg.association_layers, cfg.executive_layers,
        cfg.n_experts, cfg.top_k_experts,
        cfg.genesis_autogenic_v5, cfg.genesis_dual_memory, cfg.memory_size,
        cfg.conversational_balancer, cfg.conversational_balancer_hidden,
        cfg.paper_cortical_stack_v5,
    )


def _load_teacher_from_ckpt(ckpt_path, device, dtype):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = dict(ck.get("config") or {})
    tcfg = NordConfig()
    for k, v in raw.items():
        if hasattr(tcfg, k) and k not in ("device", "dtype"):
            setattr(tcfg, k, v)
    tcfg.device = str(device)
    tcfg.dtype = dtype if device.type == "cuda" else torch.float32
    t = NordModel(tcfg).to(device)
    sd = ck["model_state_dict"]
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    filt = {k: v for k, v in sd.items() if "_v_mem_state" not in k and "_i_syn_state" not in k}
    t.load_state_dict(filt, strict=False)
    t.eval()
    for param in t.parameters(): param.requires_grad_(False)
    return t, tcfg


# ════════════════════════════════════════════════════════════════════════
# FIX 1: TOKENIZER з fallback — не висить якщо Llama недоступний
# ════════════════════════════════════════════════════════════════════════
class NordTokenizer:
    def __init__(self, cfg):
        from transformers import AutoTokenizer
        import os

        tokenizer_id = cfg.tokenizer_id
        hf_token = os.environ.get("HF_TOKEN", None)

        # Спроба завантажити потрібний токенайзер
        loaded = False
        if hf_token:
            try:
                print(f"  [*] Loading tokenizer {tokenizer_id} (з HF_TOKEN)...", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    token=hf_token,
                    trust_remote_code=True,
                )
                loaded = True
                print(f"  [✓] Tokenizer {tokenizer_id} завантажено", flush=True)
            except Exception as e:
                print(f"  [!] Не вдалось завантажити {tokenizer_id}: {e}", flush=True)
        else:
            try:
                print(f"  [*] Loading tokenizer {tokenizer_id}...", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id,
                    trust_remote_code=True,
                    local_files_only=False,
                )
                loaded = True
                print(f"  [✓] Tokenizer {tokenizer_id} завантажено", flush=True)
            except Exception as e:
                print(f"  [!] Не вдалось завантажити {tokenizer_id}: {e}", flush=True)

        # Fallback на GPT-2 якщо Llama не завантажився
        if not loaded:
            print(f"  [!] Fallback → GPT-2 токенайзер (відкритий, без токену)", flush=True)
            try:
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                cfg.tokenizer_id = "gpt2"
                print(f"  [✓] GPT-2 токенайзер завантажено", flush=True)
            except Exception as e2:
                print(f"  [✗] Навіть GPT-2 не завантажився: {e2}", flush=True)
                raise

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_len = cfg.max_seq_len
        self.vocab_size = self.tokenizer.vocab_size
        if cfg.vocab_size < self.vocab_size:
            cfg.vocab_size = self.vocab_size
            print(f"  [*] vocab_size оновлено: {cfg.vocab_size:,}", flush=True)
        print(f"  [✓] Токенайзер готовий (vocab={self.vocab_size:,}, pad_id={self.tokenizer.pad_token_id})", flush=True)

    def encode(self, text):
        return self.tokenizer(
            text, return_tensors="pt",
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
        ).input_ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id


# ── LMDB Dataset ──
class LMDBDataset(Dataset):
    def __init__(self, db_path, max_seq_len):
        import lmdb
        self.db_path = db_path; self.max_seq_len = max_seq_len; self._env = None
        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            self.length = struct.unpack("<Q", txn.get(b"__len__"))[0]
        env.close()
        print(f"  [✓] LMDB: {self.length:,} samples", flush=True)

    def _get_env(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(
                self.db_path, readonly=True, lock=False,
                readahead=True, meminit=False, max_readers=64,
            )
        return self._env

    def __len__(self): return self.length

    def __getitem__(self, idx):
        env = self._get_env()
        with env.begin(write=False) as txn:
            raw = txn.get(f"sample_{idx:010d}".encode())
        ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        S = self.max_seq_len
        return ids[:S] if ids.shape[0] >= S else F.pad(ids, (0, S - ids.shape[0]))


def build_lmdb(jsonl_path, db_path, tokenizer, max_seq_len, map_size_gb=80.0):
    import lmdb, numpy as np
    print(f"\n  [*] Building LMDB (streaming)...", flush=True)
    t1 = time.time(); BATCH = 1024; PAD_ID = tokenizer.pad_id
    env = lmdb.open(db_path, map_size=int(map_size_gb * (1024**3)))
    txn = env.begin(write=True)
    count = 0; total_tokens = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        batch = []
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            text = obj.get("text") or obj.get("content") or obj.get("passage", "")
            if len(text) < 30: continue
            batch.append(text)

            if len(batch) >= BATCH:
                enc = tokenizer.tokenizer(
                    batch, max_length=max_seq_len, truncation=True,
                    padding="max_length", return_tensors="np", return_attention_mask=False,
                )
                ids_np = enc.input_ids.astype(np.int32)
                for j in range(ids_np.shape[0]):
                    row = ids_np[j]
                    non_pad = int(np.sum(row != PAD_ID))
                    if non_pad < 10: continue
                    txn.put(f"sample_{count:010d}".encode(), row.tobytes())
                    count += 1; total_tokens += non_pad
                batch = []

                if count % 500_000 < BATCH and count >= 500_000:
                    txn.commit(); txn = env.begin(write=True)
                    print(f"      {count:,} samples | {total_tokens/1e6:.0f}M tok", flush=True)

        # Залишок
        if batch:
            enc = tokenizer.tokenizer(
                batch, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="np", return_attention_mask=False,
            )
            ids_np = enc.input_ids.astype(np.int32)
            for j in range(ids_np.shape[0]):
                row = ids_np[j]
                non_pad = int(np.sum(row != PAD_ID))
                if non_pad < 10: continue
                txn.put(f"sample_{count:010d}".encode(), row.tobytes())
                count += 1; total_tokens += non_pad

    txn.put(b"__len__", struct.pack("<Q", count))
    txn.put(b"__total_tokens__", struct.pack("<Q", total_tokens))
    txn.commit(); env.close()
    print(f"  [✓] LMDB: {count:,} samples, {total_tokens/1e6:.1f}M tokens in {time.time()-t1:.0f}s", flush=True)


# ── LR Schedule ──
def get_lr(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Checkpoint Manager ──
class CheckpointManager:
    def __init__(self, save_dir, keep_last=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, model, optimizer, step, loss, cfg, scaler=None):
        path = self.save_dir / f"nord_v4_step_{step:07d}.pt"
        m = model.module if hasattr(model, 'module') else model
        ver = "5.0-genesis-autogenic" if getattr(cfg, "genesis_autogenic_v5", False) else "v4.2"
        d = {
            "step": step, "loss": loss, "version": ver,
            "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and k != "dtype"},
        }
        if scaler: d["scaler_state_dict"] = scaler.state_dict()
        if USE_TPU: xm.save(d, str(path))
        else: torch.save(d, path)
        latest = self.save_dir / "nord_v4_latest.pt"
        if latest.exists(): latest.unlink()
        shutil.copy2(path, latest)
        ckpts = sorted(self.save_dir.glob("nord_v4_step_*.pt"), key=lambda p: p.stat().st_mtime)
        for old in ckpts[:max(0, len(ckpts) - self.keep_last)]: old.unlink()
        print(f"  [💾] Saved: {path.name} (loss={loss:.4f})", flush=True)

    def load(self, model, optimizer, device, scaler=None):
        latest = self.save_dir / "nord_v4_latest.pt"
        if not latest.exists():
            ckpts = sorted(self.save_dir.glob("nord_v4_step_*.pt"))
            latest = ckpts[-1] if ckpts else None
        if latest is None: return 0
        print(f"  [*] Resuming from: {latest.name}", flush=True)
        ckpt = torch.load(latest, map_location="cpu", weights_only=False)
        m = model.module if hasattr(model, 'module') else model
        filtered = {k: v for k, v in ckpt["model_state_dict"].items()
                    if "_v_mem_state" not in k and "_i_syn_state" not in k}
        m.load_state_dict(filtered, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"  [✓] Resumed at step {ckpt['step']:,} (loss={ckpt.get('loss', '?')})", flush=True)
        return ckpt["step"]

    def save_final(self, model, cfg):
        path = self.save_dir / "nord_v4_final.pt"
        m = model.module if hasattr(model, 'module') else model
        ver = "5.0-genesis-autogenic" if getattr(cfg, "genesis_autogenic_v5", False) else "v4.2"
        d = {
            "version": ver,
            "model_state_dict": m.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and k != "dtype"},
        }
        if USE_TPU: xm.save(d, str(path))
        else: torch.save(d, path)
        print(f"  [⭐] Final model: {path}", flush=True)


def _stat_tensor(stats, key, device):
    t = stats.get(key)
    if t is None: return torch.tensor(0.0, device=device)
    if isinstance(t, torch.Tensor): return t.mean() if t.dim() > 0 else t
    return torch.tensor(float(t), device=device)


def _compute_output_entropy(logits):
    with torch.no_grad():
        l = logits[:, :-1, :].float()
        probs = F.softmax(l, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1).mean()
    return float(entropy.item())


# ── Training ──
def train(
    dataset_path,
    model_dir,
    lr_override=None,
    continued=False,
    genesis_dual_memory=False,
    genesis_v5=False,
    scale_preset="700m",
    teacher_ckpt=None,
    kd_weight=0.0,
    kd_temp=2.0,
    ewc_path=None,
    ewc_lambda=0.0,
    export_ewc=None,
    ewc_batches=64,
    no_gradient_checkpointing=False,
    conversational_balancer=False,
    paper_cortical_v5=False,
    stdp_active=True,
    stdp_update_every=10,
):
    device = get_device()

    base_lr = 2e-4
    if continued:
        base_lr = 5e-5
        print("  [*] Continued pretraining mode: LR=5e-5, warmup=200", flush=True)
    if lr_override is not None:
        base_lr = lr_override
        print(f"  [*] LR override: {base_lr}", flush=True)
    warmup = 200 if continued else 1000

    cfg = NordConfig(
        device=str(device),
        dtype=torch.bfloat16 if USE_TPU else torch.float16,
        d_model=1536, n_heads=24, d_ff=4096, n_clusters=128, max_seq_len=192,
        sensory_layers=3, association_layers=3, executive_layers=4,
        T=8, T_slow=2,
        # ════════════════════════════════════════════════════
        # FIX 2: persistent_mem=False під час тренування
        # Причина: gradient checkpointing + persistent LIF state
        # несумісні — різні shapes при recompute (335 vs 336)
        # persistent_mem=True використовується тільки при inference
        # ════════════════════════════════════════════════════
        persistent_mem=False,
        n_experts=4, top_k_experts=2,
        memory_size=256,
        memory_tau_mem=0.99, memory_n_read_heads=8,
        target_spike_rate=0.03, spike_loss_weight=0.5,
        v_threshold=0.12, tau_mem=0.9, lif_freeze_steps=1000,
        gradient_checkpointing=False,
        batch_size=1, grad_accum=64, lr=base_lr, min_lr=1e-5,
        warmup_steps=warmup, max_steps=50_000,
        save_every=1000, log_every=10,
        genesis_dual_memory=genesis_dual_memory and not genesis_v5,
        genesis_autogenic_v5=genesis_v5,
        conversational_balancer=conversational_balancer,
        paper_cortical_stack_v5=paper_cortical_v5,
    )
    apply_nord_scale_preset(cfg, scale_preset)

    # ════════════════════════════════════════════════════════
    # FIX 3: Автоматично вимикаємо gradient checkpointing
    # якщо VRAM >= 24GB або якщо --no-gradient-checkpointing
    # ════════════════════════════════════════════════════════
    if no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
        print("  [*] Gradient checkpointing: ВИМКНЕНО (--no-gradient-checkpointing)", flush=True)
    elif cfg.scale_preset_used in ("1b", "1.1b"):
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 24.0:
                # ≥24GB VRAM — не потрібен gradient checkpointing
                cfg.gradient_checkpointing = False
                print(f"  [*] VRAM {vram_gb:.0f}GB ≥ 24GB → gradient_checkpointing=OFF (автоматично)", flush=True)
            else:
                cfg.gradient_checkpointing = True
                print(f"  [*] VRAM {vram_gb:.0f}GB < 24GB → gradient_checkpointing=ON", flush=True)
        else:
            cfg.gradient_checkpointing = True
            print("  [*] Large preset: gradient_checkpointing=ON", flush=True)

    print(flush=True); print("═" * 60, flush=True)
    print("  PROJECT NORD — SNN Training", flush=True)
    if cfg.scale_preset_used == "1.1b":
        print("  ★ Preset 1.1b: ~1.1B params", flush=True)
    if genesis_v5:
        print("  ★ NORD 5.0 Genesis Autogenic", flush=True)
    if stdp_active:
        print(f"  ★ STDP online learning: АКТИВНИЙ (кожні {stdp_update_every} кроків)", flush=True)
    else:
        print("  ○ STDP: вимкнено", flush=True)
    print(f"  ★ persistent_mem=False (тренування режим)", flush=True)
    print(f"  ★ gradient_checkpointing={cfg.gradient_checkpointing}", flush=True)
    print("═" * 60, flush=True)

    if USE_TPU:
        print(f"  Device: TPU", flush=True)
        cfg.batch_size = 8; cfg.grad_accum = 4
    elif torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        n_gpus = torch.cuda.device_count()
        print(f"  GPU: {torch.cuda.get_device_name()} ({vram:.1f}GB)", flush=True)
        if vram < 16:
            print("  [ERROR] Need ≥16GB VRAM!", flush=True); sys.exit(1)

    print(f"  Arch: d={cfg.d_model}, h={cfg.n_heads}, ff={cfg.d_ff}", flush=True)
    print(f"  Zones: S({cfg.sensory_layers})→A({cfg.association_layers},MoE)→M→E({cfg.executive_layers})", flush=True)

    tokenizer = NordTokenizer(cfg)

    db_path = str(Path(dataset_path).with_suffix("")) + "_lmdb"
    if not Path(db_path).exists():
        build_lmdb(dataset_path, db_path, tokenizer, cfg.max_seq_len)
    dataset = LMDBDataset(db_path, cfg.max_seq_len)

    print(f"\n  [*] Building Nord model...", flush=True)
    model = NordModel(cfg).to(device)
    print(f"  [✓] {model.count_params()}", flush=True)

    n_gpus = 1
    if not USE_TPU and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        from torch.nn.parallel import DataParallel
        n_gpus = torch.cuda.device_count()
        print(f"  [⚡] {n_gpus} GPUs → DataParallel", flush=True)
        model = DataParallel(model)

    # ── Smart VRAM auto-tuning ──
    if not USE_TPU and torch.cuda.is_available():
        TARGET_VRAM_PCT = 0.85
        EFF_BATCH_TARGET = 32
        vram_total = torch.cuda.get_device_properties(0).total_memory
        vram_after_model = torch.cuda.memory_allocated()
        vram_free = vram_total - vram_after_model
        print(f"\n  [*] Smart VRAM auto-tuning...", flush=True)
        print(f"      Total VRAM: {vram_total/(1024**3):.1f}GB | Free: {vram_free/(1024**3):.1f}GB", flush=True)

        best_batch = 1
        test_seq_len = cfg.max_seq_len
        model.train()
        temp_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        temp_scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == torch.float16))

        for test_batch in [1, 2, 3, 4, 6, 8, 10, 12, 16]:
            per_gpu = test_batch // n_gpus if n_gpus > 1 else test_batch
            if per_gpu < 1: continue
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                dummy_ids = torch.randint(0, 1000, (test_batch, test_seq_len), device=device)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.dtype == torch.float16)):
                    logits, stats = model(dummy_ids)
                    loss = logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size).mean()
                temp_scaler.scale(loss).backward()
                temp_scaler.unscale_(temp_optim)
                temp_scaler.step(temp_optim)
                temp_scaler.update()
                temp_optim.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated()
                pct = peak / vram_total
                print(f"      batch={test_batch:>2} → peak {peak/(1024**3):.1f}GB ({pct:.0%})", flush=True)
                if pct <= TARGET_VRAM_PCT: best_batch = test_batch
                else: break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"      batch={test_batch:>2} → OOM!", flush=True)
                    break
                else: raise

        del temp_optim, temp_scaler
        torch.cuda.empty_cache()
        model_to_reinit = model.module if hasattr(model, 'module') else model
        model_to_reinit.__init__(cfg)
        model_to_reinit.to(device)
        if hasattr(model, 'module'): model = DataParallel(model_to_reinit)
        cfg.batch_size = best_batch
        cfg.grad_accum = max(1, EFF_BATCH_TARGET // best_batch)
        print(f"  [✓] Auto-tuned: batch={cfg.batch_size}, accum={cfg.grad_accum}, eff={cfg.batch_size*cfg.grad_accum}", flush=True)

    if USE_TPU:
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)

    print(f"  Eff batch: {cfg.batch_size}×{cfg.grad_accum}={cfg.batch_size*cfg.grad_accum}", flush=True)
    print(f"  LR: {cfg.lr}→{cfg.min_lr} (cosine, {cfg.warmup_steps} warmup)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = None
    if not USE_TPU and cfg.dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    ckpt_mgr = CheckpointManager(model_dir)
    start_step = ckpt_mgr.load(model, optimizer, device, scaler)

    if export_ewc:
        m0 = _unwrap_model(model)
        batch_list = []
        it0 = iter(dataloader)
        for _ in range(min(ewc_batches, len(dataloader))):
            try: batch_list.append(next(it0))
            except StopIteration: break
        if not batch_list:
            print("  [✗] export-ewc: no batches", flush=True); sys.exit(1)
        fish = estimate_diagonal_fisher(
            m0, iter(batch_list),
            vocab_size=cfg.vocab_size, pad_id=tokenizer.pad_id,
            device=device, n_batches=len(batch_list),
        )
        save_ewc_state(export_ewc, m0, fish)
        print(f"  [✓] EWC saved to {export_ewc}", flush=True); sys.exit(0)

    teacher = None
    if teacher_ckpt:
        tc, tcfg = _load_teacher_from_ckpt(Path(teacher_ckpt), device, cfg.dtype)
        if _nord_arch_key(cfg) != _nord_arch_key(tcfg):
            print("  [!] Teacher arch ≠ student: KD disabled.", flush=True)
        elif kd_weight <= 0:
            print("  [!] Teacher loaded but --kd-weight=0; no KD.", flush=True)
        else:
            teacher = tc
            print(f"  [✓] KD teacher (weight={kd_weight}, T={kd_temp})", flush=True)

    ewc_helper = None
    if ewc_lambda > 0:
        if ewc_path and Path(ewc_path).is_file():
            ewc_helper = EWCHelper.load(ewc_path, device)
            print(f"  [✓] EWC λ={ewc_lambda}", flush=True)
        else:
            print("  [!] EWC off (path missing)", flush=True)

    def _train_autocast():
        if USE_TPU: return torch.autocast(device_type="xla", dtype=torch.bfloat16)
        if device.type == "cuda":
            return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.dtype == torch.float16))
        return nullcontext()

    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0; running_spike_loss = 0.0; tokens_seen = 0; t_start = time.time()
    _stdp_entropy_ema = 0.0; _stdp_updates_total = 0; _stdp_blocked_total = 0

    print(f"\n  {'─'*55}", flush=True)
    print(f"  Start step {start_step:,} | {len(dataset):,} samples", flush=True)
    print(f"  Ctrl+C = save & stop", flush=True)
    print(f"  {'─'*55}\n", flush=True)

    try:
        for step in range(start_step, cfg.max_steps):
            accum_loss = 0.0; accum_spike_loss = 0.0; stats = {}

            for _ in range(cfg.grad_accum):
                try: input_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader); input_ids = next(data_iter)
                input_ids = input_ids.to(device)

                with _train_autocast():
                    logits, stats = model(input_ids)
                    ce_loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size),
                        input_ids[:, 1:].contiguous().reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )
                    spike_loss = stats.get("spike_loss", torch.tensor(0.0, device=device))
                    if isinstance(spike_loss, torch.Tensor) and spike_loss.dim() > 0:
                        spike_loss = spike_loss.mean()
                    else:
                        spike_loss = torch.tensor(float(spike_loss) if not isinstance(spike_loss, torch.Tensor) else spike_loss.item(), device=device)

                    moe_lb = stats.get("moe_lb_loss", torch.tensor(0.0, device=device))
                    if isinstance(moe_lb, torch.Tensor) and moe_lb.dim() > 0: moe_lb = moe_lb.mean()

                    gen_bal  = _stat_tensor(stats, "genesis_dual_balance", device)
                    purp     = _stat_tensor(stats, "genesis_purpose_entropy", device)
                    idreg    = _stat_tensor(stats, "identity_hidden_norm", device)
                    arch_ent = _stat_tensor(stats, "genesis_archive_attn_entropy", device)

                    base = (
                        ce_loss + spike_loss + 0.01 * moe_lb
                        + cfg.genesis_balance_loss_weight * gen_bal
                        - cfg.genesis_purpose_entropy_weight * purp
                        + cfg.genesis_identity_reg_weight * idreg
                        - cfg.genesis_archive_entropy_weight * arch_ent
                    )

                    kd_term = ce_loss.new_tensor(0.0)
                    if teacher is not None and kd_weight > 0:
                        with torch.no_grad(): t_logits, _ = teacher(input_ids)
                        kd_term = lm_kd_loss(logits, t_logits, temperature=kd_temp) * kd_weight

                    ewc_term = ce_loss.new_tensor(0.0)
                    if ewc_helper is not None:
                        ewc_term = ewc_helper.penalty(_unwrap_model(model), ewc_lambda)

                    loss = (base + kd_term + ewc_term) / cfg.grad_accum

                if USE_TPU: loss.backward()
                elif scaler is not None: scaler.scale(loss).backward()
                else: loss.backward()

                accum_loss += ce_loss.item() / cfg.grad_accum
                sp_item = spike_loss.item() if isinstance(spike_loss, torch.Tensor) else float(spike_loss)
                accum_spike_loss += sp_item / cfg.grad_accum
                tokens_seen += input_ids.numel()

            # Optimizer step
            if USE_TPU:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
            else:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # STDP online learning
            if stdp_active and step % stdp_update_every == 0:
                entropy = _compute_output_entropy(logits)
                _stdp_entropy_ema = 0.9 * _stdp_entropy_ema + 0.1 * entropy
                m_core = _unwrap_model(model)
                m_core.stdp.set_output_entropy(entropy)
                will_apply = m_core.stdp.should_apply_plasticity()
                if will_apply: _stdp_updates_total += 1
                else: _stdp_blocked_total += 1

            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups: pg["lr"] = lr
            running_loss += accum_loss; running_spike_loss += accum_spike_loss

            # Logging
            if step % cfg.log_every == 0 and step > start_step:
                avg = running_loss / cfg.log_every
                avg_sp = running_spike_loss / cfg.log_every
                tps = tokens_seen / (time.time() - t_start) / 1000
                sp = stats.get("sparsity", 0)
                if isinstance(sp, torch.Tensor): sp = sp.mean().item()
                mem_r = stats.get("memory_spike_rate", None)
                if isinstance(mem_r, torch.Tensor): mem_r = mem_r.mean().item()
                mem_s = f" | mem={mem_r:.3f}" if mem_r is not None else ""
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                dev_s = f" | VRAM {torch.cuda.memory_allocated()/(1024**3):.1f}G" if torch.cuda.is_available() else ""
                stdp_s = f" | ent={_stdp_entropy_ema:.2f}" if stdp_active else ""
                print(
                    f"  step {step:>7,} │ loss {avg:.4f} │ spike_L {avg_sp:.4f} │ "
                    f"lr {lr:.1e} │ grad {gn:.1f} │ sparsity {sp:.0%} │ "
                    f"{tps:.1f}k tok/s{mem_s}{stdp_s}{dev_s}",
                    flush=True,
                )
                running_loss = 0.0; running_spike_loss = 0.0

            if step % 100 == 0 and step > start_step:
                print(f"  {'·'*50}", flush=True)
                sr = stats.get("spike_rates_tensor", [])
                if isinstance(sr, torch.Tensor):
                    sr = sr.float()
                    if sr.dim() > 1: sr = sr.mean(dim=0)
                    sr = sr.tolist()
                if sr:
                    ns = cfg.sensory_layers + 1; na = cfg.association_layers
                    print(f"    Sensory:     {[f'{r:.4f}' for r in sr[:ns]]}", flush=True)
                    print(f"    Association: {[f'{r:.4f}' for r in sr[ns:ns+na]]}", flush=True)
                    print(f"    Executive:   {[f'{r:.4f}' for r in sr[ns+na:]]}", flush=True)
                gate = stats.get("gate_activity"); mix = stats.get("memory_mix")
                if isinstance(gate, torch.Tensor): gate = gate.mean().item()
                if isinstance(mix, torch.Tensor): mix = mix.mean().item()
                if gate is not None:
                    print(f"    Memory gate={gate:.4f} mix={mix:.4f}", flush=True)
                if stdp_active and (_stdp_updates_total + _stdp_blocked_total) > 0:
                    total_c = _stdp_updates_total + _stdp_blocked_total
                    pct = _stdp_updates_total / total_c * 100
                    print(f"    STDP ent={_stdp_entropy_ema:.3f} | активацій={_stdp_updates_total} ({pct:.0f}%)", flush=True)
                print(f"  {'·'*50}", flush=True)

            if step > 0 and step % cfg.save_every == 0:
                ckpt_mgr.save(model, optimizer, step, accum_loss, cfg, scaler)

    except KeyboardInterrupt:
        print(f"\n\n  [⏸] Stopped at step {step:,}", flush=True)
        ckpt_mgr.save(model, optimizer, step, accum_loss, cfg, scaler)

    ckpt_mgr.save_final(model, cfg)
    print(f"\n  {'═'*55}\n  Training complete! Model: {model_dir}\n  {'═'*55}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--continued", action="store_true")
    parser.add_argument("--genesis-dual-memory", action="store_true")
    parser.add_argument("--genesis-v5", action="store_true")
    parser.add_argument("--preset", type=str, default="700m", choices=("700m", "1b", "1.1b"))
    parser.add_argument("--teacher-ckpt", type=str, default=None)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--ewc-path", type=str, default=None)
    parser.add_argument("--ewc-lambda", type=float, default=0.0)
    parser.add_argument("--export-ewc", type=str, default=None)
    parser.add_argument("--ewc-batches", type=int, default=64)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--conversational-balancer", action="store_true")
    parser.add_argument("--paper-cortical-v5", action="store_true")
    parser.add_argument("--no-stdp", action="store_true")
    parser.add_argument("--stdp-every", type=int, default=10)

    args = parser.parse_args()

    if args.genesis_v5 and args.genesis_dual_memory:
        print("  [✗] Use either --genesis-v5 or --genesis-dual-memory, not both."); sys.exit(1)
    if getattr(args, "paper_cortical_v5", False) and not args.genesis_v5:
        print("  [✗] --paper-cortical-v5 requires --genesis-v5"); sys.exit(1)

    print("=" * 60, flush=True)
    print("  PROJECT NORD v5.1 — Brain-Inspired SNN Training", flush=True)
    print("=" * 60, flush=True)
    detect_backend(force_tpu=args.tpu)

    if args.dataset:
        dataset_path = args.dataset
    else:
        d = "train_data.jsonl"
        print(f"\n  Dataset? (Enter = {d})", flush=True)
        inp = input("  Dataset: ").strip()
        dataset_path = inp if inp else d
    if not Path(dataset_path).exists():
        print(f"  [✗] Not found: {dataset_path}"); sys.exit(1)

    if args.model_dir:
        model_dir = args.model_dir
    else:
        d = "nord_v4_700m"
        print(f"\n  Model dir? (Enter = {d})", flush=True)
        inp = input("  Model dir: ").strip()
        model_dir = inp if inp else d

    train(
        dataset_path, model_dir,
        lr_override=args.lr,
        continued=args.continued,
        genesis_dual_memory=args.genesis_dual_memory,
        genesis_v5=args.genesis_v5,
        scale_preset=args.preset,
        teacher_ckpt=args.teacher_ckpt,
        kd_weight=args.kd_weight,
        kd_temp=args.kd_temperature,
        ewc_path=args.ewc_path,
        ewc_lambda=args.ewc_lambda,
        export_ewc=args.export_ewc,
        ewc_batches=args.ewc_batches,
        no_gradient_checkpointing=args.no_gradient_checkpointing,
        conversational_balancer=args.conversational_balancer,
        paper_cortical_v5=args.paper_cortical_v5,
        stdp_active=not args.no_stdp,
        stdp_update_every=args.stdp_every,
    )


if __name__ == "__main__":
    main()