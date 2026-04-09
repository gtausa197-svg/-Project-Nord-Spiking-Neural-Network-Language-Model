"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v5.0 — Training Script with Distillation          ║
║         Compatible with existing checkpoints (step 22000+)             ║
║                                                                        ║
║  Usage:                                                                ║
║    Continue WITHOUT KD (як раніше):                                    ║
║      python train_nord_kd.py --dataset data.jsonl --preset 1.1b        ║
║          --genesis-v5 --model_dir nord_v4_700m                         ║
║                                                                        ║
║    Continue WITH KD from HuggingFace model:                            ║
║      python train_nord_kd.py --dataset data.jsonl --preset 1.1b        ║
║          --genesis-v5 --model_dir nord_v4_700m                         ║
║          --kd-teacher meta-llama/Llama-3.2-1B --kd-weight 0.5          ║
║                                                                        ║
║    Continue WITH KD from GPT-2 (no HF token needed):                   ║
║      python train_nord_kd.py --dataset data.jsonl --preset 1.1b        ║
║          --genesis-v5 --model_dir nord_v4_700m                         ║
║          --kd-teacher gpt2-medium --kd-weight 0.5                      ║
║                                                                        ║
║  KD Schedule (when --kd-teacher is set):                               ║
║    Phase 1 (0-30% remaining steps):  α = kd_weight (heavy guidance)    ║
║    Phase 2 (30-70% remaining):       α linearly decays to 0            ║
║    Phase 3 (70-100% remaining):      α = 0 (pure SNN + STDP)          ║
║                                                                        ║
║  IMPORTANT: Uses SAME nord_core_700m.py as before — no arch changes.   ║
║  Checkpoint format unchanged — can resume from any existing checkpoint.║
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


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE DISTILLATION — HuggingFace Teacher
# ═══════════════════════════════════════════════════════════════════════════════
class HFTeacherDistiller:
    """Loads any HuggingFace CausalLM as teacher for knowledge distillation.

    KD Schedule (relative to REMAINING steps from resume point):
      Phase 1 (first 30%):   α = kd_weight (full guidance)
      Phase 2 (30%-70%):     α linearly decays to 0
      Phase 3 (last 30%):    α = 0 (pure SNN learning)

    This means if you resume at step 22000 with max_steps=100000,
    remaining = 78000 steps. Phase 1 = steps 22000-45400, etc.
    """

    def __init__(self, model_name: str, device, dtype, vocab_size: int):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n  [*] Loading KD teacher: {model_name}...", flush=True)

        hf_token = os.environ.get("HF_TOKEN", None)
        load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
        if hf_token:
            load_kwargs["token"] = hf_token

        # Try loading with device_map for large models
        try:
            self.teacher = AutoModelForCausalLM.from_pretrained(
                model_name, device_map={"": device}, **load_kwargs
            )
        except Exception:
            # Fallback: load to CPU then move
            self.teacher = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            ).to(device)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        t_params = sum(p.numel() for p in self.teacher.parameters())
        t_vocab = self.teacher.config.vocab_size
        print(f"  [✓] Teacher: {model_name} ({t_params/1e6:.0f}M params, vocab={t_vocab:,})", flush=True)

        # Check vocab mismatch
        self.teacher_vocab = t_vocab
        self.student_vocab = vocab_size
        self.vocab_mismatch = (t_vocab != vocab_size)
        if self.vocab_mismatch:
            self.min_vocab = min(t_vocab, vocab_size)
            print(f"  [!] Vocab mismatch: teacher={t_vocab:,} student={vocab_size:,}", flush=True)
            print(f"      KD will use first {self.min_vocab:,} logits (intersection)", flush=True)
        else:
            self.min_vocab = vocab_size

        self.device = device

    def get_kd_alpha(self, current_step: int, start_step: int, max_steps: int,
                     base_weight: float) -> float:
        """Compute KD weight based on schedule relative to remaining steps."""
        if base_weight <= 0:
            return 0.0

        remaining_total = max(max_steps - start_step, 1)
        steps_done = current_step - start_step
        progress = steps_done / remaining_total

        if progress < 0.3:
            # Phase 1: full teacher guidance
            return base_weight
        elif progress < 0.7:
            # Phase 2: linear decay
            decay = (progress - 0.3) / 0.4  # 0→1
            return base_weight * (1.0 - decay)
        else:
            # Phase 3: pure SNN
            return 0.0

    @torch.no_grad()
    def get_teacher_logits(self, input_ids: Tensor) -> Tensor:
        """Get soft targets from teacher model."""
        output = self.teacher(input_ids)
        logits = output.logits.detach()

        # Handle vocab mismatch: truncate to min vocab
        if self.vocab_mismatch:
            logits = logits[:, :, :self.min_vocab]

        return logits

    def compute_kd_loss(self, student_logits: Tensor, teacher_logits: Tensor,
                        temperature: float = 3.0) -> Tensor:
        """KL divergence loss with temperature scaling.
        PDF-compatible: Δw ← Δw · (1 + d) where d comes from this loss gradient."""
        T = max(temperature, 0.5)

        # Shift logits for next-token prediction
        s_logits = student_logits[:, :-1, :].contiguous()
        t_logits = teacher_logits[:, :-1, :].contiguous()

        # Handle vocab mismatch
        if self.vocab_mismatch:
            s_logits = s_logits[:, :, :self.min_vocab]
            t_logits = t_logits[:, :, :self.min_vocab]

        s_flat = s_logits.reshape(-1, s_logits.size(-1)).float() / T
        t_flat = t_logits.reshape(-1, t_logits.size(-1)).float() / T

        log_p = F.log_softmax(s_flat, dim=-1)
        q = F.softmax(t_flat, dim=-1)

        return F.kl_div(log_p, q, reduction="batchmean", log_target=False) * (T * T)


# ═══════════════════════════════════════════════════════════════════════════════
# KD from Nord checkpoint (teacher = older/different Nord model)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_nord_teacher(ckpt_path, device, dtype):
    """Load another Nord model as teacher (for self-distillation)."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# TOKENIZER (same as original)
# ═══════════════════════════════════════════════════════════════════════════════
class NordTokenizer:
    def __init__(self, cfg):
        from transformers import AutoTokenizer

        tokenizer_id = cfg.tokenizer_id
        hf_token = os.environ.get("HF_TOKEN", None)
        loaded = False

        for tid, token in [(tokenizer_id, hf_token), ("gpt2", None)]:
            try:
                kw = {"token": token} if token else {}
                self.tokenizer = AutoTokenizer.from_pretrained(tid, trust_remote_code=True, **kw)
                cfg.tokenizer_id = tid
                loaded = True
                print(f"  [✓] Tokenizer: {tid}", flush=True)
                break
            except Exception as e:
                print(f"  [!] {tid}: {e}", flush=True)

        if not loaded:
            raise RuntimeError("No tokenizer available")

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_len = cfg.max_seq_len
        self.vocab_size = self.tokenizer.vocab_size
        if cfg.vocab_size < self.vocab_size:
            cfg.vocab_size = self.vocab_size
        print(f"  [✓] vocab={self.vocab_size:,}, pad_id={self.pad_id}", flush=True)

    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt", max_length=self.max_len,
                              truncation=True, padding="max_length").input_ids

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id


# ═══════════════════════════════════════════════════════════════════════════════
# LMDB DATASET (same as original)
# ═══════════════════════════════════════════════════════════════════════════════
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
            self._env = lmdb.open(self.db_path, readonly=True, lock=False,
                                  readahead=True, meminit=False, max_readers=64)
        return self._env

    def __len__(self): return self.length

    def __getitem__(self, idx):
        with self._get_env().begin(write=False) as txn:
            raw = txn.get(f"sample_{idx:010d}".encode())
        ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        S = self.max_seq_len
        return ids[:S] if ids.shape[0] >= S else F.pad(ids, (0, S - ids.shape[0]))


def build_lmdb(jsonl_path, db_path, tokenizer, max_seq_len, map_size_gb=80.0):
    import lmdb, numpy as np
    print(f"\n  [*] Building LMDB...", flush=True)
    t1 = time.time(); BATCH = 1024; PAD_ID = tokenizer.pad_id
    env = lmdb.open(db_path, map_size=int(map_size_gb * (1024**3)))
    txn = env.begin(write=True); count = 0; total_tokens = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        batch = []
        for line in f:
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            text = obj.get("text") or obj.get("content") or obj.get("passage", "")
            if len(text) < 30: continue
            batch.append(text)
            if len(batch) >= BATCH:
                enc = tokenizer.tokenizer(batch, max_length=max_seq_len, truncation=True,
                    padding="max_length", return_tensors="np", return_attention_mask=False)
                for row in enc.input_ids.astype(np.int32):
                    non_pad = int(np.sum(row != PAD_ID))
                    if non_pad < 10: continue
                    txn.put(f"sample_{count:010d}".encode(), row.tobytes())
                    count += 1; total_tokens += non_pad
                batch = []
                if count % 500_000 < BATCH and count >= 500_000:
                    txn.commit(); txn = env.begin(write=True)
                    print(f"      {count:,} samples | {total_tokens/1e6:.0f}M tok", flush=True)
        if batch:
            enc = tokenizer.tokenizer(batch, max_length=max_seq_len, truncation=True,
                padding="max_length", return_tensors="np", return_attention_mask=False)
            for row in enc.input_ids.astype(np.int32):
                non_pad = int(np.sum(row != PAD_ID))
                if non_pad < 10: continue
                txn.put(f"sample_{count:010d}".encode(), row.tobytes())
                count += 1; total_tokens += non_pad
    txn.put(b"__len__", struct.pack("<Q", count))
    txn.put(b"__total_tokens__", struct.pack("<Q", total_tokens))
    txn.commit(); env.close()
    print(f"  [✓] {count:,} samples, {total_tokens/1e6:.1f}M tokens in {time.time()-t1:.0f}s", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
def get_lr(step, cfg):
    if step < cfg.warmup_steps: return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _stat_tensor(stats, key, device):
    t = stats.get(key)
    if t is None: return torch.tensor(0.0, device=device)
    if isinstance(t, torch.Tensor): return t.mean() if t.dim() > 0 else t
    return torch.tensor(float(t), device=device)


def _compute_output_entropy(logits):
    with torch.no_grad():
        probs = F.softmax(logits[:, :-1, :].float(), dim=-1)
        return float(-(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item())


def _unwrap_model(m):
    return m.module if hasattr(m, "module") else m


class CheckpointManager:
    """Same format as original — fully compatible with existing checkpoints."""
    def __init__(self, save_dir, keep_last=5):
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, model, optimizer, step, loss, cfg, scaler=None):
        path = self.save_dir / f"nord_v4_step_{step:07d}.pt"
        m = model.module if hasattr(model, 'module') else model
        ver = "5.0-genesis-autogenic" if getattr(cfg, "genesis_autogenic_v5", False) else "v4.2"
        d = {"step": step, "loss": loss, "version": ver,
             "model_state_dict": m.state_dict(),
             "optimizer_state_dict": optimizer.state_dict(),
             "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and k != "dtype"}}
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
        torch.save({"version": ver, "model_state_dict": m.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and k != "dtype"}}, path)
        print(f"  [⭐] Final model: {path}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def train(
    dataset_path, model_dir,
    lr_override=None, continued=False,
    genesis_dual_memory=False, genesis_v5=False,
    scale_preset="700m",
    # ── KD from HuggingFace model (NEW) ──
    kd_teacher_hf=None,      # HF model name: "gpt2-medium", "meta-llama/Llama-3.2-1B", etc
    kd_weight=0.5,            # Base KD weight (will be scheduled)
    kd_temperature=3.0,       # Softmax temperature for KD
    # ── KD from Nord checkpoint (existing) ──
    teacher_ckpt=None,        # Path to Nord .pt checkpoint
    # ── Other ──
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
    warmup = 200 if continued else 1000

    cfg = NordConfig(
        device=str(device),
        dtype=torch.bfloat16 if USE_TPU else torch.float16,
        d_model=1536, n_heads=24, d_ff=4096, n_clusters=128, max_seq_len=192,
        sensory_layers=3, association_layers=3, executive_layers=4,
        T=8, T_slow=2, persistent_mem=False,
        n_experts=4, top_k_experts=2, memory_size=256,
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

    # Auto gradient checkpointing
    if no_gradient_checkpointing:
        cfg.gradient_checkpointing = False
    elif cfg.scale_preset_used in ("1b", "1.1b") and torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        cfg.gradient_checkpointing = vram_gb < 24.0
        print(f"  [*] VRAM {vram_gb:.0f}GB → gradient_checkpointing={'ON' if cfg.gradient_checkpointing else 'OFF'}", flush=True)

    # ── Banner ──
    print("\n" + "═" * 60, flush=True)
    print("  PROJECT NORD — SNN Training + Knowledge Distillation", flush=True)
    if cfg.scale_preset_used == "1.1b":
        print("  ★ Preset 1.1b: ~1.1B params", flush=True)
    if genesis_v5:
        print("  ★ NORD 5.0 Genesis Autogenic", flush=True)
    if kd_teacher_hf:
        print(f"  ★ KD Teacher: {kd_teacher_hf} (weight={kd_weight}, T={kd_temperature})", flush=True)
        print(f"    Schedule: 30% full → 30-70% decay → 70%+ pure SNN", flush=True)
    elif teacher_ckpt:
        print(f"  ★ KD Teacher: Nord checkpoint {teacher_ckpt}", flush=True)
    if stdp_active:
        print(f"  ★ STDP: ON (every {stdp_update_every} steps)", flush=True)
    print(f"  ★ persistent_mem=False | gradient_checkpointing={cfg.gradient_checkpointing}", flush=True)
    print("═" * 60, flush=True)

    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  GPU: {torch.cuda.get_device_name()} ({vram:.1f}GB)", flush=True)

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
        model = DataParallel(model)
        print(f"  [⚡] {n_gpus} GPUs → DataParallel", flush=True)

    # ── VRAM auto-tuning ──
    if not USE_TPU and torch.cuda.is_available():
        TARGET_VRAM_PCT = 0.80 if kd_teacher_hf else 0.85  # Leave room for teacher
        EFF_BATCH_TARGET = 32
        vram_total = torch.cuda.get_device_properties(0).total_memory
        print(f"\n  [*] VRAM auto-tuning (target {TARGET_VRAM_PCT:.0%})...", flush=True)

        best_batch = 1
        model.train()
        temp_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        temp_scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == torch.float16))

        for tb in [1, 2, 3, 4, 6, 8, 10, 12, 16]:
            if n_gpus > 1 and tb // n_gpus < 1: continue
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            try:
                dummy = torch.randint(0, 1000, (tb, cfg.max_seq_len), device=device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(cfg.dtype == torch.float16)):
                    logits, stats = model(dummy)
                    loss = logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size).mean()
                temp_scaler.scale(loss).backward()
                temp_scaler.unscale_(temp_optim); temp_scaler.step(temp_optim)
                temp_scaler.update(); temp_optim.zero_grad(set_to_none=True)
                peak = torch.cuda.max_memory_allocated()
                pct = peak / vram_total
                print(f"      batch={tb:>2} → {peak/(1024**3):.1f}GB ({pct:.0%})", flush=True)
                if pct <= TARGET_VRAM_PCT: best_batch = tb
                else: break
            except RuntimeError:
                torch.cuda.empty_cache()
                print(f"      batch={tb:>2} → OOM!", flush=True); break

        del temp_optim, temp_scaler; torch.cuda.empty_cache()
        m_reinit = model.module if hasattr(model, 'module') else model
        m_reinit.__init__(cfg); m_reinit.to(device)
        if n_gpus > 1:
            from torch.nn.parallel import DataParallel
            model = DataParallel(m_reinit)
        cfg.batch_size = best_batch
        cfg.grad_accum = max(1, EFF_BATCH_TARGET // best_batch)
        print(f"  [✓] batch={cfg.batch_size}, accum={cfg.grad_accum}, eff={cfg.batch_size*cfg.grad_accum}", flush=True)

    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
        persistent_workers=True) if not USE_TPU else DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)

    print(f"  Eff batch: {cfg.batch_size}×{cfg.grad_accum}={cfg.batch_size*cfg.grad_accum}", flush=True)
    print(f"  LR: {cfg.lr}→{cfg.min_lr} (cosine, {cfg.warmup_steps} warmup)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=True) if (
        not USE_TPU and cfg.dtype == torch.float16) else None

    ckpt_mgr = CheckpointManager(model_dir)
    start_step = ckpt_mgr.load(model, optimizer, device, scaler)

    # ── Load KD teacher ──
    hf_distiller = None
    nord_teacher = None

    if kd_teacher_hf:
        hf_distiller = HFTeacherDistiller(kd_teacher_hf, device, cfg.dtype, cfg.vocab_size)
    elif teacher_ckpt:
        nord_teacher, _ = _load_nord_teacher(Path(teacher_ckpt), device, cfg.dtype)
        print(f"  [✓] Nord teacher loaded from {teacher_ckpt}", flush=True)

    def _train_autocast():
        if USE_TPU: return torch.autocast(device_type="xla", dtype=torch.bfloat16)
        if device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=torch.float16, enabled=(cfg.dtype == torch.float16))
        return nullcontext()

    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0; running_spike_loss = 0.0; running_kd_loss = 0.0
    tokens_seen = 0; t_start = time.time()
    _stdp_entropy_ema = 0.0; _stdp_updates = 0; _stdp_blocked = 0

    print(f"\n  {'─'*55}", flush=True)
    print(f"  Start step {start_step:,} | {len(dataset):,} samples", flush=True)
    print(f"  Ctrl+C = save & stop", flush=True)
    print(f"  {'─'*55}\n", flush=True)

    try:
        for step in range(start_step, cfg.max_steps):
            accum_loss = 0.0; accum_spike_loss = 0.0; accum_kd_loss = 0.0; stats = {}

            for _ in range(cfg.grad_accum):
                try: input_ids = next(data_iter)
                except StopIteration: data_iter = iter(dataloader); input_ids = next(data_iter)
                input_ids = input_ids.to(device)

                with _train_autocast():
                    logits, stats = model(input_ids)

                    ce_loss = F.cross_entropy(
                        logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size),
                        input_ids[:, 1:].contiguous().reshape(-1),
                        ignore_index=tokenizer.pad_id)

                    spike_loss = stats.get("spike_loss", torch.tensor(0.0, device=device))
                    if isinstance(spike_loss, Tensor) and spike_loss.dim() > 0: spike_loss = spike_loss.mean()
                    elif not isinstance(spike_loss, Tensor): spike_loss = torch.tensor(float(spike_loss), device=device)

                    moe_lb = stats.get("moe_lb_loss", torch.tensor(0.0, device=device))
                    if isinstance(moe_lb, Tensor) and moe_lb.dim() > 0: moe_lb = moe_lb.mean()

                    gen_bal = _stat_tensor(stats, "genesis_dual_balance", device)
                    purp = _stat_tensor(stats, "genesis_purpose_entropy", device)
                    idreg = _stat_tensor(stats, "identity_hidden_norm", device)
                    arch_ent = _stat_tensor(stats, "genesis_archive_attn_entropy", device)

                    # ── KD Loss ──
                    kd_loss = torch.tensor(0.0, device=device)
                    kd_alpha = 0.0

                    if hf_distiller is not None:
                        kd_alpha = hf_distiller.get_kd_alpha(step, start_step, cfg.max_steps, kd_weight)
                        if kd_alpha > 0:
                            t_logits = hf_distiller.get_teacher_logits(input_ids)
                            kd_loss = hf_distiller.compute_kd_loss(logits, t_logits, kd_temperature)

                    elif nord_teacher is not None and kd_weight > 0:
                        kd_alpha = kd_weight  # No schedule for Nord teacher
                        with torch.no_grad():
                            t_logits, _ = nord_teacher(input_ids)
                        # Use simple KL div
                        sl = logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size).float() / kd_temperature
                        tl = t_logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size).float() / kd_temperature
                        kd_loss = F.kl_div(F.log_softmax(sl, -1), F.softmax(tl, -1),
                                           reduction="batchmean") * (kd_temperature ** 2)
                        kd_loss = kd_loss * kd_weight

                    # ── Total Loss ──
                    base = (
                        (1.0 - kd_alpha) * ce_loss
                        + kd_alpha * kd_loss
                        + spike_loss + 0.01 * moe_lb
                        + cfg.genesis_balance_loss_weight * gen_bal
                        - cfg.genesis_purpose_entropy_weight * purp
                        + cfg.genesis_identity_reg_weight * idreg
                        - cfg.genesis_archive_entropy_weight * arch_ent
                    )

                    loss = base / cfg.grad_accum

                if USE_TPU: loss.backward()
                elif scaler is not None: scaler.scale(loss).backward()
                else: loss.backward()

                accum_loss += ce_loss.item() / cfg.grad_accum
                accum_spike_loss += spike_loss.item() / cfg.grad_accum
                accum_kd_loss += (kd_loss.item() * kd_alpha) / cfg.grad_accum if kd_alpha > 0 else 0
                tokens_seen += input_ids.numel()

            # Optimizer step
            if USE_TPU:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                xm.optimizer_step(optimizer); optimizer.zero_grad(set_to_none=True)
            else:
                if scaler:
                    scaler.unscale_(optimizer)
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # STDP
            if stdp_active and step % stdp_update_every == 0:
                ent = _compute_output_entropy(logits)
                _stdp_entropy_ema = 0.9 * _stdp_entropy_ema + 0.1 * ent
                m_core = _unwrap_model(model)
                m_core.stdp.set_output_entropy(ent)
                if m_core.stdp.should_apply_plasticity(): _stdp_updates += 1
                else: _stdp_blocked += 1

            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups: pg["lr"] = lr
            running_loss += accum_loss
            running_spike_loss += accum_spike_loss
            running_kd_loss += accum_kd_loss

            # ── Logging ──
            if step % cfg.log_every == 0 and step > start_step:
                avg = running_loss / cfg.log_every
                avg_sp = running_spike_loss / cfg.log_every
                avg_kd = running_kd_loss / cfg.log_every
                tps = tokens_seen / (time.time() - t_start) / 1000
                sp = stats.get("sparsity", 0)
                if isinstance(sp, Tensor): sp = sp.mean().item()
                mem_r = stats.get("memory_spike_rate", None)
                if isinstance(mem_r, Tensor): mem_r = mem_r.mean().item()
                gn = grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm

                mem_s = f" | mem={mem_r:.3f}" if mem_r is not None else ""
                stdp_s = f" | ent={_stdp_entropy_ema:.2f}" if stdp_active else ""
                vram_s = f" | VRAM {torch.cuda.memory_allocated()/(1024**3):.1f}G" if torch.cuda.is_available() else ""
                kd_s = f" | KD={avg_kd:.3f}(α={kd_alpha:.2f})" if (hf_distiller or nord_teacher) else ""

                print(
                    f"  step {step:>7,} │ loss {avg:.4f} │ spike_L {avg_sp:.4f} │ "
                    f"lr {lr:.1e} │ grad {gn:.1f} │ sparsity {sp:.0%} │ "
                    f"{tps:.1f}k tok/s{kd_s}{mem_s}{stdp_s}{vram_s}",
                    flush=True,
                )
                running_loss = 0.0; running_spike_loss = 0.0; running_kd_loss = 0.0

            # ── Detailed stats ──
            if step % 100 == 0 and step > start_step:
                print(f"  {'·'*50}", flush=True)
                sr = stats.get("spike_rates_tensor", [])
                if isinstance(sr, Tensor):
                    sr = sr.float()
                    if sr.dim() > 1: sr = sr.mean(dim=0)
                    sr = sr.tolist()
                if sr:
                    ns = cfg.sensory_layers + 1; na = cfg.association_layers
                    print(f"    Sensory:     {[f'{r:.4f}' for r in sr[:ns]]}", flush=True)
                    print(f"    Association: {[f'{r:.4f}' for r in sr[ns:ns+na]]}", flush=True)
                    print(f"    Executive:   {[f'{r:.4f}' for r in sr[ns+na:]]}", flush=True)
                    avg_fr = sum(sr) / len(sr)
                    print(f"    Avg firing rate: {avg_fr:.4f} (target: {cfg.target_spike_rate})", flush=True)

                gate = stats.get("gate_activity"); mix = stats.get("memory_mix")
                if isinstance(gate, Tensor): gate = gate.mean().item()
                if isinstance(mix, Tensor): mix = mix.mean().item()
                if gate is not None:
                    print(f"    Memory gate={gate:.4f} mix={mix:.4f}", flush=True)

                if hf_distiller:
                    phase = "HEAVY" if kd_alpha >= kd_weight * 0.9 else ("DECAY" if kd_alpha > 0 else "OFF")
                    print(f"    KD α={kd_alpha:.3f} phase={phase} (teacher: {kd_teacher_hf})", flush=True)

                if stdp_active and (_stdp_updates + _stdp_blocked) > 0:
                    total_c = _stdp_updates + _stdp_blocked
                    pct = _stdp_updates / total_c * 100
                    print(f"    STDP ent={_stdp_entropy_ema:.3f} | активацій={_stdp_updates} ({pct:.0f}%)", flush=True)

                print(f"  {'·'*50}", flush=True)

            if step > 0 and step % cfg.save_every == 0:
                ckpt_mgr.save(model, optimizer, step, accum_loss, cfg, scaler)

    except KeyboardInterrupt:
        print(f"\n\n  [⏸] Stopped at step {step:,}", flush=True)
        ckpt_mgr.save(model, optimizer, step, accum_loss, cfg, scaler)

    ckpt_mgr.save_final(model, cfg)
    print(f"\n  {'═'*55}\n  Training complete! Model: {model_dir}\n  {'═'*55}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Nord v5.0 Training + Knowledge Distillation")
    parser.add_argument("--tpu", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--continued", action="store_true")
    parser.add_argument("--genesis-dual-memory", action="store_true")
    parser.add_argument("--genesis-v5", action="store_true")
    parser.add_argument("--preset", type=str, default="700m", choices=("700m", "1b", "1.1b"))
    # KD from HuggingFace
    parser.add_argument("--kd-teacher", type=str, default=None,
        help="HuggingFace model for KD: gpt2, gpt2-medium, meta-llama/Llama-3.2-1B, etc")
    parser.add_argument("--kd-weight", type=float, default=0.5,
        help="Base KD weight (will be scheduled: 30%% full → decay → 0)")
    parser.add_argument("--kd-temperature", type=float, default=3.0,
        help="Softmax temperature for KD (higher = softer targets)")
    # KD from Nord checkpoint
    parser.add_argument("--teacher-ckpt", type=str, default=None,
        help="Path to Nord checkpoint for self-distillation")
    # Other
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--conversational-balancer", action="store_true")
    parser.add_argument("--paper-cortical-v5", action="store_true")
    parser.add_argument("--no-stdp", action="store_true")
    parser.add_argument("--stdp-every", type=int, default=10)

    args = parser.parse_args()

    if args.genesis_v5 and args.genesis_dual_memory:
        print("  [✗] Use --genesis-v5 or --genesis-dual-memory, not both."); sys.exit(1)

    print("═" * 60, flush=True)
    print("  PROJECT NORD v5.0 — Training + Knowledge Distillation", flush=True)
    print("═" * 60, flush=True)
    detect_backend(force_tpu=args.tpu)

    if args.dataset:
        dataset_path = args.dataset
    else:
        d = "train_data.jsonl"
        inp = input(f"  Dataset? (Enter = {d}): ").strip()
        dataset_path = inp if inp else d
    if not Path(dataset_path).exists():
        print(f"  [✗] Not found: {dataset_path}"); sys.exit(1)

    if args.model_dir:
        model_dir = args.model_dir
    else:
        d = "nord_v4_700m"
        inp = input(f"  Model dir? (Enter = {d}): ").strip()
        model_dir = inp if inp else d

    train(
        dataset_path, model_dir,
        lr_override=args.lr, continued=args.continued,
        genesis_dual_memory=args.genesis_dual_memory,
        genesis_v5=args.genesis_v5,
        scale_preset=args.preset,
        kd_teacher_hf=args.kd_teacher,
        kd_weight=args.kd_weight,
        kd_temperature=args.kd_temperature,
        teacher_ckpt=args.teacher_ckpt,
        no_gradient_checkpointing=args.no_gradient_checkpointing,
        conversational_balancer=args.conversational_balancer,
        paper_cortical_v5=args.paper_cortical_v5,
        stdp_active=not args.no_stdp,
        stdp_update_every=args.stdp_every,
    )

if __name__ == "__main__":
    main()
