"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v4.2 — Training Script (700M)                     ║
║                                                                        ║
║  Usage:                                                                ║
║      CUDA:  python train_nord_700m.py                                  ║
║      TPU:   python train_nord_700m.py --tpu                            ║
║                                                                        ║
║  v4.2 (700M) — Supports CUDA GPU and Google Cloud TPU                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse, json, math, os, shutil, struct, sys, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
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
from nord_core_700m import NordConfig, NordModel

# ── Tokenizer ──
class NordTokenizer:
    def __init__(self, cfg):
        from transformers import AutoTokenizer
        print(f"  [*] Loading Llama-3.2 tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.max_len = cfg.max_seq_len
        self.vocab_size = self.tokenizer.vocab_size
        if cfg.vocab_size < self.vocab_size: cfg.vocab_size = self.vocab_size
        print(f"  [✓] Tokenizer ready (vocab={self.vocab_size:,})", flush=True)
    def encode(self, text):
        return self.tokenizer(text, return_tensors="pt", max_length=self.max_len, truncation=True, padding="max_length").input_ids
    def decode(self, ids): return self.tokenizer.decode(ids, skip_special_tokens=True)
    @property
    def pad_id(self): return self.tokenizer.pad_token_id

# ── LMDB Dataset ──
class LMDBDataset(Dataset):
    def __init__(self, db_path, max_seq_len):
        import lmdb
        self.db_path = db_path; self.max_seq_len = max_seq_len; self._env = None
        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn: self.length = struct.unpack("<Q", txn.get(b"__len__"))[0]
        env.close()
        print(f"  [✓] LMDB: {self.length:,} samples", flush=True)
    def _get_env(self):
        if self._env is None:
            import lmdb
            self._env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=True, meminit=False, max_readers=64)
        return self._env
    def __len__(self): return self.length
    def __getitem__(self, idx):
        env = self._get_env()
        with env.begin(write=False) as txn: raw = txn.get(f"sample_{idx:010d}".encode())
        ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).long()
        S = self.max_seq_len
        return ids[:S] if ids.shape[0] >= S else F.pad(ids, (0, S - ids.shape[0]))

def build_lmdb(jsonl_path, db_path, tokenizer, max_seq_len, map_size_gb=80.0):
    import lmdb, numpy as np
    print(f"\n  [*] Building LMDB...", flush=True)
    t0 = time.time(); texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1_000_000 == 0 and i > 0: print(f"      read {i:,} lines...", flush=True)
            line = line.strip()
            if not line: continue
            try: obj = json.loads(line)
            except: continue
            text = obj.get("text") or obj.get("content") or obj.get("passage", "")
            if len(text) >= 30: texts.append(text)
    print(f"      {len(texts):,} texts in {time.time()-t0:.0f}s", flush=True)
    t1 = time.time(); BATCH = 1024; PAD_ID = tokenizer.pad_id
    env = lmdb.open(db_path, map_size=int(map_size_gb * (1024**3))); txn = env.begin(write=True)
    count = 0; total_tokens = 0; total_batches = (len(texts) + BATCH - 1) // BATCH
    for batch_idx in range(0, len(texts), BATCH):
        batch = texts[batch_idx:batch_idx+BATCH]; batch_num = batch_idx // BATCH + 1
        enc = tokenizer.tokenizer(batch, max_length=max_seq_len, truncation=True, padding="max_length", return_tensors="np", return_attention_mask=False)
        ids_np = enc.input_ids.astype(np.int32)
        for j in range(ids_np.shape[0]):
            row = ids_np[j]; non_pad = int(np.sum(row != PAD_ID))
            if non_pad < 10: continue
            txn.put(f"sample_{count:010d}".encode(), row.tobytes()); count += 1; total_tokens += non_pad
        if batch_num % 100 == 0 or batch_num == total_batches:
            pct = batch_num / total_batches * 100
            print(f"      [{pct:5.1f}%] {count:,} samples | {total_tokens/1e6:.0f}M tok", flush=True)
        if count % 500_000 < BATCH and count >= 500_000: txn.commit(); txn = env.begin(write=True)
    txn.put(b"__len__", struct.pack("<Q", count)); txn.put(b"__total_tokens__", struct.pack("<Q", total_tokens))
    txn.commit(); env.close()
    print(f"  [✓] LMDB: {count:,} samples, {total_tokens/1e6:.1f}M tokens in {time.time()-t1:.0f}s", flush=True)

# ── LR Schedule ──
def get_lr(step, cfg):
    if step < cfg.warmup_steps: return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min((step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1), 1.0)
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))

# ── Checkpoint Manager ──
class CheckpointManager:
    def __init__(self, save_dir, keep_last=5):
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True); self.keep_last = keep_last

    def save(self, model, optimizer, step, loss, cfg, scaler=None):
        path = self.save_dir / f"nord_v4_step_{step:07d}.pt"
        m = model.module if hasattr(model, 'module') else model
        d = {"step": step, "loss": loss, "version": "v4.2", "model_state_dict": m.state_dict(),
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
        filtered = {k: v for k, v in ckpt["model_state_dict"].items() if "_v_mem_state" not in k and "_i_syn_state" not in k}
        m.load_state_dict(filtered, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in ckpt: scaler.load_state_dict(ckpt["scaler_state_dict"])
        print(f"  [✓] Resumed at step {ckpt['step']:,} (loss={ckpt.get('loss', '?')})", flush=True)
        return ckpt["step"]

    def save_final(self, model, cfg):
        path = self.save_dir / "nord_v4_final.pt"
        m = model.module if hasattr(model, 'module') else model
        d = {"version": "v4.2", "model_state_dict": m.state_dict(),
             "config": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_") and k != "dtype"}}
        if USE_TPU: xm.save(d, str(path))
        else: torch.save(d, path)
        print(f"  [⭐] Final model: {path}", flush=True)

# ── Training ──
def train(dataset_path, model_dir, lr_override=None, continued=False):
    device = get_device()

    # Determine LR: continued pretraining uses lower LR
    base_lr = 2e-4
    if continued:
        base_lr = 5e-5
        print("  [*] Continued pretraining mode: LR=5e-5, warmup=200", flush=True)
    if lr_override is not None:
        base_lr = lr_override
        print(f"  [*] LR override: {base_lr}", flush=True)

    warmup = 200 if continued else 1000

    cfg = NordConfig(
        device=str(device), dtype=torch.bfloat16 if USE_TPU else torch.float16,
        d_model=1536, n_heads=24, d_ff=4096, n_clusters=128, max_seq_len=192,
        sensory_layers=3, association_layers=3, executive_layers=4,
        T=8, T_slow=2, persistent_mem=False,
        n_experts=4, top_k_experts=2,
        memory_size=256, memory_tau_mem=0.99, memory_n_read_heads=8,
        target_spike_rate=0.03, spike_loss_weight=0.5,
        v_threshold=0.12, tau_mem=0.9, lif_freeze_steps=1000,
        gradient_checkpointing=False,
        batch_size=1, grad_accum=64, lr=base_lr, min_lr=1e-5,
        warmup_steps=warmup, max_steps=50_000,
        save_every=1000, log_every=10,
    )

    print(flush=True); print("═" * 60, flush=True)
    print("  PROJECT NORD v4.2 — 700M SNN Training", flush=True); print("═" * 60, flush=True)

    # ── Auto-adjust batch size ──
    if USE_TPU:
        print(f"  Device:    TPU ({device})", flush=True)
        print(f"  Precision: bfloat16 (native)", flush=True)
        cfg.batch_size = 8; cfg.grad_accum = 4
        print(f"  [Auto] batch=8, accum=4 (TPU)", flush=True)
    elif torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        n_gpus = torch.cuda.device_count()
        total_vram = vram * n_gpus
        print(f"  GPU:  {torch.cuda.get_device_name()} ({vram:.1f}GB)" + (f" × {n_gpus} = {total_vram:.1f}GB total" if n_gpus > 1 else ""), flush=True)
        if vram < 16:
            print("  [ERROR] Need ≥16GB VRAM per GPU!", flush=True); sys.exit(1)

    print(f"  Arch:      d={cfg.d_model}, h={cfg.n_heads}, ff={cfg.d_ff}", flush=True)
    print(f"  Zones:     S({cfg.sensory_layers})→A({cfg.association_layers},MoE)→M→E({cfg.executive_layers})", flush=True)

    tokenizer = NordTokenizer(cfg)
    db_path = str(Path(dataset_path).with_suffix("")) + "_lmdb"
    if not Path(db_path).exists(): build_lmdb(dataset_path, db_path, tokenizer, cfg.max_seq_len)
    dataset = LMDBDataset(db_path, cfg.max_seq_len)

    print(f"\n  [*] Building Nord v4 model...", flush=True)
    model = NordModel(cfg).to(device)
    print(f"  [✓] {model.count_params()}", flush=True)

    # Multi-GPU (CUDA only)
    n_gpus = 1
    if not USE_TPU and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        from torch.nn.parallel import DataParallel
        n_gpus = torch.cuda.device_count()
        print(f"  [⚡] {n_gpus} GPUs → DataParallel", flush=True)
        model = DataParallel(model)

    # ── Smart VRAM auto-tuning: probe batch sizes to fill 85% VRAM ──
    if not USE_TPU and torch.cuda.is_available():
        TARGET_VRAM_PCT = 0.85  # Fill 85% of VRAM
        EFF_BATCH_TARGET = 32   # Target effective batch size

        vram_total = torch.cuda.get_device_properties(0).total_memory
        vram_after_model = torch.cuda.memory_allocated()
        vram_free = vram_total - vram_after_model
        print(f"\n  [*] Smart VRAM auto-tuning...", flush=True)
        print(f"      Total VRAM (per GPU): {vram_total/(1024**3):.1f}GB", flush=True)
        print(f"      Model + optimizer:    {vram_after_model/(1024**3):.1f}GB", flush=True)
        print(f"      Available:            {vram_free/(1024**3):.1f}GB", flush=True)
        print(f"      Target fill:          {TARGET_VRAM_PCT:.0%}", flush=True)

        # Probe increasing batch sizes with a dummy forward+backward
        best_batch = 1
        test_seq_len = cfg.max_seq_len
        model.train()

        # Create temporary optimizer for probing
        temp_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
        temp_scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == torch.float16))

        for test_batch in [1, 2, 3, 4, 6, 8, 10, 12, 16]:
            # For DataParallel, total batch = test_batch, split across GPUs
            # Each GPU gets test_batch // n_gpus, need at least 1 per GPU
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

                if pct <= TARGET_VRAM_PCT:
                    best_batch = test_batch
                else:
                    # Exceeded target, stop probing
                    break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    print(f"      batch={test_batch:>2} → OOM!", flush=True)
                    break
                else:
                    raise

        # Clean up probe state
        del temp_optim, temp_scaler
        torch.cuda.empty_cache()

        # Re-initialize model weights since probe corrupted them
        model_to_reinit = model.module if hasattr(model, 'module') else model
        model_to_reinit.__init__(cfg)
        model_to_reinit.to(device)
        if hasattr(model, 'module'):
            # Re-wrap in DataParallel
            model = DataParallel(model_to_reinit)

        cfg.batch_size = best_batch
        cfg.grad_accum = max(1, EFF_BATCH_TARGET // best_batch)
        eff = cfg.batch_size * cfg.grad_accum

        print(f"\n  [✓] Auto-tuned: batch={cfg.batch_size}, accum={cfg.grad_accum}, effective={eff}", flush=True)
        print(f"      VRAM utilization: ~{TARGET_VRAM_PCT:.0%} target", flush=True)

    # Rebuild dataloader with tuned batch size
    if USE_TPU:
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, drop_last=True)
    else:
        dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True)

    print(f"  Eff batch: {cfg.batch_size}×{cfg.grad_accum}={cfg.batch_size*cfg.grad_accum}", flush=True)
    print(f"  LR:        {cfg.lr}→{cfg.min_lr} (cosine, {cfg.warmup_steps} warmup)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler = None
    if not USE_TPU and cfg.dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    ckpt_mgr = CheckpointManager(model_dir)
    start_step = ckpt_mgr.load(model, optimizer, device, scaler)

    model.train(); data_iter = iter(dataloader)
    running_loss = 0.0; running_spike_loss = 0.0; tokens_seen = 0; t_start = time.time()

    print(f"\n  {'─'*55}", flush=True)
    print(f"  Start step {start_step:,} | {len(dataset):,} samples | {'TPU' if USE_TPU else 'CUDA'}", flush=True)
    print(f"  Ctrl+C = save & stop", flush=True)
    print(f"  {'─'*55}\n", flush=True)

    try:
        for step in range(start_step, cfg.max_steps):
            accum_loss = 0.0; accum_spike_loss = 0.0; stats = {}
            for _ in range(cfg.grad_accum):
                try: input_ids = next(data_iter)
                except StopIteration: data_iter = iter(dataloader); input_ids = next(data_iter)
                input_ids = input_ids.to(device)

                if USE_TPU:
                    with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                        logits, stats = model(input_ids)
                        ce_loss = F.cross_entropy(logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size),
                                                  input_ids[:, 1:].contiguous().reshape(-1), ignore_index=tokenizer.pad_id)
                        spike_loss = stats.get("spike_loss", torch.tensor(0.0, device=device))
                        if isinstance(spike_loss, torch.Tensor):
                            if spike_loss.dim() > 0: spike_loss = spike_loss.mean()
                        else:
                            spike_loss = torch.tensor(float(spike_loss), device=device)
                        moe_lb = stats.get("moe_lb_loss", torch.tensor(0.0, device=device))
                        if isinstance(moe_lb, torch.Tensor):
                            if moe_lb.dim() > 0: moe_lb = moe_lb.mean()
                        else:
                            moe_lb = torch.tensor(float(moe_lb), device=device)
                        loss = (ce_loss + spike_loss + 0.01 * moe_lb) / cfg.grad_accum
                    loss.backward()
                else:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(cfg.dtype == torch.float16)):
                        logits, stats = model(input_ids)
                        ce_loss = F.cross_entropy(logits[:, :-1, :].contiguous().reshape(-1, cfg.vocab_size),
                                                  input_ids[:, 1:].contiguous().reshape(-1), ignore_index=tokenizer.pad_id)
                        spike_loss = stats.get("spike_loss", torch.tensor(0.0, device=device))
                        if isinstance(spike_loss, torch.Tensor):
                            if spike_loss.dim() > 0: spike_loss = spike_loss.mean()
                        else:
                            spike_loss = torch.tensor(float(spike_loss), device=device)
                        moe_lb = stats.get("moe_lb_loss", torch.tensor(0.0, device=device))
                        if isinstance(moe_lb, torch.Tensor):
                            if moe_lb.dim() > 0: moe_lb = moe_lb.mean()
                        else:
                            moe_lb = torch.tensor(float(moe_lb), device=device)
                        loss = (ce_loss + spike_loss + 0.01 * moe_lb) / cfg.grad_accum
                    scaler.scale(loss).backward()

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
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)

            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups: pg["lr"] = lr
            running_loss += accum_loss; running_spike_loss += accum_spike_loss

            # Logging
            if step % cfg.log_every == 0 and step > start_step:
                avg = running_loss / cfg.log_every; avg_sp = running_spike_loss / cfg.log_every
                tps = tokens_seen / (time.time() - t_start) / 1000
                # Handle both tensor and float stats (DataParallel returns averaged tensors)
                sp = stats.get("sparsity", 0)
                if isinstance(sp, torch.Tensor): sp = sp.mean().item()
                mem_r = stats.get("memory_spike_rate", None)
                if isinstance(mem_r, torch.Tensor): mem_r = mem_r.mean().item()
                mem_s = f" | mem={mem_r:.3f}" if mem_r is not None else ""
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                dev = " | TPU" if USE_TPU else (f" | VRAM {torch.cuda.memory_allocated()/(1024**3):.1f}G" if torch.cuda.is_available() else "")
                print(f"  step {step:>7,} │ loss {avg:.4f} │ spike_L {avg_sp:.4f} │ lr {lr:.1e} │ grad {gn:.1f} │ sparsity {sp:.0%} │ {tps:.1f}k tok/s{mem_s}{dev}", flush=True)
                running_loss = 0.0; running_spike_loss = 0.0

            if step % 100 == 0 and step > start_step:
                print(f"  {'·'*50}", flush=True)
                # Handle spike_rates as tensor (DataParallel) or list
                sr = stats.get("spike_rates_tensor", stats.get("spike_rates", []))
                if isinstance(sr, torch.Tensor):
                    sr = sr.float()
                    if sr.dim() > 1: sr = sr.mean(dim=0)  # average across DataParallel replicas
                    sr = sr.tolist()
                if sr:
                    ns = cfg.sensory_layers + 1; na = cfg.association_layers
                    print(f"    Sensory spike rates:     {[f'{r:.4f}' for r in sr[:ns]]}", flush=True)
                    print(f"    Association spike rates:  {[f'{r:.4f}' for r in sr[ns:ns+na]]}", flush=True)
                    print(f"    Executive spike rates:    {[f'{r:.4f}' for r in sr[ns+na:]]}", flush=True)
                gate = stats.get("gate_activity"); mix = stats.get("memory_mix")
                if isinstance(gate, torch.Tensor): gate = gate.mean().item()
                if isinstance(mix, torch.Tensor): mix = mix.mean().item()
                if gate is not None: print(f"    Memory gate={gate:.4f} mix={mix:.4f}", flush=True)
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
    parser.add_argument("--tpu", action="store_true", help="Force TPU backend")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate (e.g. 5e-5 for continued pretraining)")
    parser.add_argument("--continued", action="store_true", help="Continued pretraining mode: auto LR=5e-5, shorter warmup")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("  PROJECT NORD v4.2 — Brain-Inspired SNN Training", flush=True)
    print("=" * 60, flush=True)
    detect_backend(force_tpu=args.tpu)

    if args.dataset: dataset_path = args.dataset
    else:
        d = "train_data.jsonl"
        print(f"\n  Dataset? (Enter = {d})", flush=True)
        inp = input("  Dataset: ").strip(); dataset_path = inp if inp else d
    if not Path(dataset_path).exists(): print(f"  [✗] Not found: {dataset_path}", flush=True); sys.exit(1)

    if args.model_dir: model_dir = args.model_dir
    else:
        d = "nord_v4_700m"
        print(f"\n  Model dir? (Enter = {d})", flush=True)
        inp = input("  Model dir: ").strip(); model_dir = inp if inp else d

    train(dataset_path, model_dir, lr_override=args.lr, continued=args.continued)

if __name__ == "__main__":
    main()