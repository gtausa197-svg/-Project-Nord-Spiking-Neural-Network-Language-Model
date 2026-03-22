"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v4.2 — Training Script (700M)                     ║
║                                                                        ║
║  Usage:                                                                ║
║      python train_nord_700m.py                                         ║
║                                                                        ║
║  v4.2 (700M) — Scaling test on L40/A100                                ║
║      - Auxiliary spike loss (homeostatic regulation)                    ║
║      - MoE routing stats (expert load, entropy)                        ║
║      - Memory cortex monitoring                                        ║
║      - Zone-aware logging (sensory/association/executive)               ║
║      - Combined loss: L_total = L_CE + λ_spike * L_spike + λ_lb * L_lb║
║                                                                        ║
║  Hardware:                                                              ║
║      - RTX 5070 (8GB) — batch=2, ~3GB VRAM                             ║
║      - RTX 3090/4090 (24GB) — batch=4                                  ║
║      - A100/L40 (48-80GB) — batch=8-16                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DataParallel

# Use local nord_core_700m (fix for gradient checkpoint + MoE metadata mismatch)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nord_core_700m import NordConfig, NordModel


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────

class NordTokenizer:
    def __init__(self, cfg: NordConfig):
        from transformers import AutoTokenizer

        print(f"  [*] Loading Llama-3.2 tokenizer...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.tokenizer_id, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_len = cfg.max_seq_len
        self.vocab_size = self.tokenizer.vocab_size
        if cfg.vocab_size < self.vocab_size:
            cfg.vocab_size = self.vocab_size

        print(f"  [✓] Tokenizer ready (vocab={self.vocab_size:,})", flush=True)

    def encode(self, text: str) -> torch.Tensor:
        enc = self.tokenizer(
            text, return_tensors="pt",
            max_length=self.max_len, truncation=True, padding="max_length",
        )
        return enc.input_ids

    def decode(self, ids) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def pad_id(self) -> int:
        return self.tokenizer.pad_token_id


# ─────────────────────────────────────────────────────────────────────────────
# LMDB DATASET
# ─────────────────────────────────────────────────────────────────────────────

class LMDBDataset(Dataset):
    def __init__(self, db_path: str, max_seq_len: int):
        import lmdb
        self.db_path = db_path
        self.max_seq_len = max_seq_len
        self._env = None

        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            raw = txn.get(b"__len__")
            self.length = struct.unpack("<Q", raw)[0]
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


def build_lmdb(jsonl_path: str, db_path: str, tokenizer: NordTokenizer,
               max_seq_len: int, map_size_gb: float = 80.0):
    import lmdb
    import numpy as np

    print(f"\n  [*] Building LMDB database (fast batch mode)...", flush=True)
    print(f"      Source:  {jsonl_path}", flush=True)
    print(f"      Target:  {db_path}", flush=True)

    # Read all texts into memory
    print(f"  [*] Reading JSONL into memory...", flush=True)
    t0 = time.time()
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1_000_000 == 0 and i > 0:
                print(f"      read {i:,} lines...", flush=True)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("content") or obj.get("passage", "")
            if len(text) >= 30:
                texts.append(text)

    print(f"      {len(texts):,} valid texts in {time.time()-t0:.0f}s", flush=True)

    # Batch tokenize
    print(f"  [*] Batch tokenizing...", flush=True)
    t1 = time.time()
    BATCH = 1024
    PAD_ID = tokenizer.pad_id

    env = lmdb.open(db_path, map_size=int(map_size_gb * (1024**3)))
    txn = env.begin(write=True)
    count = 0
    total_tokens = 0
    total_batches = (len(texts) + BATCH - 1) // BATCH

    for batch_idx in range(0, len(texts), BATCH):
        batch = texts[batch_idx : batch_idx + BATCH]
        batch_num = batch_idx // BATCH + 1

        enc = tokenizer.tokenizer(
            batch, max_length=max_seq_len, truncation=True,
            padding="max_length", return_tensors="np",
            return_attention_mask=False,
        )
        ids_np = enc.input_ids.astype(np.int32)

        for j in range(ids_np.shape[0]):
            row = ids_np[j]
            non_pad = int(np.sum(row != PAD_ID))
            if non_pad < 10:
                continue
            txn.put(f"sample_{count:010d}".encode(), row.tobytes())
            count += 1
            total_tokens += non_pad

        if batch_num % 100 == 0 or batch_num == total_batches:
            elapsed = time.time() - t1
            pct = batch_num / total_batches * 100
            eta = (elapsed / batch_num) * (total_batches - batch_num)
            print(
                f"      [{pct:5.1f}%] {count:,} samples | "
                f"{total_tokens/1e6:.0f}M tok | ETA {eta:.0f}s",
                flush=True,
            )

        if count % 500_000 < BATCH and count >= 500_000:
            txn.commit()
            txn = env.begin(write=True)

    txn.put(b"__len__", struct.pack("<Q", count))
    txn.put(b"__total_tokens__", struct.pack("<Q", total_tokens))
    txn.commit()
    env.close()

    elapsed = time.time() - t1
    print(f"\n  [✓] LMDB ready!", flush=True)
    print(f"      Samples:  {count:,}", flush=True)
    print(f"      Tokens:   {total_tokens:,} ({total_tokens/1e6:.1f}M)", flush=True)
    print(f"      Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# LR SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, cfg: NordConfig) -> float:
    """Warmup → cosine decay to min_lr.
    
    Phase 1 (0 → warmup_steps): linear warmup from 0 → lr
    Phase 2 (warmup_steps → max_steps): cosine decay from lr → min_lr
    """
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    
    # Cosine decay phase
    decay_steps = cfg.max_steps - cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(decay_steps, 1)
    progress = min(progress, 1.0)  # clamp at 1.0
    
    # Cosine annealing: lr → min_lr
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, save_dir: str, keep_last: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(self, model, optimizer, scaler, step, loss, cfg):
        path = self.save_dir / f"nord_v4_step_{step:07d}.pt"
        # Handle DataParallel: save inner model
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            "step": step, "loss": loss,
            "version": "v4.1",
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("_") and k != "dtype"},
        }, path)

        latest = self.save_dir / "nord_v4_latest.pt"
        if latest.exists():
            latest.unlink()
        shutil.copy2(path, latest)

        ckpts = sorted(self.save_dir.glob("nord_v4_step_*.pt"),
                        key=lambda p: p.stat().st_mtime)
        for old in ckpts[:max(0, len(ckpts) - self.keep_last)]:
            old.unlink()

        print(f"  [💾] Saved: {path.name} (loss={loss:.4f})", flush=True)

    def load(self, model, optimizer, scaler, device) -> int:
        latest = self.save_dir / "nord_v4_latest.pt"
        if not latest.exists():
            ckpts = sorted(self.save_dir.glob("nord_v4_step_*.pt"))
            latest = ckpts[-1] if ckpts else None
        if latest is None:
            return 0

        print(f"  [*] Resuming from: {latest.name}", flush=True)
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        # Handle DataParallel: load into inner model
        model_to_load = model.module if hasattr(model, 'module') else model
        # Filter out persistent LIF state buffers — they resize with batch
        state = ckpt["model_state_dict"]
        filtered = {k: v for k, v in state.items()
                    if "_v_mem_state" not in k and "_i_syn_state" not in k}
        model_to_load.load_state_dict(filtered, strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt["step"]
        print(f"  [✓] Resumed at step {step:,} (loss={ckpt.get('loss', '?')})", flush=True)
        return step

    def save_final(self, model, cfg):
        path = self.save_dir / "nord_v4_final.pt"
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save({
            "version": "v4.1",
            "model_state_dict": model_to_save.state_dict(),
            "config": {k: v for k, v in cfg.__dict__.items()
                       if not k.startswith("_") and k != "dtype"},
        }, path)
        print(f"  [⭐] Final model: {path}", flush=True)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train(dataset_path: str, model_dir: str):
    # ── Config — v4.2 (700M) ──
    cfg = NordConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,

        d_model=1536,
        n_heads=24,
        d_ff=4096,
        n_clusters=128,
        max_seq_len=192,  # 23.5GB без checkpoint: 384 OOM → 192 вміщається

        sensory_layers=3,
        association_layers=3,
        executive_layers=4,

        T=8,
        T_slow=2,
        persistent_mem=False,  # FIX для градієнтів

        n_experts=4,
        top_k_experts=2,

        memory_size=256,
        memory_tau_mem=0.99,
        memory_n_read_heads=8,

        target_spike_rate=0.03,
        spike_loss_weight=0.5,

        v_threshold=0.12,
        tau_mem=0.9,
        lif_freeze_steps=1000,

        gradient_checkpointing=False,  # OFF: MoE+SNN несумісні з checkpoint (219 vs 220 metadata)

        batch_size=1,
        grad_accum=64,
        lr=2e-4,
        warmup_steps=1000,
        max_steps=50_000,
        save_every=1000,
        log_every=10,
    )
    

    print(flush=True)
    print("═" * 60, flush=True)
    print("  PROJECT NORD v4.2 — 700M SNN Training", flush=True)
    print("═" * 60, flush=True)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"  GPU:            {torch.cuda.get_device_name()}", flush=True)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  VRAM:           {vram:.1f} GB" + (f" × {n_gpus} GPUs" if n_gpus > 1 else ""), flush=True)

        # Auto-adjust batch size for 700M SNN
        if vram >= 80:
            cfg.batch_size = 8
            cfg.grad_accum = 4
            print(f"  [Auto] batch=8, accum=4 (A100 80GB)", flush=True)
        elif vram >= 40:
            cfg.batch_size = 4
            cfg.grad_accum = 8
            print(f"  [Auto] batch=4, accum=8 (L40 48GB)", flush=True)
        elif vram >= 20:
            cfg.batch_size = 1
            cfg.grad_accum = 32
            print(f"  [Auto] batch=1, accum=32 (24GB VRAM)", flush=True)
        else:
            print(f"  [ERROR] 700M model needs minimum 24GB VRAM!", flush=True)
            sys.exit(1)
    else:
        print("  CPU mode (not recommended!)", flush=True)

    print(f"  Architecture:   d={cfg.d_model}, heads={cfg.n_heads}, clusters={cfg.n_clusters}", flush=True)
    print(f"  Zones:          Sensory({cfg.sensory_layers}) → Association({cfg.association_layers},MoE) → Memory → Executive({cfg.executive_layers})", flush=True)
    print(f"  MoE:            {cfg.n_experts} experts, top-{cfg.top_k_experts}", flush=True)
    print(f"  Memory:         {cfg.memory_size} neurons (τ={cfg.memory_tau_mem})", flush=True)
    print(f"  Spike target:   {cfg.target_spike_rate:.0%} firing rate (λ={cfg.spike_loss_weight})", flush=True)
    print(f"  Effective batch: {cfg.batch_size} × {cfg.grad_accum} = {cfg.batch_size * cfg.grad_accum}", flush=True)
    print(f"  LR:             {cfg.lr} → {cfg.min_lr} (cosine decay, {cfg.warmup_steps} warmup)", flush=True)
    print(f"  Max steps:      {cfg.max_steps:,}", flush=True)
    print(f"  Dataset:        {dataset_path}", flush=True)
    print(f"  Model dir:      {model_dir}", flush=True)
    print(flush=True)

    # ── Tokenizer ──
    tokenizer = NordTokenizer(cfg)

    # ── LMDB ──
    db_path = str(Path(dataset_path).with_suffix("")) + "_lmdb"
    if not Path(db_path).exists():
        build_lmdb(dataset_path, db_path, tokenizer, cfg.max_seq_len)

    dataset = LMDBDataset(db_path, cfg.max_seq_len)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True, persistent_workers=True,
    )

    # ── Model ──
    print(f"\n  [*] Building Nord v4 model...", flush=True)
    model = NordModel(cfg).to(cfg.device)
    print(f"  [✓] {model.count_params()}", flush=True)

    # ── Multi-GPU support ──
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        print(f"  [⚡] {n_gpus} GPUs detected! Using DataParallel", flush=True)
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            vram_i = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"       GPU {i}: {name} ({vram_i:.1f} GB)", flush=True)
        model = DataParallel(model)
        # Scale batch size by number of GPUs
        cfg.batch_size = cfg.batch_size * n_gpus
        cfg.grad_accum = max(1, cfg.grad_accum // n_gpus)
        print(f"  [Auto] Scaled: batch={cfg.batch_size}, accum={cfg.grad_accum} "
              f"(effective={cfg.batch_size * cfg.grad_accum})", flush=True)

    # ── Gradient checkpointing for OOM prevention ──
    if cfg.gradient_checkpointing:
        print(f"  [*] Gradient checkpointing: ON (saves VRAM)", flush=True)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        print(f"  [*] Model VRAM: {allocated:.2f} GB", flush=True)

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.dtype == torch.float16))

    # ── Checkpoints ──
    ckpt_mgr = CheckpointManager(model_dir)
    start_step = ckpt_mgr.load(model, optimizer, scaler, cfg.device)

    # ── Training loop ──
    model.train()
    data_iter = iter(dataloader)
    running_loss = 0.0
    running_spike_loss = 0.0
    tokens_seen = 0
    t_start = time.time()

    print(f"\n  {'─' * 55}", flush=True)
    print(f"  Starting from step {start_step:,}  |  {len(dataset):,} samples", flush=True)
    print(f"  Ctrl+C = stop (model will be saved!)", flush=True)
    print(f"  {'─' * 55}\n", flush=True)

    try:
        for step in range(start_step, cfg.max_steps):
            accum_loss = 0.0
            accum_spike_loss = 0.0
            stats = {}

            for _ in range(cfg.grad_accum):
                try:
                    input_ids = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    input_ids = next(data_iter)

                input_ids = input_ids.to(cfg.device, non_blocking=True)

                with autocast(device_type="cuda", dtype=torch.float16,
                              enabled=(cfg.dtype == torch.float16)):
                    logits, stats = model(input_ids)

                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = input_ids[:, 1:].contiguous()

                    # Main loss: cross entropy
                    ce_loss = F.cross_entropy(
                        shift_logits.reshape(-1, cfg.vocab_size),
                        shift_labels.reshape(-1),
                        ignore_index=tokenizer.pad_id,
                    )

                    # v4.1: Auxiliary spike loss
                    spike_loss = stats.get("spike_loss", torch.tensor(0.0))
                    if isinstance(spike_loss, torch.Tensor):
                        spike_loss = spike_loss.to(ce_loss.device)
                    else:
                        spike_loss = torch.tensor(0.0, device=ce_loss.device)

                    # v4.1: MoE load balance loss
                    moe_lb_loss = stats.get("moe_lb_loss", torch.tensor(0.0))
                    if isinstance(moe_lb_loss, torch.Tensor):
                        moe_lb_loss = moe_lb_loss.to(ce_loss.device)
                    else:
                        moe_lb_loss = torch.tensor(0.0, device=ce_loss.device)

                    # Combined loss: CE + spike homeostasis + MoE load balance
                    loss = (ce_loss + spike_loss + 0.01 * moe_lb_loss) / cfg.grad_accum

                scaler.scale(loss).backward()
                accum_loss += ce_loss.item() / cfg.grad_accum
                accum_spike_loss += spike_loss.item() / cfg.grad_accum
                tokens_seen += input_ids.numel()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # LR schedule
            lr = get_lr(step, cfg)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            running_loss += accum_loss
            running_spike_loss += accum_spike_loss

            if step % cfg.log_every == 0 and step > start_step:
                avg = running_loss / cfg.log_every
                avg_spike = running_spike_loss / cfg.log_every
                elapsed = time.time() - t_start
                tps = tokens_seen / elapsed / 1000 if elapsed > 0 else 0
                sp = stats.get("sparsity", 0)

                # VRAM monitoring
                vram_used = ""
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.memory_allocated() / (1024**3)
                    vram_used = f" | VRAM {vram_gb:.1f}G"

                # MoE routing info
                moe_info = ""
                entropy = stats.get("moe_route_entropy", None)
                if entropy is not None:
                    moe_info = f" | MoE H={entropy:.2f}"

                # Memory info
                mem_info = ""
                mem_rate = stats.get("memory_spike_rate", None)
                if mem_rate is not None:
                    mem_info = f" | mem={mem_rate:.3f}"

                print(
                    f"  step {step:>7,} │ "
                    f"loss {avg:.4f} │ "
                    f"spike_L {avg_spike:.4f} │ "
                    f"lr {lr:.1e} │ "
                    f"grad {grad_norm:.1f} │ "
                    f"sparsity {sp:.0%} │ "
                    f"{tps:.1f}k tok/s"
                    f"{moe_info}{mem_info}{vram_used}",
                    flush=True,
                )
                running_loss = 0.0
                running_spike_loss = 0.0

            # Detailed stats every 100 steps
            if step % 100 == 0 and step > start_step:
                print(f"  {'·' * 50}", flush=True)
                # Spike rates per zone
                spike_rates = stats.get("spike_rates", [])
                if spike_rates:
                    s_rates = spike_rates[:cfg.sensory_layers + 1]
                    a_rates = spike_rates[cfg.sensory_layers + 1:
                                          cfg.sensory_layers + 1 + cfg.association_layers]
                    e_rates = spike_rates[cfg.sensory_layers + 1 + cfg.association_layers:]

                    print(f"    Sensory spike rates:     {[f'{r:.4f}' for r in s_rates]}", flush=True)
                    print(f"    Association spike rates:  {[f'{r:.4f}' for r in a_rates]}", flush=True)
                    print(f"    Executive spike rates:    {[f'{r:.4f}' for r in e_rates]}", flush=True)

                # Expert load balance
                loads = [stats.get(f"expert_{e}_load", 0) for e in range(cfg.n_experts)]
                if any(l > 0 for l in loads):
                    print(f"    Expert loads: {[f'{l:.2f}' for l in loads]}", flush=True)

                # Memory stats
                gate = stats.get("gate_activity", None)
                mix = stats.get("memory_mix", None)
                if gate is not None:
                    print(f"    Memory gate={gate:.4f} mix={mix:.4f}", flush=True)

                print(f"  {'·' * 50}", flush=True)

            if step > 0 and step % cfg.save_every == 0:
                ckpt_mgr.save(model, optimizer, scaler, step, accum_loss, cfg)

    except KeyboardInterrupt:
        print(f"\n\n  [⏸] Stopped at step {step:,}", flush=True)
        ckpt_mgr.save(model, optimizer, scaler, step, accum_loss, cfg)
        print(f"  To resume — just run the script again.", flush=True)

    ckpt_mgr.save_final(model, cfg)

    print(f"\n  {'═' * 55}", flush=True)
    print(f"  Training complete!", flush=True)
    print(f"  Model saved in: {model_dir}", flush=True)
    print(f"  {'═' * 55}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60, flush=True)
    print("  PROJECT NORD v4 — Brain-Inspired SNN Training", flush=True)
    print("=" * 60, flush=True)

    default_data = "train_data.jsonl"
    print(f"\n  Dataset path? (JSONL file)", flush=True)
    print(f"  (Enter = {default_data})", flush=True)
    data_input = input("  Dataset: ").strip()
    dataset_path = data_input if data_input else default_data

    if not Path(dataset_path).exists():
        print(f"\n  [✗] File not found: {dataset_path}", flush=True)
        sys.exit(1)

    default_model = "nord_v4_700m"
    print(f"\n  Model save directory?", flush=True)
    print(f"  (Enter = {default_model})", flush=True)
    model_input = input("  Model dir: ").strip()
    model_dir = model_input if model_input else default_model

    train(dataset_path, model_dir)


if __name__ == "__main__":
    main()