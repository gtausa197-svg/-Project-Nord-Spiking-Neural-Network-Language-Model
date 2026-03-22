"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD — Fast LMDB Tokenizer                             ║
║                                                                        ║
║  Usage:                                                                ║
║      python build_lmdb.py                              (interactive)   ║
║      python build_lmdb.py --src data.jsonl             (auto)          ║
║      python build_lmdb.py --src data.jsonl --dst out_lmdb --seq 512   ║
║                                                                        ║
║  Batch tokenization with progress bar and resume support               ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import argparse, json, struct, time, os, sys

def build_lmdb(src, dst, seq_len=512, batch_size=1024):
    import lmdb
    import numpy as np
    from transformers import AutoTokenizer

    print("=" * 60, flush=True)
    print("  PROJECT NORD — Fast LMDB Tokenizer", flush=True)
    print("=" * 60, flush=True)
    print(f"  Source:   {src}", flush=True)
    print(f"  Output:   {dst}", flush=True)
    print(f"  Seq len:  {seq_len}", flush=True)
    print(f"  Batch:    {batch_size}", flush=True)
    print(flush=True)

    # ── Check if already exists ──
    if os.path.exists(dst):
        try:
            env = lmdb.open(dst, readonly=True, lock=False)
            with env.begin(write=False) as txn:
                existing = struct.unpack("<Q", txn.get(b"__len__"))[0]
                existing_tok = struct.unpack("<Q", txn.get(b"__total_tokens__"))[0]
            env.close()
            print(f"  [!] LMDB already exists: {existing:,} samples, {existing_tok/1e6:.0f}M tokens", flush=True)
            print(f"  Overwrite? (y/n, Enter = n)")
            choice = input("  > ").strip().lower()
            if choice not in ("y", "yes"):
                print("  [*] Skipped.", flush=True)
                return dst
            import shutil
            shutil.rmtree(dst)
            print("  [*] Deleted old LMDB.", flush=True)
        except:
            pass

    # ── Init tokenizer ──
    print("  [*] Loading tokenizer...", flush=True)
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    PAD_ID = tok.pad_token_id
    print(f"  [✓] Tokenizer ready (vocab={tok.vocab_size:,})", flush=True)

    # ── Read all texts ──
    print(f"\n  [1/3] Reading JSONL into memory...", flush=True)
    t0 = time.time()
    texts = []
    with open(src, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 1_000_000 == 0 and i > 0:
                print(f"      read {i:,} lines...", flush=True)
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except:
                continue
            text = obj.get("text") or obj.get("content") or obj.get("passage", "")
            if len(text) >= 30:
                texts.append(text)
    print(f"      {len(texts):,} valid texts in {time.time()-t0:.0f}s", flush=True)

    if not texts:
        print("  [✗] No valid texts found!", flush=True)
        return None

    # ── Batch tokenize ──
    print(f"\n  [2/3] Batch tokenizing {len(texts):,} texts (batch={batch_size})...", flush=True)
    t1 = time.time()

    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else ".", exist_ok=True)
    env = lmdb.open(dst, map_size=80 * (1024**3))
    txn = env.begin(write=True)

    count = 0
    total_tok = 0
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(texts), batch_size):
        batch = texts[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1

        enc = tok(
            batch,
            max_length=seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="np",
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
            total_tok += non_pad

        # Progress
        if batch_num % 100 == 0 or batch_num == total_batches:
            elapsed = time.time() - t1
            pct = batch_num / total_batches * 100
            eta = (elapsed / batch_num) * (total_batches - batch_num)
            speed = count / elapsed if elapsed > 0 else 0
            bar_len = 30
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"    [{bar}] {pct:5.1f}% | "
                f"{count:,} samples | {total_tok/1e6:.0f}M tok | "
                f"{speed:.0f} doc/s | ETA {eta:.0f}s",
                flush=True,
            )

        # Commit every 500k
        if count % 500_000 < batch_size and count >= 500_000:
            txn.commit()
            txn = env.begin(write=True)

    # Save metadata
    txn.put(b"__len__", struct.pack("<Q", count))
    txn.put(b"__total_tokens__", struct.pack("<Q", total_tok))
    txn.commit()
    env.close()

    elapsed = time.time() - t1
    print(f"\n  [3/3] Done!", flush=True)
    print(f"  {'═' * 50}", flush=True)
    print(f"    Samples:  {count:,}", flush=True)
    print(f"    Tokens:   {total_tok:,} ({total_tok/1e6:.0f}M)", flush=True)
    print(f"    Time:     {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
    print(f"    Speed:    {count/elapsed:.0f} doc/s", flush=True)
    print(f"  {'═' * 50}", flush=True)
    print(f"\n  Тепер тренуй:", flush=True)
    print(f"    python train_nord_700m.py --dataset {src}", flush=True)
    print(flush=True)
    return dst


def main():
    parser = argparse.ArgumentParser(description="Nord LMDB Tokenizer")
    parser.add_argument("--src", type=str, default=None, help="Source JSONL file")
    parser.add_argument("--dst", type=str, default=None, help="Output LMDB directory")
    parser.add_argument("--seq", type=int, default=512, help="Max sequence length (default: 512)")
    parser.add_argument("--batch", type=int, default=1024, help="Batch size (default: 1024)")
    args = parser.parse_args()

    # Interactive mode if no args
    if args.src is None:
        print("=" * 60)
        print("  PROJECT NORD — Fast LMDB Tokenizer")
        print("=" * 60)
        print()
        print("  Шлях до JSONL датасету?")
        print("  (наприклад: /nord_dataset/train_data.jsonl)")
        args.src = input("  Source: ").strip()
        if not args.src:
            print("  [✗] Потрібно вказати шлях!", flush=True)
            sys.exit(1)

    if not os.path.exists(args.src):
        print(f"  [✗] Файл не знайдено: {args.src}", flush=True)
        sys.exit(1)

    if args.dst is None:
        # Auto: same path but _lmdb suffix
        args.dst = args.src.replace(".jsonl", "") + "_lmdb"
        print(f"\n  Output LMDB? (Enter = {args.dst})")
        user_dst = input("  Output: ").strip()
        if user_dst:
            args.dst = user_dst

    print(f"\n  Sequence length? (Enter = {args.seq})")
    seq_input = input("  Seq: ").strip()
    if seq_input:
        args.seq = int(seq_input)

    print()
    build_lmdb(args.src, args.dst, args.seq, args.batch)


if __name__ == "__main__":
    main()