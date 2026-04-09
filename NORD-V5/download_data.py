"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD — Завантаження датасетів                          ║
║                                                                        ║
║  Просто запусти:  python download_data.py                              ║
║                                                                        ║
║  Датасети для різних фаз навчання:                                     ║
║    1. FineWeb-Edu     — загальні освітні тексти (база)                  ║
║    2. OpenWebMath     — математика і reasoning                         ║
║    3. The Stack v2    — код (Python, JS, C++ та інші)                  ║
║    4. peS2o           — наукові статті                                  ║
║    5. OpenHermes 2.5  — інструкції (chat/assistant формат)             ║
║    6. SlimPajama      — різноманітний веб-текст                        ║
║    7. Wikipedia       — енциклопедичні знання                          ║
║    8. Cosmopedia      — синтетичні підручники                          ║
║                                                                        ║
║  Потрібно:  pip install datasets tqdm                                  ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import json, os, sys, time

DATASETS = {
    "1": {
        "name": "FineWeb-Edu (освітні тексти)",
        "desc": "Високоякісні освітні тексти. Найкраще для базового навчання.",
        "hf_id": "HuggingFaceFW/fineweb-edu", "hf_name": "sample-10BT",
        "split": "train", "field": "text", "gb": 40,
        "phase": "Фаза 1 — Базова мова",
    },
    "2": {
        "name": "OpenWebMath (математика)",
        "desc": "Математика: формули, докази, задачі. Для reasoning.",
        "hf_id": "open-web-math/open-web-math", "hf_name": None,
        "split": "train", "field": "text", "gb": 15,
        "phase": "Фаза 2 — Reasoning і математика",
    },
    "3": {
        "name": "StarCoder Data (код)",
        "desc": "Код на Python, JS, C++, Java та інших мовах.",
        "hf_id": "bigcode/starcoderdata", "hf_name": None,
        "split": "train", "field": "content", "gb": 20,
        "phase": "Фаза 3 — Програмування",
    },
    "4": {
        "name": "peS2o (наукові статті)",
        "desc": "Наукові papers від Semantic Scholar.",
        "hf_id": "allenai/peS2o", "hf_name": "v2",
        "split": "train", "field": "text", "gb": 20,
        "phase": "Фаза 4 — Наукові тексти",
    },
    "5": {
        "name": "OpenHermes 2.5 (інструкції)",
        "desc": "Chat/Assistant формат. Перетворює base model в chat bot.",
        "hf_id": "teknium/OpenHermes-2.5", "hf_name": None,
        "split": "train", "field": "conversations", "gb": 2,
        "phase": "Фаза 5 — Інструкції (chat)", "is_chat": True,
    },
    "6": {
        "name": "SlimPajama (різноманітний текст)",
        "desc": "Змішаний: веб, книги, Wikipedia, GitHub.",
        "hf_id": "cerebras/SlimPajama-627B", "hf_name": None,
        "split": "train", "field": "text", "gb": 30,
        "phase": "Альтернатива — Різноманітний текст",
    },
    "7": {
        "name": "Wikipedia (енциклопедія)",
        "desc": "Вся англійська Wikipedia. Чисті факти.",
        "hf_id": "wikimedia/wikipedia", "hf_name": "20231101.en",
        "split": "train", "field": "text", "gb": 6,
        "phase": "Додаток — Енциклопедичні знання",
    },
    "8": {
        "name": "Cosmopedia (синтетичні підручники)",
        "desc": "AI-згенеровані освітні тексти у форматі підручників.",
        "hf_id": "HuggingFaceTB/cosmopedia", "hf_name": None,
        "split": "train", "field": "text", "gb": 15,
        "phase": "Додаток — Синтетичні освітні тексти",
    },
}

def fmt(b):
    for u in ["B","KB","MB","GB","TB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} PB"

def format_chat(convs):
    if isinstance(convs, str): return convs
    parts = []
    for m in convs:
        role = m.get("from", m.get("role", "user"))
        text = m.get("value", m.get("content", ""))
        if role in ("system","human","user"): parts.append(f"User: {text}")
        elif role in ("gpt","assistant"): parts.append(f"Assistant: {text}")
    return "\n".join(parts)

def download_one(ds, save_dir, target_gb=None):
    if target_gb is None: target_gb = ds["gb"]
    target_bytes = int(target_gb * (1024**3))
    safe = ds["hf_id"].split("/")[-1].replace("-","_").lower()
    path = os.path.join(save_dir, f"{safe}.jsonl")

    print(f"\n  {'═'*55}")
    print(f"  📦 {ds['name']}")
    print(f"  📁 {path}")
    print(f"  🎯 {target_gb:.0f} GB")
    print(f"  {'═'*55}")

    os.makedirs(save_dir, exist_ok=True)
    written = 0; count = 0; mode = "w"

    if os.path.exists(path):
        sz = os.path.getsize(path)
        if sz >= target_bytes:
            print(f"  [✓] Вже є! ({fmt(sz)})"); return path
        if sz > 0:
            written = sz
            with open(path,"r",encoding="utf-8") as f: count = sum(1 for _ in f)
            mode = "a"
            print(f"  [*] Продовжуємо з {fmt(written)} ({count:,} зразків)")

    print(f"  [*] Підключаємося до HuggingFace...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [✗] pip install datasets"); return None

    kw = {"path": ds["hf_id"], "split": ds["split"], "streaming": True}
    if ds.get("hf_name"): kw["name"] = ds["hf_name"]

    try:
        data = load_dataset(**kw)
    except Exception as e:
        print(f"  [✗] Помилка: {e}"); return None

    it = iter(data)
    is_chat = ds.get("is_chat", False)
    field = ds["field"]

    if count > 0:
        print(f"  [*] Пропускаємо {count:,} зразків...")
        for _ in range(count):
            try: next(it)
            except StopIteration: break

    print(f"  [*] Записуємо... (Ctrl+C = пауза)")
    t0 = time.time(); lp = t0; start_b = written

    try:
        with open(path, mode, encoding="utf-8") as f:
            for sample in it:
                if is_chat:
                    text = format_chat(sample.get(field, []))
                else:
                    text = sample.get(field, "")
                if not text or len(text) < 50: continue

                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                lb = len(line.encode("utf-8"))
                f.write(line); written += lb; count += 1

                now = time.time()
                if now - lp >= 2.0:
                    el = now - t0
                    spd = (written - start_b) / el if el > 0 else 0
                    pct = written / target_bytes * 100
                    fl = int(30 * min(pct,100) / 100)
                    bar = "█"*fl + "░"*(30-fl)
                    eta = (target_bytes - written) / spd if spd > 0 else 0
                    es = f"{eta/60:.0f}хв" if eta < 3600 else f"{eta/3600:.1f}год"
                    print(f"\r  [{bar}] {pct:.1f}%  {fmt(written)}/{fmt(target_bytes)}  {count:,} зр.  {fmt(int(spd))}/s  ETA {es}    ", end="", flush=True)
                    lp = now
                    if count % 10000 == 0: f.flush()

                if written >= target_bytes: break

    except KeyboardInterrupt:
        print(f"\n  [⏸] Пауза: {fmt(written)} ({count:,} зр.)")
        return path

    el = time.time() - t0
    print(f"\n  [✓] {ds['name']}: {fmt(written)} | {count:,} зр. | {el/60:.0f}хв")
    return path

def main():
    print("=" * 60)
    print("  PROJECT NORD — Завантаження датасетів")
    print("=" * 60)
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    for k, ds in DATASETS.items():
        print(f"  │ [{k}] {ds['name']:<45}│")
        print(f"  │     {ds['phase']:<41} ~{ds['gb']:>2}GB │")
    print(f"  │                                                     │")
    print(f"  │ [A] Завантажити ВСЕ (Фази 1-5)                      │")
    print(f"  │ [M] Кілька (через кому: 1,2,5)                      │")
    print(f"  └─────────────────────────────────────────────────────┘")
    print()

    choice = input("  Вибери: ").strip().upper()

    default_dir = os.path.join(os.sep, "nord_dataset")
    print(f"\n  Папка? (Enter = {default_dir})")
    di = input("  Папка: ").strip()
    save_dir = di if di else default_dir

    if choice == "A":
        for k in ["1","2","4","5","7"]: download_one(DATASETS[k], save_dir)
    elif choice == "M":
        nums = input("  Номери (через кому): ").strip()
        for k in [x.strip() for x in nums.split(",")]:
            if k in DATASETS: download_one(DATASETS[k], save_dir)
            else: print(f"  [!] Невідомий: {k}")
    elif choice in DATASETS:
        ds = DATASETS[choice]
        print(f"\n  {ds['desc']}")
        print(f"  Рекомендовано: {ds['gb']}GB")
        print(f"  Скільки GB? (Enter = {ds['gb']})")
        si = input("  GB: ").strip()
        gb = float(si) if si else ds["gb"]
        download_one(ds, save_dir, gb)
    else:
        print(f"  [!] Невідомий вибір: {choice}"); return

    print(f"\n  {'═'*55}")
    print(f"  [✓] ГОТОВО! Датасети в: {save_dir}")
    print(f"  {'═'*55}")
    print(f"\n  Як тренувати:")
    print(f"    Базове:      python train_nord_700m.py --dataset {save_dir}/fineweb_edu.jsonl")
    print(f"    Математика:  python train_nord_700m.py --dataset {save_dir}/open_web_math.jsonl --continued")
    print(f"    Код:         python train_nord_700m.py --dataset {save_dir}/starcoderdata.jsonl --continued")
    print(f"    Наука:       python train_nord_700m.py --dataset {save_dir}/pes2o.jsonl --continued")
    print(f"    Chat:        python train_nord_700m.py --dataset {save_dir}/openhermes_2.5.jsonl --continued")
    print()

if __name__ == "__main__":
    main()