"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD — Крок 1: Завантаження датасету                   ║
║                                                                        ║
║  Просто запусти:                                                       ║
║      python download_data.py                                           ║
║                                                                        ║
║  Воно запитає куди зберегти і почне качати.                            ║
║  Датасет: FineWeb-Edu (високоякісні освітні тексти англійською)         ║
║  Розмір: ~40 GB тексту (JSONL формат)                                  ║
╚══════════════════════════════════════════════════════════════════════════╝

Потрібно встановити один раз:
    pip install datasets tqdm
"""

import json
import os
import sys
import time


def format_size(bytes_val: int) -> str:
    """Форматувати байти в людський вигляд."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def download():
    print("=" * 60)
    print("  PROJECT NORD — Завантаження датасету")
    print("=" * 60)
    print()

    # ── Запитати куди зберегти ──
    default_path = os.path.join("D:", os.sep, "nord_dataset", "train_data.jsonl")
    print(f"  Куди зберегти датасет?")
    print(f"  (Enter = {default_path})")
    user_path = input("  Шлях: ").strip()
    save_path = user_path if user_path else default_path

    # ── Запитати розмір ──
    print()
    print("  Скільки гігабайт завантажити?")
    print("  Рекомендовано:  10 GB — швидкий тест")
    print("                  40 GB — повне навчання")
    print(f"  (Enter = 40)")
    size_input = input("  Розмір (GB): ").strip()
    target_gb = float(size_input) if size_input else 40.0
    target_bytes = int(target_gb * (1024 ** 3))

    # Створити папку
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print()
    print(f"  📁 Зберігаємо в:  {save_path}")
    print(f"  📦 Цільовий розмір: {target_gb:.0f} GB")
    print()

    # ── Перевірити чи вже є частина файлу (для продовження) ──
    bytes_written = 0
    samples_written = 0
    mode = "w"

    if os.path.exists(save_path):
        existing_size = os.path.getsize(save_path)
        if existing_size > 0:
            print(f"  [!] Файл вже існує ({format_size(existing_size)})")
            print(f"  Продовжити дозавантаження? (y/n, Enter = y)")
            choice = input("  > ").strip().lower()
            if choice in ("", "y", "yes", "так", "д"):
                bytes_written = existing_size
                # Count existing lines
                print("  Підраховуємо існуючі рядки...")
                with open(save_path, "r", encoding="utf-8") as f:
                    samples_written = sum(1 for _ in f)
                mode = "a"
                print(f"  Продовжуємо з {samples_written:,} зразків ({format_size(bytes_written)})")
            else:
                print("  Починаємо з нуля...")

    if bytes_written >= target_bytes:
        print(f"\n  [✓] Датасет вже повний! ({format_size(bytes_written)})")
        print(f"  Тепер запускай:  python train_nord.py")
        return save_path

    # ── Завантаження ──
    print()
    print("  [*] Підключаємося до HuggingFace...")
    print("  [*] Датасет: HuggingFaceFW/fineweb-edu (sample-10BT)")
    print("      Це високоякісні освітні тексти — найкраще для навчання LLM")
    print()

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [✗] Бібліотека 'datasets' не встановлена!")
        print("      Виконай:  pip install datasets")
        sys.exit(1)

    # Stream dataset — НІКОЛИ не вантажить все в RAM
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    # Якщо продовжуємо — пропустити вже завантажені зразки
    data_iter = iter(dataset)
    if samples_written > 0:
        print(f"  [*] Пропускаємо {samples_written:,} вже завантажених зразків...")
        for _ in range(samples_written):
            try:
                next(data_iter)
            except StopIteration:
                break

    print(f"  [*] Починаємо запис... (Ctrl+C щоб зупинити, можна продовжити пізніше)")
    print()

    t_start = time.time()
    last_print = t_start

    try:
        with open(save_path, mode, encoding="utf-8") as f:
            for sample in data_iter:
                text = sample.get("text", "")
                if not text or len(text) < 50:
                    continue

                line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
                line_bytes = len(line.encode("utf-8"))
                f.write(line)

                bytes_written += line_bytes
                samples_written += 1

                # Прогрес кожні 2 секунди
                now = time.time()
                if now - last_print >= 2.0:
                    elapsed = now - t_start
                    speed = (bytes_written - (0 if mode == "w" else bytes_written)) / elapsed if elapsed > 0 else 0
                    pct = bytes_written / target_bytes * 100
                    bar_len = 30
                    filled = int(bar_len * min(pct, 100) / 100)
                    bar = "█" * filled + "░" * (bar_len - filled)

                    print(
                        f"\r  [{bar}] {pct:.1f}%  "
                        f"{format_size(bytes_written)}/{format_size(target_bytes)}  "
                        f"{samples_written:,} зразків  "
                        f"{format_size(int(speed))}/s    ",
                        end="", flush=True,
                    )
                    last_print = now

                    # Flush periodically
                    if samples_written % 10000 == 0:
                        f.flush()

                # Досягли цільового розміру
                if bytes_written >= target_bytes:
                    break

    except KeyboardInterrupt:
        print(f"\n\n  [⏸] Зупинено! Збережено {format_size(bytes_written)} ({samples_written:,} зразків)")
        print(f"  Щоб продовжити пізніше — просто запусти цей скрипт знову.")
        return save_path

    elapsed = time.time() - t_start
    print(f"\n\n  {'═' * 50}")
    print(f"  [✓] ГОТОВО!")
    print(f"  📁 Файл:       {save_path}")
    print(f"  📦 Розмір:     {format_size(bytes_written)}")
    print(f"  📝 Зразків:    {samples_written:,}")
    print(f"  ⏱  Час:        {elapsed/60:.0f} хвилин")
    print(f"  {'═' * 50}")
    print()
    print(f"  Наступний крок:")
    print(f"    python train_nord.py")
    print()

    return save_path


if __name__ == "__main__":
    download()
