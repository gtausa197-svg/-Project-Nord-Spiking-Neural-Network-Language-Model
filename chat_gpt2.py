"""
╔══════════════════════════════════════════════════════════════╗
║   GPT-2 Small (124M) — Чат для порівняння з Nord            ║
║                                                              ║
║   Запусти:  python chat_gpt2.py                              ║
║   Потрібно: pip install torch transformers                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn.functional as F
import time

def main():
    print()
    print("═" * 60)
    print("  🤖 GPT-2 Small (124M) — Baseline Chat")
    print("  Для порівняння з Nord SNN-LLM")
    print("═" * 60)

    # Завантаження
    print("\n  [*] Завантажуємо GPT-2 Small (124M)...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [✓] Готово! ({param_count:.1f}M параметрів, {device})")

    # Налаштування
    temperature = 0.8
    max_tokens = 200
    top_k = 50
    top_p = 0.9
    rep_penalty = 1.3

    print(f"\n  {'─' * 50}")
    print(f"  Пиши повідомлення і натискай Enter.")
    print(f"  Команди:")
    print(f"    /quit          — вийти")
    print(f"    /temp 0.5      — змінити temperature")
    print(f"    /tokens 300    — макс. токенів")
    print(f"    /rep 1.5       — repetition penalty")
    print(f"  {'─' * 50}\n")

    while True:
        try:
            user_input = input("  Ти: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Бувай! 👋")
            break

        if not user_input:
            continue

        # Команди
        if user_input.startswith("/"):
            parts = user_input.split()
            cmd = parts[0].lower()

            if cmd == "/quit":
                print("  Бувай! 👋")
                break
            elif cmd == "/temp" and len(parts) > 1:
                try:
                    temperature = float(parts[1])
                    print(f"  [⚙] Temperature = {temperature}")
                except ValueError:
                    print(f"  [!] Невірне значення")
            elif cmd == "/tokens" and len(parts) > 1:
                try:
                    max_tokens = int(parts[1])
                    print(f"  [⚙] Max tokens = {max_tokens}")
                except ValueError:
                    print(f"  [!] Невірне значення")
            elif cmd == "/rep" and len(parts) > 1:
                try:
                    rep_penalty = float(parts[1])
                    print(f"  [⚙] Repetition penalty = {rep_penalty}")
                except ValueError:
                    print(f"  [!] Невірне значення")
            else:
                print(f"  [!] Невідома команда: {cmd}")
            continue

        # Генерація
        t0 = time.time()

        input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=rep_penalty,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.time() - t0

        # Декодуємо тільки нові токени
        new_tokens = output[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)

        print(f"\n  GPT-2: {response}")

        resp_tokens = len(new_tokens)
        tps = resp_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{resp_tokens} tok, {elapsed:.1f}s, {tps:.1f} tok/s]\n")


if __name__ == "__main__":
    main()
