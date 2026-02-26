"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD — Крок 3: Чат з моделлю  v3.1                     ║
║                                                                        ║
║  Просто запусти:                                                       ║
║      python chat.py                                                    ║
║                                                                        ║
║  Воно запитає де лежить модель і запустить інтерактивний чат.           ║
║  Підтримує STDP: модель вчиться новим словам прямо під час розмови!    ║
║  v3.1: Repetition Penalty — менше повторень у генерації                 ║
╚══════════════════════════════════════════════════════════════════════════╝

Потрібно:
    pip install torch transformers
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F

from nord_core import NordConfig, NordModel


# ─────────────────────────────────────────────────────────────────────────────
# ЗАВАНТАЖЕННЯ МОДЕЛІ
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: str) -> tuple:
    """Завантажити модель і токенізатор."""
    from transformers import AutoTokenizer

    model_path = Path(model_dir)

    # Знайти файл моделі
    candidates = ["nord_final.pt", "nord_latest.pt"]
    ckpt_path = None
    for name in candidates:
        p = model_path / name
        if p.exists():
            ckpt_path = p
            break

    if ckpt_path is None:
        steps = sorted(model_path.glob("nord_step_*.pt"))
        if steps:
            ckpt_path = steps[-1]

    if ckpt_path is None:
        print(f"  [✗] Не знайдено моделі в: {model_dir}")
        print(f"  Спочатку натренуй:  python train_nord.py")
        sys.exit(1)

    print(f"  [*] Завантажуємо: {ckpt_path.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    saved_cfg = ckpt.get("config", {})
    cfg = NordConfig(
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        d_model=saved_cfg.get("d_model", 512),
        n_heads=saved_cfg.get("n_heads", 8),
        n_layers=saved_cfg.get("n_layers", 6),
        d_ff=saved_cfg.get("d_ff", 1024),
        T=saved_cfg.get("T", 8),
        T_slow=saved_cfg.get("T_slow", 2),
        max_seq_len=saved_cfg.get("max_seq_len", 512),
        vocab_size=saved_cfg.get("vocab_size", 128_256),
        persistent_mem=False,
    )

    model = NordModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [*] Завантажуємо Llama-3.2 токенізатор...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.tokenizer_id, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  [✓] Модель завантажена! ({param_count:.1f}M параметрів)")

    return model, tokenizer, cfg


# ─────────────────────────────────────────────────────────────────────────────
# REPETITION PENALTY
# ─────────────────────────────────────────────────────────────────────────────

def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.3,
    window: int = 50,
) -> torch.Tensor:
    """
    Зменшує ймовірність токенів які вже з'явились в останніх `window` токенах.
    penalty > 1.0 = зменшує повторення (рекомендовано 1.2-1.5)
    Чим більше разів токен з'явився — тим сильніший penalty (до 5x).
    """
    if penalty <= 1.0:
        return logits

    recent_ids = generated_ids[0, -window:].tolist()
    token_counts = Counter(recent_ids)

    for token_id, count in token_counts.items():
        if token_id < logits.size(-1):
            # Експоненційний penalty: penalty^min(count, 5)
            effective_penalty = penalty ** min(count, 5)
            if logits[0, token_id] > 0:
                logits[0, token_id] = logits[0, token_id] / effective_penalty
            else:
                logits[0, token_id] = logits[0, token_id] * effective_penalty

    return logits


# ─────────────────────────────────────────────────────────────────────────────
# ГЕНЕРАЦІЯ ТЕКСТУ
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model: NordModel,
    tokenizer,
    cfg: NordConfig,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    enable_stdp: bool = True,
    repetition_penalty: float = 1.3,
    rep_window: int = 50,
) -> str:
    """
    Авторегресивна генерація з SNN.
    v3.1: + repetition penalty для різноманітнішого тексту.
    """
    device = cfg.device

    model.reset_state()

    max_prompt_len = max(32, cfg.max_seq_len - max_new_tokens)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True,
                    max_length=max_prompt_len)
    input_ids = enc.input_ids.to(device)
    generated_ids = input_ids.clone()

    for _ in range(max_new_tokens):
        context = generated_ids[:, -cfg.max_seq_len:]

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            logits, stats = model(context, enable_stdp=enable_stdp)

        next_logits = logits[:, -1, :].float()

        # ── Repetition Penalty (до temperature!) ──
        next_logits = apply_repetition_penalty(
            next_logits, generated_ids,
            penalty=repetition_penalty,
            window=rep_window,
        )

        if temperature > 0:
            next_logits = next_logits / temperature

        if top_k > 0:
            top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            next_logits[next_logits < threshold] = float("-inf")

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove_mask] = float("-inf")
            next_logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        # v3: Reward-modulated STDP
        if enable_stdp:
            loss_proxy = -torch.log(probs.max() + 1e-8).item()
            model.stdp_update(current_loss=loss_proxy)

        if next_token.item() == tokenizer.eos_token_id:
            break

    new_ids = generated_ids[0, input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# ІНТЕРАКТИВНИЙ ЧАТ
# ─────────────────────────────────────────────────────────────────────────────

def chat_loop(model: NordModel, tokenizer, cfg: NordConfig):
    """Головний цикл чату."""

    temperature = 0.8
    max_tokens = 200
    stdp_enabled = True
    rep_penalty = 1.3
    rep_window = 50

    print(f"\n  {'─' * 50}")
    print(f"  Пиши повідомлення і натискай Enter.")
    print(f"  Команди:")
    print(f"    /quit          — вийти")
    print(f"    /temp 0.5      — змінити temperature")
    print(f"    /tokens 300    — макс. токенів у відповіді")
    print(f"    /stdp on|off   — STDP навчання під час чату")
    print(f"    /rep 1.5       — repetition penalty (1.0=вимк, 1.2-1.5=норм)")
    print(f"    /stats         — показати спайк-статистику")
    print(f"    /reset         — скинути STDP кеш")
    print(f"  {'─' * 50}\n")

    last_stats = {}

    while True:
        try:
            user_input = input("  Ти: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n  Бувай! 👋")
            break

        if not user_input:
            continue

        # ── Команди ──
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

            elif cmd == "/stdp":
                if len(parts) > 1 and parts[1].lower() in ("off", "0", "ні"):
                    stdp_enabled = False
                    print(f"  [⚙] STDP вимкнено")
                else:
                    stdp_enabled = True
                    print(f"  [⚙] STDP увімкнено — модель вчиться під час чату!")

            elif cmd == "/rep" and len(parts) > 1:
                try:
                    rep_penalty = float(parts[1])
                    print(f"  [⚙] Repetition penalty = {rep_penalty}")
                    if rep_penalty > 2.0:
                        print(f"  [!] Увага: значення > 2.0 може зламати генерацію")
                except ValueError:
                    print(f"  [!] Невірне значення")

            elif cmd == "/stats":
                if last_stats:
                    print(f"  [📊] Остання статистика:")
                    for k, v in last_stats.items():
                        print(f"       {k}: {v:.4f}")
                else:
                    print(f"  [!] Ще нема статистики — напиши щось спочатку")

            elif cmd == "/reset":
                model._stdp_cache.clear()
                print(f"  [⚙] STDP кеш скинуто")

            else:
                print(f"  [!] Невідома команда: {cmd}")

            continue

        # ── Генерація ──
        t0 = time.time()

        response = generate(
            model, tokenizer, cfg,
            prompt=user_input,
            max_new_tokens=max_tokens,
            temperature=temperature,
            enable_stdp=stdp_enabled,
            repetition_penalty=rep_penalty,
            rep_window=rep_window,
        )

        elapsed = time.time() - t0

        print(f"\n  Nord: {response}")

        resp_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        tps = resp_tokens / elapsed if elapsed > 0 else 0
        stdp_tag = " [STDP ✓]" if stdp_enabled else ""
        rep_tag = f" [REP {rep_penalty}]" if rep_penalty > 1.0 else ""
        print(f"  [{resp_tokens} tok, {elapsed:.1f}s, {tps:.1f} tok/s{stdp_tag}{rep_tag}]\n")

        # Зберегти статистику
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(cfg.device == "cuda")):
            ids = tokenizer(user_input, return_tensors="pt",
                          truncation=True, max_length=cfg.max_seq_len).input_ids.to(cfg.device)
            _, last_stats = model(ids)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print()
    print("═" * 60)
    print("  ⚡ PROJECT NORD — Spiking Neural Network Chat v3.1")
    print("═" * 60)

    default_model = os.path.join("D:", os.sep, "nord_model")
    print(f"\n  Де лежить навчена модель?")
    print(f"  (Enter = {default_model})")
    model_input = input("  Шлях: ").strip()
    model_dir = model_input if model_input else default_model

    if not Path(model_dir).exists():
        print(f"\n  [✗] Папка не знайдена: {model_dir}")
        print(f"  Спочатку натренуй:  python train_nord.py")
        sys.exit(1)

    model, tokenizer, cfg = load_model(model_dir)
    chat_loop(model, tokenizer, cfg)


if __name__ == "__main__":
    main()