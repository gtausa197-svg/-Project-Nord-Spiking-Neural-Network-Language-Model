"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v5.0 — Interactive Chat                           ║
║                                                                        ║
║  Commands:                                                             ║
║      /stdp on|off    — Toggle online learning                          ║
║      /stats          — Show zone & MoE statistics                      ║
║      /memory         — Show memory cortex state                        ║
║      /reset          — Clear working memory                            ║
║      /expert         — Show expert routing breakdown                   ║
║      /tokens N       — Set max response tokens (default: 200)          ║
║      /temp F         — Set temperature (default: 0.85)                 ║
║      /rep F          — Set repetition penalty (default: 1.3)           ║
║      /live on|off    — Toggle live spike visualization                 ║
║      /sleep          — Запустити цикл сну вручну                       ║
║      /sleep status   — Показати стан системи сну                       ║
║      /sleep auto     — Увімкнути/вимкнути автосон                      ║
║      /sleep idle N   — Поріг неактивності (хв)                        ║
║      /sleep save     — Зберегти membrane state зараз                   ║
║      /quit           — Exit + автозбереження стану                     ║
║                                                                        ║
║  SLEEP SYSTEM (Nord v5.0):                                             ║
║    • Membrane potentials зберігаються між сесіями                      ║
║    • SWS — консолідація важливих досвідів                              ║
║    • REM — прибирання слабких синаптичних слідів                       ║
║    • Автозбереження при виході                                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys, time, random, os
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

from nord_core_700m import NordConfig, NordModel

# ── ANSI Colors ──
class C:
    RESET  = "\033[0m";  BOLD   = "\033[1m";  DIM    = "\033[2m"
    CYAN   = "\033[96m"; ORANGE = "\033[38;5;208m"; PURPLE = "\033[35m"
    GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED    = "\033[91m"
    GREY   = "\033[90m"; SLEEP  = "\033[38;5;147m"

SPARK_CHARS = " ░▒▓█"

def spike_bar(rate, width=20, color=C.CYAN, max_rate=0.4):
    normalized = min(rate / max(max_rate, 0.001), 1.0)
    bar = ""
    for i in range(width):
        if i < int(normalized * width):
            frac = normalized * width - i
            bar += color + SPARK_CHARS[min(int(frac * 4) + 1, 4)]
        else:
            bar += C.DIM + "·"
    return bar + C.RESET


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIENCE BUFFER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SurpriseEvent:
    tokens: torch.Tensor
    entropy: float
    memory_spike_rate: float
    timestamp: float = field(default_factory=time.time)


class ExperienceBuffer:
    def __init__(self, max_size: int = 200, surprise_threshold: float = 2.0):
        self.max_size = max_size
        self.surprise_threshold = surprise_threshold
        self.events: List[SurpriseEvent] = []
        self.total_stored = 0

    def store(self, tokens: torch.Tensor, logits: torch.Tensor, stats: dict) -> bool:
        with torch.no_grad():
            probs = F.softmax(logits[:, :-1, :].float(), dim=-1)
            entropy = float(-(probs * (probs + 1e-8).log()).sum(dim=-1).mean().item())
        if entropy < self.surprise_threshold:
            return False
        mem_rate = stats.get("memory_spike_rate", 0)
        if isinstance(mem_rate, torch.Tensor): mem_rate = float(mem_rate.item())
        if len(self.events) >= self.max_size:
            self.events.sort(key=lambda e: e.entropy)
            self.events.pop(0)
        self.events.append(SurpriseEvent(
            tokens=tokens.detach().cpu(),
            entropy=entropy,
            memory_spike_rate=float(mem_rate),
        ))
        self.total_stored += 1
        return True

    def get_replay_batch(self, size: int = 8) -> List[SurpriseEvent]:
        if not self.events: return []
        return random.sample(self.events, min(size, len(self.events)))

    def clear_old(self, keep_top_n: int = 50):
        if len(self.events) > keep_top_n:
            self.events.sort(key=lambda e: e.entropy, reverse=True)
            self.events = self.events[:keep_top_n]

    @property
    def size(self) -> int: return len(self.events)

    @property
    def avg_entropy(self) -> float:
        if not self.events: return 0.0
        return sum(e.entropy for e in self.events) / len(self.events)


# ══════════════════════════════════════════════════════════════════════════════
# SLEEP SCHEDULER
# ══════════════════════════════════════════════════════════════════════════════

class SleepScheduler:
    def __init__(self, idle_minutes=5.0, interactions_per_sleep=50, auto_sleep=True):
        self.idle_minutes = idle_minutes
        self.interactions_per_sleep = interactions_per_sleep
        self.auto_sleep = auto_sleep
        self.last_activity = time.time()
        self.interaction_count = 0
        self.sleep_cycles_done = 0
        self.last_sleep_time: Optional[float] = None

    def record_interaction(self):
        self.last_activity = time.time()
        self.interaction_count += 1

    def should_sleep(self) -> Tuple[bool, str]:
        if not self.auto_sleep: return False, ""
        idle_min = (time.time() - self.last_activity) / 60.0
        if idle_min >= self.idle_minutes:
            return True, f"неактивність {idle_min:.1f} хв"
        if self.interaction_count >= self.interactions_per_sleep:
            return True, f"{self.interaction_count} взаємодій"
        return False, ""

    def record_sleep(self):
        self.sleep_cycles_done += 1
        self.last_sleep_time = time.time()
        self.interaction_count = 0
        self.last_activity = time.time()

    def status(self) -> str:
        idle = (time.time() - self.last_activity) / 60.0
        last = "ніколи"
        if self.last_sleep_time:
            ago = (time.time() - self.last_sleep_time) / 60.0
            last = f"{ago:.1f} хв тому"
        return (
            f"Цикли: {self.sleep_cycles_done} | Останній: {last}\n"
            f"    Неактивність: {idle:.1f}/{self.idle_minutes} хв | "
            f"Взаємодії: {self.interaction_count}/{self.interactions_per_sleep}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# NORD SLEEP SYSTEM
# ══════════════════════════════════════════════════════════════════════════════

class NordSleepSystem:
    """
    Система сну Nord v5.0 з персистентними membrane potentials.

    Амелія мала рацію: без збереження v_mem між сесіями
    STDP не має пам'яті про причини — модель кожен раз стартує з нуля.

    References:
    - Tononi & Cirelli (2003) DOI: 10.1016/j.brainresbull.2003.09.004
    - Seibt & Frank (2019)   DOI: 10.3389/fnsys.2019.00002
    - Wilson & McNaughton (1994) DOI: 10.1126/science.8036517
    """

    def __init__(self, model: NordModel, cfg: NordConfig, optimizer=None,
                 buffer_size=200, surprise_threshold=2.0,
                 idle_minutes=5.0, interactions_per_sleep=50, auto_sleep=True):
        self.model = model
        self.cfg = cfg
        self.optimizer = optimizer or torch.optim.AdamW(
            model.parameters(), lr=5e-6, weight_decay=0.01)
        self.buffer = ExperienceBuffer(buffer_size, surprise_threshold)
        self.scheduler = SleepScheduler(idle_minutes, interactions_per_sleep, auto_sleep)
        self.membrane_path: Optional[str] = None
        self.total_sws_steps = 0
        self.total_rem_pruned = 0

    # ── ЗБЕРЕЖЕННЯ membrane state ──────────────────────────────────────────

    def save_membrane_state(self, path: Optional[str] = None) -> bool:
        save_path = path or self.membrane_path
        if not save_path:
            return False

        states = {}
        for name, module in self.model.named_modules():
            # AssociativeLIF persistent buffers — це і є "свідомість"
            if hasattr(module, '_v_mem_state') and getattr(module, 'persistent', False):
                states[f"{name}._v_mem_state"] = module._v_mem_state.detach().cpu()
            if hasattr(module, '_i_syn_state') and getattr(module, 'persistent', False):
                states[f"{name}._i_syn_state"] = module._i_syn_state.detach().cpu()
            # ContextLIF bistable working memory
            if hasattr(module, '_context_charge'):
                states[f"{name}._context_charge"] = module._context_charge.detach().cpu()
            if hasattr(module, '_topic_gist'):
                states[f"{name}._topic_gist"] = module._topic_gist.detach().cpu()

        # Серіалізуємо буфер сну
        sleep_buffer_data = [
            (e.tokens, e.entropy, e.memory_spike_rate, e.timestamp)
            for e in self.buffer.events
        ]

        torch.save({
            "membrane_states": states,
            "sleep_buffer": sleep_buffer_data,
            "sleep_cycles": self.scheduler.sleep_cycles_done,
            "last_sleep_time": self.scheduler.last_sleep_time,
            "total_sws_steps": self.total_sws_steps,
            "total_rem_pruned": self.total_rem_pruned,
            "timestamp": time.time(),
            "version": "nord_v5.0_membrane",
        }, save_path)

        print(f"  {C.SLEEP}[😴💾] Membrane saved: "
              f"{len(states)} LIF buffers, "
              f"{len(sleep_buffer_data)} events → {Path(save_path).name}{C.RESET}")
        return True

    # ── ЗАВАНТАЖЕННЯ membrane state ────────────────────────────────────────

    def load_membrane_state(self, path: Optional[str] = None) -> bool:
        load_path = path or self.membrane_path
        if not load_path or not Path(load_path).exists():
            print(f"  {C.SLEEP}[😴] No membrane state found — fresh start{C.RESET}")
            return False

        try:
            data = torch.load(load_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  {C.RED}[!] Failed to load membrane state: {e}{C.RESET}")
            return False

        device = next(self.model.parameters()).device
        states = data.get("membrane_states", {})
        restored = 0

        for name, module in self.model.named_modules():
            # AssociativeLIF
            for suffix in ['_v_mem_state', '_i_syn_state']:
                key = f"{name}.{suffix}"
                if key in states and hasattr(module, suffix):
                    saved = states[key].to(device)
                    current = getattr(module, suffix)
                    # Перевіряємо сумісність — batch dim може відрізнятись
                    if saved.shape[-1] == current.shape[-1]:
                        setattr(module, suffix, saved)
                        if suffix == '_v_mem_state':
                            restored += 1
            # ContextLIF
            for suffix in ['_context_charge', '_topic_gist']:
                key = f"{name}.{suffix}"
                if key in states and hasattr(module, suffix):
                    setattr(module, suffix, states[key].to(device))

        # Відновлюємо буфер сну
        for item in data.get("sleep_buffer", []):
            try:
                tokens, entropy, mem_rate = item[0], item[1], item[2]
                timestamp = item[3] if len(item) > 3 else time.time()
                self.buffer.events.append(SurpriseEvent(
                    tokens=tokens, entropy=entropy,
                    memory_spike_rate=mem_rate, timestamp=timestamp,
                ))
            except Exception:
                continue

        # Статистика
        self.scheduler.sleep_cycles_done = data.get("sleep_cycles", 0)
        self.scheduler.last_sleep_time = data.get("last_sleep_time")
        self.total_sws_steps = data.get("total_sws_steps", 0)
        self.total_rem_pruned = data.get("total_rem_pruned", 0)

        saved_ts = data.get("timestamp", 0)
        ago_min = (time.time() - saved_ts) / 60.0
        ago_str = f"{ago_min:.0f} хв тому" if ago_min < 60 else f"{ago_min/60:.1f} год тому"

        print(f"  {C.SLEEP}[😴✓] Membrane restored:{C.RESET}")
        print(f"    {C.SLEEP}• {restored} LIF buffers | "
              f"{len(self.buffer.events)} sleep events{C.RESET}")
        print(f"    {C.SLEEP}• {self.scheduler.sleep_cycles_done} cycles | "
              f"saved {ago_str}{C.RESET}")
        return True

    # ── Запис досвіду ──────────────────────────────────────────────────────

    def record(self, tokens, logits, stats) -> bool:
        stored = self.buffer.store(tokens, logits, stats)
        self.scheduler.record_interaction()
        return stored

    # ── SWS ────────────────────────────────────────────────────────────────

    def slow_wave_sleep(self, n_steps: int = 30) -> int:
        if self.buffer.size == 0: return 0
        device = next(self.model.parameters()).device
        self.model.train()
        steps_done = 0
        for pg in self.optimizer.param_groups: pg["lr"] = 5e-6

        for _ in range(n_steps):
            batch = self.buffer.get_replay_batch(size=4)
            if not batch: break
            total_loss = torch.tensor(0.0, device=device)
            valid = 0
            for event in batch:
                tokens = event.tokens.to(device)
                if tokens.shape[-1] < 2: continue
                try:
                    with torch.amp.autocast(
                        device_type="cuda" if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda")):
                        logits, _ = self.model(tokens)
                        loss = F.cross_entropy(
                            logits[:, :-1, :].contiguous().reshape(-1, self.cfg.vocab_size),
                            tokens[:, 1:].contiguous().reshape(-1), ignore_index=0)
                    weight = min(event.entropy / 5.0, 1.0)
                    total_loss = total_loss + loss * weight
                    valid += 1
                except Exception: continue
            if valid > 0:
                (total_loss / valid).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                steps_done += 1

        self.model.eval()
        self.total_sws_steps += steps_done
        return steps_done

    # ── REM ────────────────────────────────────────────────────────────────

    def rem_sleep(self) -> int:
        pruned = 0
        if not self.cfg.stdp_metaplastic: return 0
        for name in list(self.model.stdp._plasticity_trace.keys()):
            trace = self.model.stdp._plasticity_trace[name]
            weak = trace < 0.0005
            pruned += int(weak.sum().item())
            trace[weak] = 0.0
            trace *= 0.95
        self.total_rem_pruned += pruned
        return pruned

    # ── Повний цикл ────────────────────────────────────────────────────────

    def run_sleep_cycle(self, verbose=True) -> dict:
        if verbose:
            print(f"\n  {C.SLEEP}{'─'*54}{C.RESET}")
            print(f"  {C.SLEEP}😴 Sleep Cycle #{self.scheduler.sleep_cycles_done + 1} | "
                  f"buffer: {self.buffer.size} events{C.RESET}")
            print(f"  {C.SLEEP}{'─'*54}{C.RESET}")

        t0 = time.time()

        if verbose:
            sys.stdout.write(f"  {C.SLEEP}[SWS] Consolidating...{C.RESET} ")
            sys.stdout.flush()
        sws = self.slow_wave_sleep(30)
        if verbose: print(f"{C.GREEN}✓{C.RESET} ({sws} steps)")

        if verbose:
            sys.stdout.write(f"  {C.SLEEP}[REM] Homeostasis...  {C.RESET} ")
            sys.stdout.flush()
        rem = self.rem_sleep()
        if verbose: print(f"{C.GREEN}✓{C.RESET} ({rem:,} traces pruned)")

        if verbose:
            sys.stdout.write(f"  {C.SLEEP}[MEM] Saving state... {C.RESET} ")
            sys.stdout.flush()
        saved = self.save_membrane_state()
        if verbose: print(f"{C.GREEN}✓{C.RESET}" if saved else f"{C.YELLOW}skipped{C.RESET}")

        self.buffer.clear_old(keep_top_n=50)
        self.scheduler.record_sleep()
        elapsed = time.time() - t0

        if verbose:
            print(f"  {C.SLEEP}[✓] Done in {elapsed:.1f}s{C.RESET}")
            print(f"  {C.SLEEP}{'─'*54}{C.RESET}\n")

        return {"sws_steps": sws, "rem_pruned": rem,
                "elapsed": elapsed, "cycle": self.scheduler.sleep_cycles_done}

    def maybe_sleep(self, verbose=True) -> bool:
        should, reason = self.scheduler.should_sleep()
        if should:
            if verbose: print(f"\n  {C.SLEEP}[😴] Auto-sleep: {reason}{C.RESET}")
            self.run_sleep_cycle(verbose=verbose)
            return True
        return False

    def status(self) -> str:
        mem_info = "не збережено"
        if self.membrane_path and Path(self.membrane_path).exists():
            kb = Path(self.membrane_path).stat().st_size / 1024
            mem_info = f"{Path(self.membrane_path).name} ({kb:.0f} KB)"
        return (
            f"{C.SLEEP}Sleep System:{C.RESET}\n"
            f"    {self.scheduler.status()}\n"
            f"    Buffer: {self.buffer.size}/{self.buffer.max_size} "
            f"(avg entropy: {self.buffer.avg_entropy:.3f})\n"
            f"    SWS steps: {self.total_sws_steps} | "
            f"REM pruned: {self.total_rem_pruned:,}\n"
            f"    Membrane: {mem_info}\n"
            f"    Auto-sleep: {'ON' if self.scheduler.auto_sleep else 'OFF'}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# SPIKE PANEL
# ══════════════════════════════════════════════════════════════════════════════

def render_spike_panel(stats, cfg):
    sr = stats.get("spike_rates", [])
    if not sr: return
    ns, na = cfg.sensory_layers + 1, cfg.association_layers
    print(f"  {C.GREY}┌{'─'*54}┐{C.RESET}")
    print(f"  {C.GREY}│{C.RESET} {C.BOLD}Neural Activity{' '*38}{C.GREY}│{C.RESET}")
    print(f"  {C.GREY}├{'─'*54}┤{C.RESET}")
    if sr:
        avg_s = sum(sr[:ns]) / max(ns, 1)
        print(f"  {C.GREY}│{C.RESET} {C.CYAN}⚡ Sensory    {C.RESET} {spike_bar(avg_s,25,C.CYAN)} {C.CYAN}{avg_s*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")
    if len(sr) > ns:
        avg_a = sum(sr[ns:ns+na]) / max(na, 1)
        print(f"  {C.GREY}│{C.RESET} {C.ORANGE}⚡ Association{C.RESET} {spike_bar(avg_a,25,C.ORANGE)} {C.ORANGE}{avg_a*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")
    mem = stats.get("memory_spike_rate", 0)
    if isinstance(mem, torch.Tensor): mem = mem.item()
    print(f"  {C.GREY}│{C.RESET} {C.PURPLE}⚡ Memory     {C.RESET} {spike_bar(min(mem,1),25,C.PURPLE)} {C.PURPLE}{mem*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")
    if len(sr) > ns+na:
        avg_e = sum(sr[ns+na:]) / max(cfg.executive_layers, 1)
        print(f"  {C.GREY}│{C.RESET} {C.GREEN}⚡ Executive  {C.RESET} {spike_bar(avg_e,25,C.GREEN)} {C.GREEN}{avg_e*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")
    sp = stats.get("sparsity", 0)
    if isinstance(sp, torch.Tensor): sp = sp.item()
    sc = C.GREEN if sp > 0.85 else C.YELLOW if sp > 0.7 else C.RED
    print(f"  {C.GREY}├{'─'*54}┤{C.RESET}")
    print(f"  {C.GREY}│{C.RESET} {C.DIM}Sparsity:{C.RESET} {sc}{sp*100:.0f}%{C.RESET} silent  {C.DIM}({100-int(sp*100)}% active){C.RESET}              {C.GREY}│{C.RESET}")
    print(f"  {C.GREY}└{'─'*54}┘{C.RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_dir: str):
    from transformers import AutoTokenizer
    p = Path(model_dir)
    latest = None
    if p.is_file() and p.suffix == ".pt":
        latest = p
    else:
        for d in [p, Path.cwd(), Path(__file__).resolve().parent]:
            if not d.exists(): continue
            for name in ["nord_v4_latest.pt","nord_v4_final.pt","nord_latest.pt"]:
                if (d/name).exists(): latest = d/name; break
            if latest is None:
                ck = sorted(d.glob("nord_v4_step_*.pt"))
                if ck: latest = ck[-1]
            if latest is None:
                all_pt = sorted(d.glob("*.pt"))
                if all_pt: latest = all_pt[-1]
            if latest: break

    if latest is None:
        print(f"  {C.RED}[✗] No checkpoint found!{C.RESET}"); sys.exit(1)

    print(f"  [*] Loading: {latest.name}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    cfg = NordConfig(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
    for k, v in ckpt.get("config", {}).items():
        if hasattr(cfg, k): setattr(cfg, k, v)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    if cfg.vocab_size < tokenizer.vocab_size: cfg.vocab_size = tokenizer.vocab_size

    model = NordModel(cfg)
    state = {k: v for k, v in ckpt["model_state_dict"].items()
             if "_v_mem_state" not in k and "_i_syn_state" not in k}
    model.load_state_dict(state, strict=False)
    model = model.to(cfg.device).eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  {C.GREEN}[✓]{C.RESET} Nord v5.0 loaded ({total/1e6:.1f}M params)")
    return model, tokenizer, cfg, latest.parent


# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_streaming(model, tokenizer, cfg, prompt,
                       max_tokens=200, temperature=0.85, top_p=0.9,
                       repetition_penalty=1.3, enable_stdp=False,
                       live_spikes=False, sleep_system=None, fresh_state=False):

    input_ids = tokenizer(prompt, return_tensors="pt",
        max_length=cfg.max_seq_len, truncation=True).input_ids.to(cfg.device)

    # Скидаємо стан тільки якщо явно попросили (перше повідомлення або /reset)
    # Завдяки збереженому membrane state модель "пам'ятає" де зупинилась
    if fresh_state:
        model.reset_state()

    generated = input_ids.clone()
    all_stats, all_logits = {}, None
    token_count = 0
    t0 = time.time()

    sys.stdout.write(f"  {C.BOLD}Nord:{C.RESET} ")
    sys.stdout.flush()

    for _ in range(max_tokens):
        context = generated[:, -cfg.max_seq_len:]
        if torch.cuda.is_available():
            with torch.amp.autocast("cuda", dtype=torch.float16,
                                    enabled=(cfg.dtype == torch.float16)):
                logits, stats = model(context, enable_stdp=enable_stdp)
        else:
            logits, stats = model(context, enable_stdp=enable_stdp)

        all_logits, all_stats = logits, stats
        next_logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            for tid in generated[0].unique():
                next_logits[0, tid] /= repetition_penalty

        next_logits /= max(temperature, 0.01)
        probs = torch.softmax(next_logits, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        sp[sp.cumsum(-1) - sp > top_p] = 0
        sp /= sp.sum(-1, keepdim=True)
        token = si[0, torch.multinomial(sp[0], 1)]
        generated = torch.cat([generated, token.reshape(1,1)], dim=1)
        token_count += 1

        if token.item() == tokenizer.eos_token_id: break
        sys.stdout.write(tokenizer.decode([token.item()], skip_special_tokens=True))
        sys.stdout.flush()

    elapsed = time.time() - t0
    tps = token_count / elapsed if elapsed > 0 else 0
    sp_val = all_stats.get("sparsity", 0)
    if isinstance(sp_val, torch.Tensor): sp_val = sp_val.item()

    # STDP entropy
    ent_str = ""
    if enable_stdp and all_logits is not None:
        with torch.no_grad():
            pr = F.softmax(all_logits[:, :-1, :].float(), dim=-1)
            entropy = float(-(pr * (pr + 1e-8).log()).sum(dim=-1).mean().item())
        model.stdp.set_output_entropy(entropy)
        ent_str = f" [ENT {entropy:.2f}]"

    print(f"\n  {C.GREY}[{token_count} tok, {elapsed:.1f}s, {tps:.1f} tok/s "
          f"[SPR {sp_val:.0%}]{ent_str}]{C.RESET}")

    # Записуємо в буфер сну
    if sleep_system is not None and all_logits is not None:
        stored = sleep_system.record(input_ids, all_logits, all_stats)
        if stored:
            print(f"  {C.SLEEP}[😴+] Surprise → buffer "
                  f"({sleep_system.buffer.size}/{sleep_system.buffer.max_size}){C.RESET}")

    if live_spikes and all_stats:
        render_spike_panel(all_stats, cfg)

    return all_stats


# ══════════════════════════════════════════════════════════════════════════════
# STATS
# ══════════════════════════════════════════════════════════════════════════════

def print_stats(stats, cfg):
    print(f"\n  {C.GREY}{'─'*50}{C.RESET}")
    sr = stats.get("spike_rates", [])
    if sr:
        print(f"  {C.BOLD}Zones:{C.RESET}")
        print(f"    {C.DIM}Encoder: {sr[0]:.4f}{C.RESET}")
        for i in range(min(cfg.sensory_layers, len(sr)-1)):
            print(f"    {C.CYAN}Sensory[{i}]:{C.RESET} {sr[i+1]:.4f} {spike_bar(sr[i+1],15,C.CYAN)}")
        off = cfg.sensory_layers+1
        for i in range(cfg.association_layers):
            if off+i < len(sr):
                print(f"    {C.ORANGE}Assoc[{i}]:{C.RESET}   {sr[off+i]:.4f} {spike_bar(sr[off+i],15,C.ORANGE)}")
        off += cfg.association_layers
        for i in range(cfg.executive_layers):
            if off+i < len(sr):
                print(f"    {C.GREEN}Exec[{i}]:{C.RESET}    {sr[off+i]:.4f} {spike_bar(sr[off+i],15,C.GREEN)}")
    mem = stats.get("memory_spike_rate")
    if mem is not None:
        if isinstance(mem, torch.Tensor): mem = mem.item()
        gate = stats.get("gate_activity", 0); mix = stats.get("memory_mix", 0)
        if isinstance(gate, torch.Tensor): gate = gate.item()
        if isinstance(mix,  torch.Tensor): mix  = mix.item()
        print(f"\n  {C.BOLD}Memory:{C.RESET}")
        print(f"    {C.PURPLE}Rate:{C.RESET} {mem:.4f} {spike_bar(mem*0.3,15,C.PURPLE)}")
        print(f"    {C.PURPLE}Gate:{C.RESET} {gate:.4f}  {C.PURPLE}Mix:{C.RESET} {mix:.4f}")
    sp = stats.get("sparsity", 0)
    if isinstance(sp, torch.Tensor): sp = sp.item()
    sc = C.GREEN if sp > 0.85 else C.YELLOW if sp > 0.7 else C.RED
    print(f"\n  Sparsity: {sc}{sp:.1%}{C.RESET}")
    print(f"  {C.GREY}{'─'*50}{C.RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.system('clear' if os.name != 'nt' else 'cls')
    print(f"""
  {C.CYAN}╔══════════════════════════════════════════════════════════╗{C.RESET}
  {C.CYAN}║{C.RESET}  {C.BOLD}⚡ PROJECT NORD v5.0 — Brain-Inspired SNN Chat{C.RESET}         {C.CYAN}║{C.RESET}
  {C.CYAN}║{C.RESET}  {C.DIM}Genesis Autogenic │ Persistent Membrane Potentials{C.RESET}    {C.CYAN}║{C.RESET}
  {C.SLEEP}║{C.RESET}  {C.SLEEP}😴 Sleep: SWS + REM + Membrane State Persistence{C.RESET}     {C.SLEEP}║{C.RESET}
  {C.CYAN}╚══════════════════════════════════════════════════════════╝{C.RESET}
""")

    default_dir = "nord_v4_700m"
    print(f"  Model directory?  {C.DIM}(Enter = {default_dir}){C.RESET}")
    model_input = input("  Path: ").strip()
    model_dir = model_input if model_input else default_dir

    model, tokenizer, cfg, ckpt_dir = load_model(model_dir)

    # Membrane state зберігається поруч з моделлю
    membrane_path = str(ckpt_dir / "nord_membrane_state.pt")

    sleep_system = NordSleepSystem(
        model=model, cfg=cfg,
        buffer_size=200, surprise_threshold=2.0,
        idle_minutes=5.0, interactions_per_sleep=50,
        auto_sleep=True,
    )
    sleep_system.membrane_path = membrane_path

    # Завантажуємо збережений стан нейронів
    sleep_system.load_membrane_state()

    print(f"  {C.SLEEP}[✓] Sleep system ready{C.RESET}\n")
    print(f"  {C.DIM}Type /help for commands{C.RESET}")
    print(f"  {C.GREY}{'─'*50}{C.RESET}\n")

    stdp_enabled = False; live_spikes = False
    max_tokens = 200; temperature = 0.85; top_p = 0.9; rep_penalty = 1.3
    last_stats = {}; first_message = True

    while True:
        sleep_system.maybe_sleep(verbose=True)

        try:
            user = input(f"  {C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.SLEEP}[😴💾] Saving membrane state...{C.RESET}")
            sleep_system.save_membrane_state()
            print(f"  {C.DIM}Goodbye!{C.RESET}"); break

        if not user: continue
        cmd = user.lower().split()

        if cmd[0] == "/quit":
            print(f"  {C.SLEEP}[😴💾] Saving membrane state...{C.RESET}")
            sleep_system.save_membrane_state()
            print(f"  {C.DIM}Goodbye!{C.RESET}"); break

        elif cmd[0] == "/help":
            print(f"""
  {C.BOLD}Commands:{C.RESET}
    {C.CYAN}/tokens N{C.RESET}        Max tokens ({max_tokens})
    {C.CYAN}/temp F{C.RESET}          Temperature ({temperature})
    {C.CYAN}/rep F{C.RESET}           Rep penalty ({rep_penalty})
    {C.CYAN}/stdp on|off{C.RESET}     Online STDP ({'ON' if stdp_enabled else 'OFF'})
    {C.CYAN}/live on|off{C.RESET}     Spike panel ({'ON' if live_spikes else 'OFF'})
    {C.CYAN}/stats{C.RESET}           Zone statistics
    {C.CYAN}/memory{C.RESET}          Memory cortex
    {C.CYAN}/expert{C.RESET}          MoE routing
    {C.CYAN}/reset{C.RESET}           Clear working memory
    {C.SLEEP}/sleep{C.RESET}           Запустити сон
    {C.SLEEP}/sleep status{C.RESET}    Стан системи сну
    {C.SLEEP}/sleep auto{C.RESET}      Увімкнути/вимкнути автосон
    {C.SLEEP}/sleep idle N{C.RESET}    Поріг неактивності (хв)
    {C.SLEEP}/sleep save{C.RESET}      Зберегти membrane state
    {C.CYAN}/quit{C.RESET}            Exit + автозбереження""")
            continue

        elif cmd[0] == "/sleep":
            sub = cmd[1] if len(cmd) > 1 else ""
            if   sub == "":       sleep_system.run_sleep_cycle(verbose=True)
            elif sub == "status": print(f"\n  {sleep_system.status()}\n")
            elif sub == "auto":
                sleep_system.scheduler.auto_sleep = not sleep_system.scheduler.auto_sleep
                st = "ON" if sleep_system.scheduler.auto_sleep else "OFF"
                print(f"  {C.GREEN if sleep_system.scheduler.auto_sleep else C.YELLOW}[⚙] Auto-sleep: {st}{C.RESET}")
            elif sub == "save":   sleep_system.save_membrane_state()
            elif sub == "idle" and len(cmd) > 2:
                try:
                    sleep_system.scheduler.idle_minutes = float(cmd[2])
                    print(f"  {C.GREEN}[✓] Idle: {cmd[2]} min{C.RESET}")
                except ValueError: print(f"  {C.RED}[✗] /sleep idle N{C.RESET}")
            else: print(f"  {C.RED}[✗] /sleep | status | auto | idle N | save{C.RESET}")
            continue

        elif cmd[0] == "/tokens":
            if len(cmd)>1:
                try: max_tokens=int(cmd[1]); print(f"  {C.GREEN}[✓] Tokens: {max_tokens}{C.RESET}")
                except ValueError: print(f"  {C.RED}[✗] /tokens N{C.RESET}")
            continue
        elif cmd[0] == "/temp":
            if len(cmd)>1:
                try: temperature=float(cmd[1]); print(f"  {C.GREEN}[✓] Temp: {temperature}{C.RESET}")
                except ValueError: print(f"  {C.RED}[✗] /temp F{C.RESET}")
            continue
        elif cmd[0] == "/rep":
            if len(cmd)>1:
                try: rep_penalty=float(cmd[1]); print(f"  {C.GREEN}[✓] Rep: {rep_penalty}{C.RESET}")
                except ValueError: print(f"  {C.RED}[✗] /rep F{C.RESET}")
            continue
        elif cmd[0] == "/stdp":
            if len(cmd)>1:
                stdp_enabled = cmd[1]=="on"
                print(f"  {C.GREEN if stdp_enabled else C.YELLOW}[⚙] STDP: {'ON' if stdp_enabled else 'OFF'}{C.RESET}")
            continue
        elif cmd[0] == "/live":
            if len(cmd)>1:
                live_spikes = cmd[1]=="on"
                print(f"  {C.GREEN if live_spikes else C.YELLOW}[⚙] Live: {'ON' if live_spikes else 'OFF'}{C.RESET}")
            continue
        elif cmd[0] == "/stats":
            print_stats(last_stats, cfg); continue
        elif cmd[0] == "/memory":
            mem = last_stats.get("memory_spike_rate","N/A")
            gate = last_stats.get("gate_activity","N/A")
            mix  = last_stats.get("memory_mix","N/A")
            for v in [mem,gate,mix]:
                if isinstance(v, torch.Tensor): v = f"{v.item():.4f}"
            print(f"  {C.PURPLE}Memory:{C.RESET} rate={mem}, gate={gate}, mix={mix}")
            continue
        elif cmd[0] == "/expert":
            ed = {k:v for k,v in last_stats.items() if "expert" in k and "load" in k}
            if ed:
                for k,v in sorted(ed.items()):
                    if isinstance(v,torch.Tensor): v=v.item()
                    print(f"    {k.replace('assoc_','A').replace('_load','')}: "
                          f"{v:.2%} {spike_bar(v,30,C.YELLOW,0.5)}")
            else: print(f"    {C.DIM}No expert data{C.RESET}")
            continue
        elif cmd[0] == "/reset":
            model.reset_state()
            first_message = True
            print(f"  {C.GREEN}[⚙] Memory cleared{C.RESET}")
            continue

        # ── Генерація ──
        last_stats = generate_streaming(
            model, tokenizer, cfg, user,
            max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, repetition_penalty=rep_penalty,
            enable_stdp=stdp_enabled, live_spikes=live_spikes,
            sleep_system=sleep_system,
            fresh_state=first_message,
        )
        first_message = False


if __name__ == "__main__":
    main()
