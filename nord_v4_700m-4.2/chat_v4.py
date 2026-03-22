"""
╔══════════════════════════════════════════════════════════════════════════╗
║         PROJECT NORD v4 — Interactive Chat v4.0                         ║
║                                                                        ║
║  Commands:                                                             ║
║      /stdp on|off   — Toggle online learning                           ║
║      /stats         — Show zone & MoE statistics                        ║
║      /memory        — Show memory cortex state                          ║
║      /reset         — Clear working memory                              ║
║      /expert        — Show expert routing breakdown                     ║
║      /tokens N      — Set max response tokens (default: 200)            ║
║      /temp F        — Set temperature (default: 0.85)                   ║
║      /rep F         — Set repetition penalty (default: 1.3)             ║
║      /live on|off   — Toggle live spike visualization                   ║
║      /quit          — Exit                                              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import sys
import time
import torch
import os
from pathlib import Path

from nord_core_700m import NordConfig, NordModel

# ── ANSI Colors ──
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    CYAN    = "\033[96m"
    ORANGE  = "\033[38;5;208m"
    PURPLE  = "\033[35m"
    GREEN   = "\033[92m"
    BLUE    = "\033[94m"
    YELLOW  = "\033[93m"
    RED     = "\033[91m"
    WHITE   = "\033[97m"
    GREY    = "\033[90m"

SPARK_CHARS = " ░▒▓█"

def spike_bar(rate, width=20, color=C.CYAN, max_rate=0.4):
    """Colored bar with adjustable scale. max_rate=0.4 means 40% rate fills full bar"""
    normalized = min(rate / max(max_rate, 0.001), 1.0)
    filled = int(normalized * width)
    bar = ""
    for i in range(width):
        if i < filled:
            frac = normalized * width - i
            intensity = min(4, int(frac * 4))
            bar += color + SPARK_CHARS[min(intensity + 1, 4)]
        else:
            bar += C.DIM + "·"
    return bar + C.RESET


def render_live_spikes(stats, cfg):
    spike_rates = stats.get("spike_rates", [])
    if not spike_rates:
        return

    lines = []
    lines.append(f"  {C.GREY}{'─' * 56}{C.RESET}")

    ns = cfg.sensory_layers + 1
    if len(spike_rates) > 0:
        avg_s = sum(spike_rates[:ns]) / max(ns, 1)
        bar = spike_bar(avg_s, 20, C.CYAN)
        lines.append(f"  {C.CYAN}⚡ SEN{C.RESET} {bar} {C.CYAN}{avg_s*100:5.1f}%{C.RESET}")

    na = cfg.association_layers
    if len(spike_rates) > ns:
        assoc_rates = spike_rates[ns:ns+na]
        avg_a = sum(assoc_rates) / max(len(assoc_rates), 1) if assoc_rates else 0
        bar = spike_bar(avg_a, 20, C.ORANGE)
        lines.append(f"  {C.ORANGE}⚡ ASC{C.RESET} {bar} {C.ORANGE}{avg_a*100:5.1f}%{C.RESET}")

    mem_rate = stats.get("memory_spike_rate", 0)
    if isinstance(mem_rate, torch.Tensor):
        mem_rate = mem_rate.item()
    bar = spike_bar(mem_rate * 0.3, 20, C.PURPLE)
    lines.append(f"  {C.PURPLE}⚡ MEM{C.RESET} {bar} {C.PURPLE}{mem_rate*100:5.1f}%{C.RESET}")

    ne = cfg.executive_layers
    offset = ns + na
    if len(spike_rates) > offset:
        exec_rates = spike_rates[offset:]
        avg_e = sum(exec_rates) / max(len(exec_rates), 1) if exec_rates else 0
        bar = spike_bar(avg_e, 20, C.GREEN)
        lines.append(f"  {C.GREEN}⚡ EXE{C.RESET} {bar} {C.GREEN}{avg_e*100:5.1f}%{C.RESET}")

    sp = stats.get("sparsity", 0)
    if isinstance(sp, torch.Tensor):
        sp = sp.item()
    sp_color = C.GREEN if sp > 0.85 else C.YELLOW if sp > 0.7 else C.RED
    lines.append(f"  {C.GREY}  SPR{C.RESET} {sp_color}{sp*100:.0f}%{C.RESET} {C.DIM}neurons silent{C.RESET}")
    lines.append(f"  {C.GREY}{'─' * 56}{C.RESET}")

    output = "\n".join(lines)
    n_lines = len(lines)
    sys.stdout.write(f"\033[{n_lines}A")
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


def init_live_display(cfg):
    for _ in range(7):
        print()


def render_spike_panel(stats, cfg):
    """Render a clean spike panel BELOW the generated text"""
    spike_rates = stats.get("spike_rates", [])
    if not spike_rates:
        return

    ns = cfg.sensory_layers + 1
    na = cfg.association_layers

    print(f"  {C.GREY}┌{'─' * 54}┐{C.RESET}")
    print(f"  {C.GREY}│{C.RESET} {C.BOLD}Neural Activity{C.RESET}{' ' * 38}{C.GREY}│{C.RESET}")
    print(f"  {C.GREY}├{'─' * 54}┤{C.RESET}")

    # Sensory
    if len(spike_rates) > 0:
        avg_s = sum(spike_rates[:ns]) / max(ns, 1)
        bar = spike_bar(avg_s, 25, C.CYAN)
        print(f"  {C.GREY}│{C.RESET} {C.CYAN}⚡ Sensory    {C.RESET} {bar} {C.CYAN}{avg_s*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")

    # Association
    if len(spike_rates) > ns:
        assoc_rates = spike_rates[ns:ns+na]
        avg_a = sum(assoc_rates) / max(len(assoc_rates), 1) if assoc_rates else 0
        bar = spike_bar(avg_a, 25, C.ORANGE)
        print(f"  {C.GREY}│{C.RESET} {C.ORANGE}⚡ Association{C.RESET} {bar} {C.ORANGE}{avg_a*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")

    # Memory
    mem_rate = stats.get("memory_spike_rate", 0)
    if isinstance(mem_rate, torch.Tensor):
        mem_rate = mem_rate.item()
    bar = spike_bar(min(mem_rate, 1.0), 25, C.PURPLE)
    print(f"  {C.GREY}│{C.RESET} {C.PURPLE}⚡ Memory     {C.RESET} {bar} {C.PURPLE}{mem_rate*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")

    # Executive
    offset = ns + na
    if len(spike_rates) > offset:
        exec_rates = spike_rates[offset:]
        avg_e = sum(exec_rates) / max(len(exec_rates), 1) if exec_rates else 0
        bar = spike_bar(avg_e, 25, C.GREEN)
        print(f"  {C.GREY}│{C.RESET} {C.GREEN}⚡ Executive  {C.RESET} {bar} {C.GREEN}{avg_e*100:5.1f}%{C.RESET} {C.GREY}│{C.RESET}")

    # Sparsity
    sp = stats.get("sparsity", 0)
    if isinstance(sp, torch.Tensor):
        sp = sp.item()
    sp_color = C.GREEN if sp > 0.85 else C.YELLOW if sp > 0.7 else C.RED
    silent = int(sp * 100)
    active = 100 - silent
    print(f"  {C.GREY}├{'─' * 54}┤{C.RESET}")
    print(f"  {C.GREY}│{C.RESET} {C.DIM}Sparsity:{C.RESET} {sp_color}{sp*100:.0f}%{C.RESET} silent  {C.DIM}({active}% neurons active per token){C.RESET}  {C.GREY}│{C.RESET}")
    print(f"  {C.GREY}└{'─' * 54}┘{C.RESET}")


def load_model(model_dir: str):
    from transformers import AutoTokenizer

    model_dir = Path(model_dir)

    # ── Smart checkpoint search ──
    # 1. If user gave a direct .pt file path
    if model_dir.is_file() and model_dir.suffix == ".pt":
        latest = model_dir
    else:
        latest = None

        # 2. Search in the given directory
        search_dirs = [model_dir]

        # 3. Also search in current working directory (where the script is run from)
        cwd = Path.cwd()
        if cwd != model_dir:
            search_dirs.append(cwd)

        # 4. Also search in the script's own directory
        script_dir = Path(__file__).resolve().parent
        if script_dir != cwd and script_dir != model_dir:
            search_dirs.append(script_dir)

        # Search order: nord_v4_latest.pt, nord_v4_final.pt, step checkpoints, legacy names
        checkpoint_names = [
            "nord_v4_latest.pt",
            "nord_v4_final.pt",
            "nord_500m_latest.pt",
            "nord_latest.pt",
        ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            # Try known names
            for name in checkpoint_names:
                p = search_dir / name
                if p.exists():
                    latest = p
                    break

            # Try step checkpoints
            if latest is None:
                ckpts = sorted(search_dir.glob("nord_v4_step_*.pt"))
                if ckpts:
                    latest = ckpts[-1]

            # Try any .pt file
            if latest is None:
                all_pt = sorted(search_dir.glob("*.pt"))
                if all_pt:
                    latest = all_pt[-1]

            if latest is not None:
                break

    if latest is None:
        print(f"  {C.RED}[✗] No checkpoint found!{C.RESET}")
        print(f"  {C.DIM}Searched in:{C.RESET}")
        for d in search_dirs:
            exists = "✓" if d.exists() else "✗"
            print(f"    [{exists}] {d}")
        print(f"\n  {C.DIM}Place your .pt file in the same folder as chat.py{C.RESET}")
        print(f"  {C.DIM}Or give the full path: /path/to/nord_v4_latest.pt{C.RESET}")
        sys.exit(1)

    print(f"  [*] Loading: {latest.name}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    saved_cfg = ckpt.get("config", {})
    cfg = NordConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16,
    )
    for k, v in saved_cfg.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.vocab_size < tokenizer.vocab_size:
        cfg.vocab_size = tokenizer.vocab_size

    model = NordModel(cfg)
    state = ckpt["model_state_dict"]
    filtered = {k: v for k, v in state.items()
                if "_v_mem_state" not in k and "_i_syn_state" not in k}
    model.load_state_dict(filtered, strict=False)
    model = model.to(cfg.device)
    model.eval()

    total = sum(p.numel() for p in model.parameters())
    print(f"  {C.GREEN}[✓]{C.RESET} Nord v4 loaded ({total/1e6:.1f}M params)")
    print(f"  {C.GREEN}[✓]{C.RESET} {model.count_params()}")

    return model, tokenizer, cfg


@torch.no_grad()
def generate_streaming(model, tokenizer, cfg, prompt: str,
                       max_tokens: int = 200, temperature: float = 0.85,
                       top_p: float = 0.9, repetition_penalty: float = 1.3,
                       enable_stdp: bool = False, live_spikes: bool = False):

    input_ids = tokenizer(
        prompt, return_tensors="pt",
        max_length=cfg.max_seq_len, truncation=True,
    ).input_ids.to(cfg.device)

    model.reset_state()

    generated = input_ids.clone()
    all_stats = {}
    token_count = 0

    t_start = time.time()

    sys.stdout.write(f"  {C.BOLD}Nord:{C.RESET} ")
    sys.stdout.flush()

    for i in range(max_tokens):
        context = generated[:, -cfg.max_seq_len:]

        if torch.cuda.is_available():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(cfg.dtype == torch.float16)):
                logits, stats = model(context, enable_stdp=enable_stdp)
        else:
            logits, stats = model(context, enable_stdp=enable_stdp)

        next_logits = logits[:, -1, :].float()

        if repetition_penalty != 1.0:
            for token_id in generated[0].unique():
                next_logits[0, token_id] /= repetition_penalty

        next_logits = next_logits / max(temperature, 0.01)

        probs = torch.softmax(next_logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = sorted_probs.cumsum(dim=-1)
        mask = cumsum - sorted_probs > top_p
        sorted_probs[mask] = 0
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

        token = sorted_idx[0, torch.multinomial(sorted_probs[0], 1)]
        generated = torch.cat([generated, token.reshape(1, 1)], dim=1)
        token_count += 1

        if token.item() == tokenizer.eos_token_id:
            break

        # ── Stream token ──
        decoded_token = tokenizer.decode([token.item()], skip_special_tokens=True)
        sys.stdout.write(decoded_token)
        sys.stdout.flush()

        all_stats = stats

    elapsed = time.time() - t_start
    tps = token_count / elapsed if elapsed > 0 else 0

    rep_score = 1.0
    if token_count > 5:
        out_ids = generated[0][input_ids.shape[1]:].tolist()
        unique = len(set(out_ids))
        rep_score = len(out_ids) / max(unique, 1)

    sp = all_stats.get("sparsity", 0)
    if isinstance(sp, torch.Tensor):
        sp = sp.item()

    print(f"\n  {C.GREY}[{token_count} tok, {elapsed:.1f}s, {tps:.1f} tok/s "
          f"[REP {rep_score:.1f}] [SPR {sp:.0%}]]{C.RESET}")

    if live_spikes and all_stats:
        render_spike_panel(all_stats, cfg)

    return all_stats


def print_stats(stats: dict, cfg: NordConfig):
    print(f"\n  {C.GREY}{'─' * 50}{C.RESET}")
    print(f"  {C.BOLD}Zone Statistics:{C.RESET}")

    spike_rates = stats.get("spike_rates", [])
    if spike_rates:
        print(f"    {C.DIM}Encoder:     {spike_rates[0]:.4f}{C.RESET}")
        for i in range(min(cfg.sensory_layers, len(spike_rates)-1)):
            rate = spike_rates[i+1]
            bar = spike_bar(rate, 15, C.CYAN)
            print(f"    {C.CYAN}Sensory[{i}]:{C.RESET}  {rate:.4f} {bar}")
        offset = cfg.sensory_layers + 1
        for i in range(cfg.association_layers):
            if offset + i < len(spike_rates):
                rate = spike_rates[offset+i]
                bar = spike_bar(rate, 15, C.ORANGE)
                print(f"    {C.ORANGE}Assoc[{i}]:{C.RESET}    {rate:.4f} {bar} {C.DIM}(MoE){C.RESET}")
        offset += cfg.association_layers
        for i in range(cfg.executive_layers):
            if offset + i < len(spike_rates):
                rate = spike_rates[offset+i]
                bar = spike_bar(rate, 15, C.GREEN)
                print(f"    {C.GREEN}Exec[{i}]:{C.RESET}     {rate:.4f} {bar}")

    print(f"\n  {C.BOLD}MoE Routing:{C.RESET}")
    expert_loads = stats.get("expert_loads", None)
    moe_entropy = stats.get("moe_route_entropy", None)

    # Also check for entropy with assoc_ prefix
    if moe_entropy is None:
        for key in stats:
            if "route_entropy" in key:
                moe_entropy = stats[key]
                break

    if expert_loads is not None:
        if isinstance(expert_loads, torch.Tensor):
            expert_loads = expert_loads.detach().cpu().tolist()
            if isinstance(expert_loads, float):
                expert_loads = [expert_loads]
        for e, load in enumerate(expert_loads):
            pct = load if isinstance(load, float) else float(load)
            bar = spike_bar(pct, 30, C.YELLOW, max_rate=0.5)
            print(f"    Expert {e}: {pct:.2%} {bar}")
    else:
        found = False
        # Search with ALL possible key patterns including assoc_ prefix
        for e in range(cfg.n_experts):
            load = None
            for key_pattern in [
                f"expert_{e}_load",
                f"expert_load_{e}",
                f"moe_expert_{e}",
            ]:
                # Direct match
                if key_pattern in stats:
                    load = stats[key_pattern]
                    break
                # Prefixed match (assoc_0_expert_0_load, etc.)
                for k, v in stats.items():
                    if key_pattern in k:
                        load = v
                        break
                if load is not None:
                    break

            if load is not None:
                found = True
                if isinstance(load, torch.Tensor): load = load.item()
                bar = spike_bar(load, 30, C.YELLOW, max_rate=0.5)
                print(f"    Expert {e}: {load:.2%} {bar}")

        if not found:
            # Last resort: scan all stats keys for anything with "expert" and "load"
            expert_data = {k: v for k, v in stats.items() if "expert" in k and "load" in k}
            if expert_data:
                found = True
                for k, v in sorted(expert_data.items()):
                    if isinstance(v, torch.Tensor): v = v.item()
                    bar = spike_bar(v, 30, C.YELLOW, max_rate=0.5)
                    name = k.split("_expert_")[-1] if "_expert_" in k else k
                    print(f"    {name}: {v:.2%} {bar}")

            if not found:
                moe_lb = stats.get("moe_lb_loss", None)
                if moe_lb is None:
                    for k, v in stats.items():
                        if "load_balance" in k or "moe_lb" in k:
                            moe_lb = v
                            break
                if moe_lb is not None:
                    if isinstance(moe_lb, torch.Tensor): moe_lb = moe_lb.item()
                    print(f"    {C.DIM}Load balance loss: {moe_lb:.4f}{C.RESET}")
                print(f"    {C.DIM}Per-expert loads not in top-level stats.{C.RESET}")
                print(f"    {C.DIM}They exist as assoc_N_expert_N_load — fixing...{C.RESET}")

    if moe_entropy is not None:
        if isinstance(moe_entropy, torch.Tensor): moe_entropy = moe_entropy.item()
        print(f"    Entropy: {moe_entropy:.3f}")

    mem_rate = stats.get("memory_spike_rate", None)
    if mem_rate is not None:
        if isinstance(mem_rate, torch.Tensor): mem_rate = mem_rate.item()
        gate = stats.get("gate_activity", 0)
        mix = stats.get("memory_mix", 0)
        if isinstance(gate, torch.Tensor): gate = gate.item()
        if isinstance(mix, torch.Tensor): mix = mix.item()
        bar = spike_bar(mem_rate * 0.3, 15, C.PURPLE)
        print(f"\n  {C.BOLD}Memory Cortex:{C.RESET}")
        print(f"    {C.PURPLE}Spike rate:{C.RESET}  {mem_rate:.4f} {bar}")
        print(f"    {C.PURPLE}Gate:{C.RESET}        {gate:.4f}")
        print(f"    {C.PURPLE}Mix weight:{C.RESET}  {mix:.4f}")

    sparsity = stats.get("sparsity", 0)
    if isinstance(sparsity, torch.Tensor): sparsity = sparsity.item()
    sp_color = C.GREEN if sparsity > 0.85 else C.YELLOW if sparsity > 0.7 else C.RED
    print(f"\n  Overall Sparsity: {sp_color}{sparsity:.1%}{C.RESET}")
    print(f"  {C.GREY}{'─' * 50}{C.RESET}")


def main():
    os.system('clear' if os.name != 'nt' else 'cls')

    print(f"""
  {C.CYAN}╔══════════════════════════════════════════════════════════╗{C.RESET}
  {C.CYAN}║{C.RESET}  {C.BOLD}⚡ PROJECT NORD v4.2 — Brain-Inspired SNN Chat{C.RESET}         {C.CYAN}║{C.RESET}
  {C.CYAN}║{C.RESET}  {C.DIM}618M params │ Spike-driven │ Zonal architecture{C.RESET}       {C.CYAN}║{C.RESET}
  {C.CYAN}╚══════════════════════════════════════════════════════════╝{C.RESET}
""")

    default_dir = "nord_v4_700m"
    print(f"  Model directory?")
    print(f"  {C.DIM}(Enter = {default_dir}){C.RESET}")
    model_input = input("  Path: ").strip()
    model_dir = model_input if model_input else default_dir

    model, tokenizer, cfg = load_model(model_dir)

    stdp_enabled = False
    live_spikes = False
    max_tokens = 200
    temperature = 0.85
    top_p = 0.9
    rep_penalty = 1.3
    last_stats = {}

    print(f"\n  {C.DIM}Type /help for commands{C.RESET}")
    print(f"  {C.GREY}{'─' * 50}{C.RESET}\n")

    while True:
        try:
            user = input(f"  {C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {C.DIM}Goodbye!{C.RESET}")
            break

        if not user:
            continue

        cmd = user.lower().split()

        if cmd[0] == "/quit":
            break
        elif cmd[0] == "/help":
            print(f"""
  {C.BOLD}Commands:{C.RESET}
    {C.CYAN}/tokens N{C.RESET}     — Max response tokens (current: {max_tokens})
    {C.CYAN}/temp F{C.RESET}       — Temperature (current: {temperature})
    {C.CYAN}/rep F{C.RESET}        — Repetition penalty (current: {rep_penalty})
    {C.CYAN}/stdp on|off{C.RESET}  — Toggle online learning ({C.GREEN if stdp_enabled else C.RED}{'ON' if stdp_enabled else 'OFF'}{C.RESET})
    {C.CYAN}/live on|off{C.RESET}  — Live spike visualization ({C.GREEN if live_spikes else C.RED}{'ON' if live_spikes else 'OFF'}{C.RESET})
    {C.CYAN}/stats{C.RESET}        — Zone & MoE statistics
    {C.CYAN}/memory{C.RESET}       — Memory cortex state
    {C.CYAN}/expert{C.RESET}       — Expert routing breakdown
    {C.CYAN}/reset{C.RESET}        — Clear working memory
    {C.CYAN}/quit{C.RESET}         — Exit""")
            continue
        elif cmd[0] == "/tokens":
            if len(cmd) > 1:
                try:
                    max_tokens = int(cmd[1])
                    print(f"  {C.GREEN}[✓]{C.RESET} Max tokens: {max_tokens}")
                except ValueError:
                    print(f"  {C.RED}[✗]{C.RESET} Usage: /tokens 300")
            else:
                print(f"  Max tokens: {max_tokens}")
            continue
        elif cmd[0] == "/temp":
            if len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    print(f"  {C.GREEN}[✓]{C.RESET} Temperature: {temperature}")
                except ValueError:
                    print(f"  {C.RED}[✗]{C.RESET} Usage: /temp 0.7")
            else:
                print(f"  Temperature: {temperature}")
            continue
        elif cmd[0] == "/rep":
            if len(cmd) > 1:
                try:
                    rep_penalty = float(cmd[1])
                    print(f"  {C.GREEN}[✓]{C.RESET} Repetition penalty: {rep_penalty}")
                except ValueError:
                    print(f"  {C.RED}[✗]{C.RESET} Usage: /rep 1.3")
            else:
                print(f"  Repetition penalty: {rep_penalty}")
            continue
        elif cmd[0] == "/stdp":
            if len(cmd) > 1 and cmd[1] == "on":
                stdp_enabled = True
                print(f"  {C.GREEN}[⚙] STDP enabled{C.RESET}")
            elif len(cmd) > 1 and cmd[1] == "off":
                stdp_enabled = False
                print(f"  {C.YELLOW}[⚙] STDP disabled{C.RESET}")
            else:
                print(f"  STDP: {'ON' if stdp_enabled else 'OFF'}")
            continue
        elif cmd[0] == "/live":
            if len(cmd) > 1 and cmd[1] == "on":
                live_spikes = True
                print(f"  {C.GREEN}[⚙] Live spike visualization ON{C.RESET}")
            elif len(cmd) > 1 and cmd[1] == "off":
                live_spikes = False
                print(f"  {C.YELLOW}[⚙] Live spike visualization OFF{C.RESET}")
            else:
                print(f"  Live spikes: {'ON' if live_spikes else 'OFF'}")
            continue
        elif cmd[0] == "/stats":
            print_stats(last_stats, cfg)
            continue
        elif cmd[0] == "/memory":
            mem_rate = last_stats.get("memory_spike_rate", "N/A")
            gate = last_stats.get("gate_activity", "N/A")
            mix = last_stats.get("memory_mix", "N/A")
            if isinstance(mem_rate, torch.Tensor): mem_rate = f"{mem_rate.item():.4f}"
            if isinstance(gate, torch.Tensor): gate = f"{gate.item():.4f}"
            if isinstance(mix, torch.Tensor): mix = f"{mix.item():.4f}"
            print(f"  {C.PURPLE}Memory:{C.RESET} rate={mem_rate}, gate={gate}, mix={mix}")
            continue
        elif cmd[0] == "/expert":
            # Search all stats keys for expert load data
            expert_data = {k: v for k, v in last_stats.items() if "expert" in k and "load" in k}
            if expert_data:
                for k, v in sorted(expert_data.items()):
                    if isinstance(v, torch.Tensor): v = v.item()
                    bar = spike_bar(v, 30, C.YELLOW, max_rate=0.5)
                    # Clean up key name for display
                    display_name = k.replace("assoc_", "A").replace("_load", "")
                    print(f"    {display_name}: {v:.2%} {bar}")
            else:
                print(f"    {C.DIM}No expert load data in stats{C.RESET}")
                moe_keys = [k for k in last_stats.keys() if "moe" in k or "expert" in k]
                if moe_keys:
                    print(f"    {C.DIM}Related keys: {moe_keys}{C.RESET}")
            continue
        elif cmd[0] == "/reset":
            model.reset_state()
            print(f"  {C.GREEN}[⚙] Working memory cleared{C.RESET}")
            continue

        last_stats = generate_streaming(
            model, tokenizer, cfg, user,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            enable_stdp=stdp_enabled,
            live_spikes=live_spikes,
        )


if __name__ == "__main__":
    main()
