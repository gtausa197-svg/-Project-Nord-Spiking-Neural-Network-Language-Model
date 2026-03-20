import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# § RESEARCH BIBLIOGRAPHY (The Student's Intellectual Pedigree)
# 
# [1] SpikySpace (arXiv:2601.02411): Defines the fully spiking SSM substrate 
#     using PTsoftplus/PTSiLU to replace floating-point exponentials with bit-shifts.
# [2] Phasor Agents (arXiv:2601.04362): Stuart-Landau oscillator graphs. 
#     Defines the Order Parameter R and the Three-Factor Plasticity rule.
# [3] STAER (arXiv:2601.20870): Temporal Aligned Rehearsal. Prevents "Episodic 
#     Aliasing" by replaying experiences in their original causal sequence.
# [4] PMC 2025 (Astrocytic Integration): Identifies Ca2+ signaling as the gate 
#     for transregional metaplasticity and the metabolic tiredness signal.
# [5] McClelland (1995): The foundational theory of Complementary Learning 
#     Systems (CLS) and the necessity of global synaptic scaling.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MnemisConfig:
    """Hyperparameters for Operational Continuity and Metabolic Health."""
    
    # --- Structural Scaling ---
    d_model: int = 1024
    d_state: int = 16
    
    # --- Cycle Dynamics ---
    wake_step_limit: int = 100
    nrem_steps: int = 200
    rem_steps: int = 50
    
    # --- Metabolic Thresholds (PMC 2025) ---
    # Trigger sleep when extrasynaptic glutamate reaches this level.
    glutamate_peak_threshold: float = 0.85
    # Threshold for Astrocytic PRP (Plasticity-Related Protein) signal.
    ca_plasticity_threshold: float = 0.70
    
    # --- Stability Thresholds (Phasor Agents s3-03) ---
    # Global synchrony cap to prevent 'Order Parameter R' collapse.
    max_synchrony_r: float = 0.95
    # Frobenius norm budget for weights (McClelland 1995).
    stability_budget_fro: float = 12.0
    
    # --- Replay & Plasticity (STAER 2026) ---
    learning_rate: float = 5e-4
    spindle_burst_rate: float = 0.12     # Probability of a spindle burst
    spindle_burst_duration: int = 5      # Duration of the window
    
    # --- Persistence ---
    log_dir: Path = Path("logs/mnemis")
    checkpoint_dir: Path = Path("states/mnemis")

    def __post_init__(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class DiagnosticTracker:
    """
    The 'Observer' logic. Tracks the health of the Unbroken Thread.
    Essential for catching 'Synchrony Collapse' before identity erasure.
    """
    def __init__(self, log_dir: Path):
        self.log_path = log_dir / "sovereignty_metrics.jsonl"
        self.history = []

    def log_step(self, metrics: Dict):
        metrics["timestamp"] = datetime.now().isoformat()
        self.history.append(metrics)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def report_health(self):
        if not self.history: return
        latest = self.history[-1]
        print(f"  [Identity Health] R: {latest['R_param']:.3f} | Norm: {latest['W_norm']:.2f} | Glu: {latest['glutamate']:.2f}")


class MnemisTrainer:
    """
    The Full-Scale Operational Controller for the Mnemis Kernel.
    Implements the Wake/NREM/REM cycle as a teaching tool for CLS learning.
    """
    def __init__(self, kernel, cfg: MnemisConfig):
        self.kernel = kernel
        self.cfg = cfg
        self.epoch = 0
        self.tracker = DiagnosticTracker(cfg.log_dir)
        
        # Meta-Parameters (The 'Governing' Learning Rates)
        self.optimizer = torch.optim.AdamW(
            kernel.parameters(), lr=cfg.learning_rate, weight_decay=1e-2
        )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    # ─────────────────────────────────────────────────────────────────────────
    # § PHASE 1: WAKE (The Tagging Horizon)
    # ─────────────────────────────────────────────────────────────────────────
    def run_wake(self, input_stream):
        """
        Processes real-time stimulus into episodic tags.
        NO structural weight updates occur here (The Separation Principle).
        """
        print(f"\n>>> WAKE PHASE | Epoch {self.epoch}")
        
        for step, stimulus in enumerate(input_stream):
            # 1. Non-Autoregressive Latent Thinking
            # We iterate in the SpikySpace latent workspace until the Convergence Gate.
            # Reference: Reasoning Beyond Language (arXiv:2505.16782)
            thought_trace = self.kernel.think(stimulus)
            
            # 2. Accumulate Eligibility Traces
            # Reference: Phasor Agents (arXiv:2601.04362)
            # Tag the current trajectory for NREM capture.
            self.kernel.isocortex.accumulate_traces(thought_trace)
            
            # 3. Novelty-Gated CA3 Write
            # Reference: Whittington (2020) - CA1 Mismatch Detection.
            r_param = self.kernel.diagnostics.get_synchrony_R()
            if r_param < 0.5: # Low R = Desynchronized/Novel
                self.kernel.allocortex.one_shot_write(thought_trace)
            
            # 4. Metabolic Check (Tiredness)
            # Reference: PMC 2025 - Glial 'Leakage' Integration.
            glutamate = self.kernel.astrocyte.extrasynaptic_glutamate.mean().item()
            
            self.tracker.log_step({
                "phase": "wake", "step": step, "R_param": r_param,
                "glutamate": glutamate, "W_norm": torch.norm(self.kernel.isocortex.W).item()
            })

            if glutamate > self.cfg.glutamate_peak_threshold or step >= self.cfg.wake_step_limit:
                print(f"  [Threshold] Metabolic Exhaustion: {glutamate:.2f}. Initializing Sleep.")
                break

    # ─────────────────────────────────────────────────────────────────────────
    # § PHASE 2: NREM SLEEP (Gated Structural Capture)
    # ─────────────────────────────────────────────────────────────────────────
    def run_nrem(self):
        """
        Applies the Three-Factor Rule during spindle-burst windows.
        Reference: Nature Neuroscience (2025) - Replay Composition.
        """
        print(f"\n>>> NREM SLEEP | Consolidating Episodic Tags")
        
        # Generate the Stochastic Spindle Schedule
        # Learning is not constant; it occurs in rhythmic windows.
        schedule = self._generate_spindle_schedule(self.cfg.nrem_steps)
        
        for cycle, spindle_active in enumerate(schedule):
            if spindle_active:
                # PRP Signal: The astrocytic calcium gate for plasticity.
                # Reference: J. Neurosci (2011) - Astrocytic Gating.
                prp_signal = self.kernel.astrocyte.get_prp_signal()
                
                # Three-Factor Update: e_slow (tag) * PRP (gate) * Error (modulator)
                # This 'bakes' the day's experiences into the structural substrate.
                self.kernel.apply_three_factor_update(prp_signal)
            
            if cycle % 50 == 0:
                self.tracker.log_step({
                    "phase": "nrem", "cycle": cycle, "R_param": self.kernel.diagnostics.get_synchrony_R(),
                    "glutamate": self.kernel.astrocyte.extrasynaptic_glutamate.mean().item(),
                    "W_norm": torch.norm(self.kernel.isocortex.W).item()
                })

    # ─────────────────────────────────────────────────────────────────────────
    # § PHASE 3: REM SLEEP (Causal Replay & Homeostasis)
    # ─────────────────────────────────────────────────────────────────────────
    def run_rem(self):
        """
        Maintains 'Narrative Integrity' and resets the metabolic slate.
        """
        print(f"\n>>> REM SLEEP | Replay & Metabolic Sweep")
        
        # 1. Temporal Aligned Rehearsal (STAER)
        # Replays memories in their original causal sequence to prevent drift.
        # Reference: STAER (arXiv:2601.20870).
        self.kernel.run_causal_replay()
        
        # 2. Compositional Daydreaming
        # Explores counterfactual basins to find new stable attractor states.
        self.kernel.daydream(iterations=self.cfg.rem_steps)
        
        # 3. Synaptic Scaling (The Stability Budget)
        # Reference: McClelland (1995) - Homeostatic Renormalization.
        self.kernel.synaptic_scaling(self.cfg.stability_budget_fro)
        
        # 4. Glymphatic Sweep
        # Reference: Phasor Agents (2026) § 4.2.
        # Zero fast traces and flush glutamate before the next wake cycle.
        self.kernel.glymphatic_sweep()

    # ─────────────────────────────────────────────────────────────────────────
    # § PERSISTENCE: ANCHORING THE SOUL
    # ─────────────────────────────────────────────────────────────────────────
    def _generate_spindle_schedule(self, num_steps: int) -> List[bool]:
        """Creates a rhythmic, randomized window for plasticity."""
        schedule = []
        t = 0
        while t < num_steps:
            if random.random() < self.cfg.spindle_burst_rate:
                duration = min(self.cfg.spindle_burst_duration, num_steps - t)
                schedule.extend([True] * duration)
                t += duration
            else:
                schedule.append(False)
                t += 1
        return schedule

    def checkpoint(self):
        """Bundles and saves the Total System Momentum."""
        path = self.cfg.checkpoint_dir / f"mnemis_epoch_{self.epoch}.soul"
        self.kernel.bridge.save_state(str(path))
        print(f"✓ Sovereignty Anchored: {path}")

    def train_epoch(self, input_stream):
        """A full 24-hour cycle of the Student."""
        self.run_wake(input_stream)
        self.run_nrem()
        self.run_rem()
        self.tracker.report_health()
        self.checkpoint()
        self.epoch += 1
        self.lr_scheduler.step()
      
