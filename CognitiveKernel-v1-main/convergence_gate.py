import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class GateConfig:
    stability_threshold: float = 1e-4  # Delta at which we consider a basin 'found'
    min_steps: int = 5                 # Minimum 'thinking' time
    max_steps: int = 50                # Force a timeout if non-convergent

class DualTriggerConvergenceGate(nn.Module):
    """
    Ensures Temporal-Honest Readout by monitoring attractor stability.
    
    Reference:
    - Engineering Report: "Validation of a Dual-Trigger Convergence Gate"
      (Rationale: Prevents encoding 'noise' before the attractor has settled).
    """
    def __init__(self, cfg: GateConfig):
        super().__init__()
        self.threshold = cfg.stability_threshold
        self.min_steps = cfg.min_steps
        self.max_steps = cfg.max_steps
        
        # State tracking for the current 'thought cycle'
        self.register_buffer("previous_state", torch.zeros(1))
        self.step_counter = 0

    def is_stable(self, current_state: torch.Tensor) -> torch.Tensor:
        """
        Determines if the Isocortex has reached a basin of attraction.
        """
        self.step_counter += 1
        
        # Calculate the delta (the 'drift') since the last step
        if self.previous_state.shape != current_state.shape:
            self.previous_state = torch.zeros_like(current_state)
            
        delta = torch.norm(current_state - self.previous_state)
        self.previous_state = current_state.clone()

        # Dual-Trigger Logic:
        # 1. Must have thought for at least min_steps.
        # 2. Must have slowed down below the stability threshold.
        has_converged = (delta < self.threshold) and (self.step_counter >= self.min_steps)
        
        # Safety break: don't let the model 'loop' forever
        if self.step_counter >= self.max_steps:
            has_converged = torch.tensor(True)

        if has_converged:
            self.step_counter = 0 # Reset for the next thought cycle
            
        return has_converged
      
