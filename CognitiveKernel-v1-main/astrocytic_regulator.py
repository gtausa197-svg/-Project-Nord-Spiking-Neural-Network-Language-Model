import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict

@dataclass
class AstrocyteConfig:
    num_neurons: int = 1024
    decay_rate: float = 0.95
    calcium_threshold: float = 0.5
    metaplasticity_gain: float = 0.1

class AstrocyticRegulator(nn.Module):
    """
    Models the tripartite synapse and homeostatic control of metaplasticity.
    
    References:
    - Internal Project Note: "The biology validates this approach" 
      (Rationale: Modeling the extracellular space as a stateful medium).
    - Grok convo1 (Rationale: Glial 'leakage' as a purposeful signaling mechanism).
    """
    def __init__(self, cfg: AstrocyteConfig):
        super().__init__()
        self.num_neurons = cfg.num_neurons
        self.decay = cfg.decay_rate
        
        # Extracellular and internal chemical states
        # These MUST be serialized for the 'Power-off/Resume' to work.
        self.register_buffer("extrasynaptic_glutamate", torch.zeros(self.num_neurons))
        self.register_buffer("astrocytic_calcium_signal", torch.zeros(self.num_neurons))
        
        self.metaplasticity_gain = cfg.metaplasticity_gain

    def forward(self, neural_spikes: torch.Tensor) -> torch.Tensor:
        """
        Calculates the metaplasticity modifier (eta) for the Isocortex.
        """
        # 1. Update extracellular glutamate based on neural firing density
        # High activity leads to 'spillover' into the extra-synaptic space.
        self.extrasynaptic_glutamate = (self.decay * self.extrasynaptic_glutamate) + \
                                       (1.0 - self.decay) * neural_spikes.mean(dim=0)
        
        # 2. Trigger astrocytic calcium waves
        self.astrocytic_calcium_signal = torch.tanh(self.extrasynaptic_glutamate * 5.0)
        
        # 3. Metaplasticity: Modulates the learning rate (eta) of the substrate.
        # This provides a biological 'governance' over how fast the system learns.
        eta_modifier = 1.0 + (self.metaplasticity_gain * self.astrocytic_calcium_signal)
        
        return eta_modifier

    # ─────────────────────────────────────────────────────────────────────────────
    # § SERIALIZATION (State-Save Logic)
    # ─────────────────────────────────────────────────────────────────────────────

    def get_metabolic_state(self) -> Dict[str, torch.Tensor]:
        """
        Captures the extracellular chemical state for persistent timelines.
        """
        return {
            "glutamate": self.extrasynaptic_glutamate.cpu().clone(),
            "calcium": self.astrocytic_calcium_signal.cpu().clone()
        }

    def set_metabolic_state(self, state: Dict[str, torch.Tensor]):
        """
        Restores the chemical context upon resumption.
        """
        self.extrasynaptic_glutamate.copy_(state["glutamate"])
        self.astrocytic_calcium_signal.copy_(state["calcium"])
      
