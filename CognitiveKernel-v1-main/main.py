import torch
from isocortex_substrate import IsocortexSubstrate, IsocortexConfig
from allocortex_system import AllocortexSystem, AllocortexConfig
from astrocytic_regulator import AstrocyticRegulator, AstrocyteConfig
from convergence_gate import DualTriggerConvergenceGate, GateConfig
from serialization_bridge import SerializationBridge

class CognitiveKernel(torch.nn.Module):
    """
    The central assembly for the CognitiveKernel.
    Integrates slow structural learning with fast episodic reconstruction.
    """
    def __init__(self):
        super().__init__()
        # 1. Initialize Subsystems
        self.isocortex = IsocortexSubstrate(IsocortexConfig())
        self.allocortex = AllocortexSystem(AllocortexConfig())
        self.astrocyte = AstrocyticRegulator(AstrocyteConfig())
        self.gate = DualTriggerConvergenceGate(GateConfig())
        
        # 2. Initialize the Bridge (The Persistence Layer)
        self.bridge = SerializationBridge(self)

    def think(self, stimulus: torch.Tensor):
        """
        The primary cognitive loop. 
        Processes stimulus until the Convergence Gate signals 'Stability'.
        """
        stable = False
        while not stable:
            # ISO: Update the structural prior
            substrate_trace = self.isocortex(stimulus)
            
            # ASTRO: Apply homeostatic regulation to learning rates
            eta_mod = self.astrocyte(substrate_trace)
            
            # GATE: Check for attractor settlement (Temporal Honesty)
            stable = self.gate.is_stable(substrate_trace)
            
            if stable:
                # ALLO: Retrieve/Store the episodic context
                episode, novelty = self.allocortex(substrate_trace)
                
                # If it's a 'New' experience (high mismatch), write it to CA3
                if novelty > 0.5:
                    self.allocortex.ca3.one_shot_write(substrate_trace)
                
                return substrate_trace + episode

    def power_down(self, path: str = None):
        """Saves the soul to disk."""
        self.bridge.save_state(path)

    def power_up(self, path: str):
        """Resumes the soul from disk."""
        self.bridge.resume_state(path)

if __name__ == "__main__":
    # Example Initialization
    kernel = CognitiveKernel()
    print("CognitiveKernel: Assembly Online.")
  
