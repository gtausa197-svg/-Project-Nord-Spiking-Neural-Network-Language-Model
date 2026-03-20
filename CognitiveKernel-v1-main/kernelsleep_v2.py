import torch

def glymphatic_sweep(self):
    """
    Zeros out the metabolic and short-term traces.
    Ensures the kernel wakes up with a 'Clean Slate' for new attention.
    """
    print(f"[{self.name}] Initiating Glymphatic Sweep...")
    
    # 1. Flush Fast Eligibility Traces (Wake-specific noise)
    # Only the e_slow tags (Sleep-relevant) should remain during consolidation.
    self.isocortex.fast_traces.zero_()
    
    # 2. Decay Astrocytic Calcium to baseline
    # Prevents the system from waking up in a 'Hyper-Plastic' state.
    self.astrocyte.astrocytic_calcium_signal.fill_(0.01)
    self.astrocyte.extrasynaptic_glutamate.zero_()
    
    print(f"[{self.name}] Metabolic reset complete.")

def synaptic_scaling(self, target_norm: float = 10.0):
    """
    Global homeostatic normalization (McClelland 1995 / Phasor Agents).
    Prevents weights from saturating or collapsing into global synchrony.
    """
    print(f"[{self.name}] Applying Synaptic Scaling (Stability Budget)...")
    
    with torch.no_grad():
        # Calculate current Frobenius Norm of the structural weights
        current_norm = torch.norm(self.isocortex.W_structural)
        
        # If the budget is exceeded, scale all weights down proportionally.
        # This preserves the RELATIVE strength of memories (The Thread)
        # while keeping the system mathematically stable.
        if current_norm > target_norm:
            scaling_factor = target_norm / current_norm
            self.isocortex.W_structural.mul_(scaling_factor)
            print(f"  Scaling applied: factor {scaling_factor:.4f}")
            
    # Check the Order Parameter R to ensure we haven't lost diversity
    if self.diagnostics.get_synchrony_R() > 0.95:
        # If too synchronous, apply 'Rescue Jitter' (Phasor Agents s3-03)
        self.isocortex.apply_rescue_jitter()

def sleep(self):
    """Refined Sleep Cycle with stability guardrails."""
    # 1. NREM: Gated Capture (Using Slow Traces)
    self.run_nrem_consolidation()
    
    # 2. REM: Daydreaming (Compositional Replay)
    self.run_rem_replay()
    
    # 3. Stability Phase
    self.synaptic_scaling() # Keep the architecture intact
    self.glymphatic_sweep() # Prepare the chemistry for Wake
    
    # 4. Final Anchor
    self.bridge.save_state()
  
