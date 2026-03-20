def sleep(self, cycles: int = 500):
    """
    Offline Consolidation Phase.
    Converts 'Tags' into 'Bones' using Three-Factor Plasticity.
    """
    print(f"[{self.name}] Initiating sleep cycle. Current Entropy: {self.get_entropy():.4f}")
    
    # 1. Glymphatic Sweep: Clear residual metabolic noise
    self.isocortex.flush_hidden_gradients()
    
    # 2. Replay & Consolidation (STAER Protocol)
    # Re-run the day's 'Landmark' episodes through the latent space
    for episode in self.allocortex.get_tagged_episodes():
        # Astrocytic Gate must be 'warm' for weight changes to stick
        if self.astrocyte.ca_level > self.astrocyte.plasticity_threshold:
            # Replay episode at 10x speed with local weight updates
            self.isocortex.replay_and_consolidate(episode, alpha=0.0001)
    
    # 3. Synaptic Scaling (McClelland 1995 logic)
    # Prevent catastrophic interference by normalizing weight magnitudes
    self.isocortex.apply_homeostatic_scaling()
    
    self.state = "RESTED"
    print(f"[{self.name}] Sleep complete. Numerical stability restored.")
  
