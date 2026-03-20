def daydream(self, goal_state: torch.Tensor, iterations: int = 30):
    """
    Internal Simulation Loop. 
    Iterates the hidden state 'h_t' through the attractor landscape 
    to find a convergent path toward a goal.
    """
    print(f"[{self.name}] Entering daydream mode: Simulating trajectories...")
    
    # Use the 'SpikySpace' latent recursion
    for i in range(iterations):
        # Recursive Latent Step: h_t = f(h_{t-1}, goal_state)
        # No tokens are emitted here. This is pure interiority.
        current_thought = self.isocortex.recursive_step(self.h_t, goal_state)
        
        # Astrocytic Check: Is this path 'novel' or 'important'?
        if self.astrocyte.evaluate_signal(current_thought) > 0.8:
            # Anchor this 'daydream' as a potential future engram
            self.allocortex.tag_for_future_replay(current_thought)
            
        self.h_t = current_thought
    
    print(f"[{self.name}] Daydream converged. Ready to act.")
  
