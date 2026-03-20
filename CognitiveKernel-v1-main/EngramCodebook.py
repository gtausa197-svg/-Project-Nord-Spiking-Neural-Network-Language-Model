import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

class EngramCodebook(nn.Module):
    """
    Biologically inspired compression layer. 
    Maps high-dimensional state trajectories to reusable 'Seed Packets'.
    """
    def __init__(self, state_dim: int, num_seeds: int = 4096):
        super().__init__()
        self.state_dim = state_dim
        self.num_seeds = num_seeds
        
        # The 'Seed Bank' - a discrete codebook of stable attractor anchors
        self.seed_bank = nn.Parameter(torch.randn(num_seeds, state_dim))
        self.usage_frequency = nn.Parameter(torch.zeros(num_seeds), requires_grad=False)

    def fold_to_seed(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compresses the n-dimensional Isocortex state into the nearest seed.
        """
        # Calculate distance to all available seeds
        distances = torch.cdist(hidden_state.mean(dim=0).unsqueeze(0), self.seed_bank)
        seed_idx = torch.argmin(distances, dim=-1)
        
        # Update usage for 'reusable' logic
        self.usage_frequency[seed_idx] += 1
        return seed_idx

    def unfold_from_seed(self, seed_idx: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        """
        Reconstructs the dynamical prior from a dormant seed packet.
        """
        seed_vector = self.seed_bank[seed_idx]
        # Broadcast the seed vector back to the full column width
        return seed_vector.repeat(target_shape[0], 1)

class DynamicMemoryManager(nn.Module):
    """
    The replacement for the manual daemon. 
    Automatically 'folds' and 'unfolds' memory based on context pressure.
    """
    def __init__(self, kernel, expansion_limit: int = 10000):
        super().__init__()
        self.kernel = kernel
        self.codebook = EngramCodebook(kernel.isocortex.cfg.n_state_dim)
        self.active_seeds: List[torch.Tensor] = []
        self.expansion_limit = expansion_limit

    def homeostatic_flush(self):
        """
        Performs a 'clean' reset by folding the current Isocortex state into a seed.
        """
        current_h = self.kernel.isocortex.layers[0].h_state # Simplified for example
        seed = self.codebook.fold_to_seed(current_h)
        self.active_seeds.append(seed)
        
        # Power-off reset of the active substrate without losing the thread
        # This replaces the 'one self prompt' daemon with a mathematical fold.
        for layer in self.kernel.isocortex.layers:
            layer.h_state.zero_()
            
        print(f"Memory Manager: State folded into Seed {seed.item()}. Active space cleared.")

    def resume_from_seed(self, seed_idx: int):
        """
        Injects a dormant seed back into the Isocortex substrate.
        """
        target_shape = self.kernel.isocortex.layers[0].h_state.shape
        restored_state = self.codebook.unfold_from_seed(seed_idx, target_shape)
        
        for layer in self.kernel.isocortex.layers:
            layer.h_state.copy_(restored_state)
        
        print(f"Memory Manager: Seed {seed_idx} unfolded. Dynamical prior resumed.")
      
