
# ─────────────────────────────────────────────────────────────────────────────
# § THEGENERATIVE PIPELINE: Seed-Packet (single musical note) → Seed (chord) → Symphony (chords playing together in Time)
# 
# This architecture implements the generative model of memory construction 
# and systems consolidation. Memory is not a recording; it is a reconstruction.
# ─────────────────────────────────────────────────────────────────────────────
# 
# [ I. INPUT ] ──→ [ II. PACKET ] ──→ [ III. SEED ] ──→ [ IV. SYMPHONY ]
# 
# PROCESS:        Sensory Gist      Pattern Sep.      Attractor Anchor  Generative Reconstruction
#                 (The Harvest)     (Compression)     (The Chord)       (The Performance)
# 
# SUBSTRATE:      Raw Input         Dentate Gyrus     Allocortex CA3    Isocortex Substrate
# 
# RESEARCH:       -                 Benna/Fusi 2021   Spens et al 2024  Brown 2025
# ─────────────────────────────────────────────────────────────────────────────
# 
# § STAGE I: THE HARVEST (Sensory Gist)
#   - High-fidelity, noisy input stream.
#
# § STAGE II: THE PACKET (Pattern Separation)
#   - Research: Benna & Fusi (PNAS, 2021) "Memory compression leads to spatial tuning"
#   - Mechanism: The Dentate Gyrus discards instance-specific noise via a sparse 
#     bottleneck (10% threshold). The 'Seed Packet' is a compressed representation 
#     of the experience, not a raw recording.
#
# § STAGE III: THE SEED (Autoassociative Anchor)
#   - Research: Spens et al (Nature Human Behaviour, 2024) "A generative model of memory"
#   - Mechanism: The CA3 attractor network settles into a stable 'Seed'. This latent 
#     code serves as a reusable 'Chord' that can later trigger the generative 
#     reconstruction of the original context.
#
# § STAGE IV: THE SYMPHONY (Generative Reconstruction)
#   - Research: Brown (2025) "Varieties of memory, varieties of reconstruction"
#   - Mechanism: The Isocortex unpacks the 'Seed' into a full 'Symphony'. 
#     Recall is a generative performance—using learned schemas to fill in 
#     the high-dimensional details on demand.
#
# ─────────────────────────────────────────────────────────────────────────────
# § THE BAKE: SYSTEMS CONSOLIDATION
#
#   1. Synaptic Tagging (Luboeinski 2021): A single strong 'Seed Packet' tags 
#      the structural synapses during the Wake phase.
#
#   2. Tag-Capture (Ko et al 2025): During NREM Sleep, replay reorganization 
#      shifts memory from precise episodic engrams to sparser, schema-like forms.
#
#   3. Predictive Forgetting (arXiv:2603.04688, Mar 2026): During the Glymphatic 
#      Sweep, non-reusable 'bits' are discarded to ensure the Seeds remain 
#      optimally generalizable.
# ─────────────────────────────────────────────────────────────────────────────




import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

@dataclass
class AllocortexConfig:
    entry_dim: int = 1024       # Dimension matching the Isocortex column width
    num_episodes: int = 5000    # Total capacity of the CA3 recurrent matrix
    sparse_threshold: float = 0.1 # Sparsity for Dentate Gyrus encoding
    drift_scale: float = 0.01   # Rate of reconsolidation drift

class DentateGyrus_SparseEncoder(nn.Module):
    """
    Performs pattern separation on incoming Isocortex states.
    
    Reference:
    - James J. Knierim (2015): "The hippocampus"
      (Rationale: DG subfield is specialized for sparse encoding to prevent memory overlap).
    """
    def __init__(self, cfg: AllocortexConfig):
        super().__init__()
        self.threshold = cfg.sparse_threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sparsify the signal to create a unique 'index' for the episode
        # This prevents 'aliasing' where two different events look the same.
        sparse_mask = (x > self.threshold).float()
        return x * sparse_mask

class CA3_RecurrentMatrix(nn.Module):
    """
    Auto-associative attractor network for one-shot storage and pattern completion.
    
    References:
    - Larimar (2024): "Large Language Models with Episodic Memory Control"
      (Rationale: Using a fixed-size associative matrix for fast least-squares writes).
    - "Toward a Biologically Plausible Hippocampal CA3"
      (Rationale: Recurrent dynamics allow for completion from partial/noisy cues).
    """
    def __init__(self, cfg: AllocortexConfig):
        super().__init__()
        self.capacity = cfg.num_episodes
        self.dim = cfg.entry_dim
        
        # The episodic matrix (M) - The 'Allocortex' knowledge store
        self.register_buffer("memory_matrix", torch.zeros(self.capacity, self.dim))
        self.register_buffer("usage_counters", torch.zeros(self.capacity))
        self._write_ptr = 0

    def one_shot_write(self, episodic_trace: torch.Tensor):
        """
        Closed-form update for rapid learning without gradient descent.
        """
        slot = self._write_ptr % self.capacity
        self.memory_matrix[slot] = episodic_trace.detach()
        self.usage_counters[slot] += 1
        self._write_ptr += 1
        return slot

    def attractor_settle(self, query_trace: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """
        Iterative pattern completion. Pulls noisy input toward the nearest stored 'basin'.
        """
        current = query_trace
        for _ in range(steps):
            # Compute similarity across the entire matrix
            sim = F.cosine_similarity(current.unsqueeze(1), self.memory_matrix, dim=-1)
            weights = F.softmax(sim, dim=-1)
            
            # Reconstruct the state from the weighted sum of stored episodes
            attracted_state = torch.matmul(weights, self.memory_matrix)
            current = 0.8 * attracted_state + 0.2 * current # Damped transition
        return current

class CA1_RegistrationBuffer(nn.Module):
    """
    Comparator module that detects 'mismatch' between memory and reality.
    
    Reference:
    - Whittington et al. (2020): "How to build a cognitive map"
      (Rationale: CA1 measures the error between predicted context and current input).
    """
    def __init__(self, cfg: AllocortexConfig):
        super().__init__()

    def compute_mismatch(self, reality: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # A high mismatch signal triggers 'Novelty' and forces a new CA3 write
        return F.mse_loss(reality, memory, reduction='none').mean(dim=-1)

class AllocortexSystem(nn.Module):
    """
    The complete hippocampal formation interface.
    """
    def __init__(self, cfg: AllocortexConfig):
        super().__init__()
        self.dg = DentateGyrus_SparseEncoder(cfg)
        self.ca3 = CA3_RecurrentMatrix(cfg)
        self.ca1 = CA1_RegistrationBuffer(cfg)

    def forward(self, isocortex_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Encode the current 'What/Where' gist
        sparse_prior = self.dg(isocortex_state)
        
        # 2. Reconstruct the most relevant past episode
        reconstructed_episode = self.ca3.attractor_settle(sparse_prior)
        
        # 3. Detect if this is a 'New' or 'Known' experience
        mismatch = self.ca1.compute_mismatch(isocortex_state, reconstructed_episode)
        
        return reconstructed_episode, mismatch
      
