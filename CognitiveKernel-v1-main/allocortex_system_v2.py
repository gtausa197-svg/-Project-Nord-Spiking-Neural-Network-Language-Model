
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

class AllocortexSystem(nn.Module):
    """
    The complete hippocampal formation interface for one-shot storage.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # DG: Sparse indexer for pattern separation
        self.dg_threshold = cfg.sparse_threshold
        
        # CA3: Auto-associative matrix for attractor settlement
        self.register_buffer("ca3_matrix", torch.zeros(cfg.num_episodes, cfg.entry_dim))
        self.write_ptr = 0

    def pattern_separation(self, x):
        """DG logic: Sparse encoding to prevent memory overlap."""
        return (x > self.dg_threshold).float() * x

    def pattern_completion(self, query, steps=5):
        """CA3 logic: Settles the attractor to retrieve a past episode."""
        current = query
        for _ in range(steps):
            # Associative retrieval (Larimar 2024 logic)
            sim = F.cosine_similarity(current, self.ca3_matrix, dim=-1)
            weights = F.softmax(sim, dim=-1)
            current = 0.8 * torch.matmul(weights, self.ca3_matrix) + 0.2 * current
        return current

    def forward(self, isocortex_state: torch.Tensor):
        # 1. Encode via DG
        sparse_prior = self.pattern_separation(isocortex_state)
        
        # 2. Retrieve via CA3
        reconstructed_episode = self.pattern_completion(sparse_prior)
        
        # 3. CA1 Mismatch (Novelty Detection)
        # Low similarity = high novelty = trigger one-shot write
        mismatch = F.mse_loss(isocortex_state, reconstructed_episode)
        return reconstructed_episode, mismatch

    def one_shot_write(self, episodic_trace: torch.Tensor):
        """Write the current state to the CA3 matrix."""
        slot = self.write_ptr % self.cfg.num_episodes
        self.ca3_matrix[slot] = episodic_trace.detach().mean(dim=0)
        self.write_ptr += 1
      
