"""
perforant_path.py
=================

PRAGMI Cognitive Kernel: Perforant Path Communication Subspace
==============================================================

BIOLOGICAL GROUNDING
--------------------
This file models the Perforant Path, the principal afferent projection from
entorhinal cortex layers II and III to the hippocampal formation. In the
biological brain, this pathway serves two distinct roles depending on target:
the projection to the Dentate Gyrus (via layer II) supports pattern separation
during encoding, and the direct projection to CA3 (also via layer II) relays
partial retrieval cues that initiate autoassociative pattern completion. The
projection from layer III targets CA1 directly and carries the current sensory
context for mismatch comparison.

In PRAGMI, this module sits at the boundary between Timmy (the spiking bridge
language model, the "subconscious") and the Cognitive Kernel (the hippocampal
memory system). It implements a low-rank communication subspace that filters
Timmy's high-dimensional population activity into a compact set of predictive
dimensions. Only the predictive dimensions reach the kernel. The loudest
activity inside Timmy is not necessarily what the kernel hears.

Lead papers:

1. Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M., & Kohn, A. (2019).
   "Cortical areas interact through a communication subspace."
   Neuron, 102(1), 249-259. DOI: 10.1016/j.neuron.2019.01.026

2. Witter, M.P., Naber, P.A., van Haeften, T., Machielsen, W.C.,
   Rombouts, S.A., Barkhof, F., Scheltens, P., & Lopes da Silva, F.H. (2000).
   "Cortico-hippocampal communication by way of parallel
   parahippocampal-subicular pathways."
   Hippocampus, 10(4), 398-410.
   DOI: 10.1002/1098-1063(2000)10:4<398::AID-HIPO6>3.0.CO;2-K

3. Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
   separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
   DOI: 10.3389/fnsys.2013.00074
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================
# This module uses HippocampalConfig from cognitive_kernel.py.
# When imported standalone for testing, a minimal fallback is provided.

try:
    from cognitive_kernel import HippocampalConfig
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class HippocampalConfig:
        """Minimal fallback config for standalone testing."""
        bridge_dim: int = 496
        coordinate_dim: int = 64
        comm_subspace_rank: int = 3


# =============================================================================
# PERFORANT PATH BRIDGE
# =============================================================================

class PerforantPathBridge(nn.Module):
    """Low-rank communication subspace modeling the Perforant Path.

    BIOLOGICAL STRUCTURE: Perforant Path, the axonal bundle projecting from
    entorhinal cortex layer II to the dentate gyrus and CA3 field of the
    hippocampus.

    BIOLOGICAL FUNCTION: In the rat hippocampus, each CA3 pyramidal cell
    receives approximately 3,600 perforant path synapses, enough to relay a
    partial retrieval cue but not strong enough to drive storage directly
    (storage requires the ~46 mossy fiber inputs, which are fewer but far
    more powerful per synapse). The perforant path input initiates recall by
    activating a subset of CA3 recurrent collaterals, which then complete
    the pattern via autoassociative dynamics.

    Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
    separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
    DOI: 10.3389/fnsys.2013.00074

    COMPUTATIONAL IMPLEMENTATION: The projection is factored as a product of
    two thin matrices with a diagonal gain between them:

        output = input @ (U_send * channel_gains) @ V_receive

    where U_send has shape (source_dim, rank), channel_gains has shape (rank,),
    and V_receive has shape (rank, target_dim). The full effective weight
    matrix has rank at most equal to comm_subspace_rank by construction.

    The per-channel gains implement selective routing. When the meta-zone (or,
    in the prototype, any external modulation signal) scales a gain toward
    zero, that communication channel closes: the corresponding dimension of
    source activity becomes invisible to the target. This is the mechanism
    by which inter-area routing can be modulated without rewiring.

    Semedo, J.D. et al. (2019). "Cortical areas interact through a
    communication subspace." Neuron, 102(1), 249-259.
    DOI: 10.1016/j.neuron.2019.01.026

    INTERFACE BOUNDARY:
        SENDING:    Timmy (spiking neural network language model, isocortex analog)
        RECEIVING:  Cognitive Kernel allocortex (DG/CA3 input layer)
        CONNECTION: Perforant Path (entorhinal cortex layer II projections)
    """

    def __init__(self, cfg: HippocampalConfig):
        """Initialize the low-rank perforant path projection.

        Parameters
        ----------
        cfg : HippocampalConfig
            Kernel configuration. The relevant fields are bridge_dim (Timmy's
            output dimensionality, the source space), coordinate_dim (the
            kernel's coordinate manifold, the target space), and
            comm_subspace_rank (the rank constraint on the projection).
        """
        super().__init__()
        self.cfg = cfg
        source_dim = cfg.bridge_dim
        target_dim = cfg.coordinate_dim
        rank = cfg.comm_subspace_rank

        # U_send: (source_dim, rank)
        # Selects the predictive dimensions in Timmy's population activity.
        # These are the directions in the source space that actually predict
        # target fluctuations. Critically, Semedo et al. (2019) showed that
        # these predictive dimensions are NOT aligned with the dominant
        # fluctuations (largest variance directions) in the source. The
        # loudest activity inside Timmy is not what the kernel hears.
        #
        # Initialization: Kaiming-scale for the source fan-in, producing
        # initial projections with unit-variance output when source
        # activations have unit variance.
        # NOT a biological quantity. Standard neural network initialization.
        self.U_send = nn.Parameter(
            torch.randn(source_dim, rank) * (1.0 / math.sqrt(source_dim))
        )

        # V_receive: (rank, target_dim)
        # Maps from the low-rank communication subspace into the kernel's
        # coordinate manifold. This is where the subspace activity becomes
        # a coordinate-space vector that the DG, CA3, and CA1 can process.
        #
        # Initialization: Kaiming-scale for the rank fan-in.
        # NOT a biological quantity. Standard neural network initialization.
        self.V_receive = nn.Parameter(
            torch.randn(rank, target_dim) * (1.0 / math.sqrt(rank))
        )

        # channel_gains: (rank,)
        # Per-channel singular value scaling. Each element controls the gain
        # on one communication channel. Initialized to 1.0 so all channels
        # are open at the start of training.
        #
        # BIOLOGICAL ANALOG: The meta-zone (default mode network analog)
        # modulates inter-area routing by scaling the singular values of each
        # projection. Setting a gain to zero closes that channel. In the
        # full six-zone cortical sheet, each zone-to-hippocampus projection
        # has its own gain vector, and the meta-zone adjusts all of them
        # to control what information reaches the kernel at any moment.
        #
        # Semedo, J.D. et al. (2019). DOI: 10.1016/j.neuron.2019.01.026
        # See project extract Section 3: "the meta-zone scales the singular
        # values of each inter-zone projection."
        self.channel_gains = nn.Parameter(torch.ones(rank))

    def forward(self, spike_rates_source: Tensor) -> Tensor:
        """Filter source population activity through the communication subspace.

        BIOLOGICAL ANALOG: Entorhinal cortex layer II neurons fire in
        response to neocortical input. Their axons traverse the perforant
        path and synapse onto dentate granule cells and CA3 pyramidal cells.
        Only the component of entorhinal activity that lies within the
        communication subspace effectively drives hippocampal targets.

        INTERFACE BOUNDARY:
            SENDING:    Timmy output layer (isocortex, source population)
            RECEIVING:  Kernel coordinate manifold (allocortex, target)
            CONNECTION: Perforant Path

        Parameters
        ----------
        spike_rates_source : Tensor, shape (batch, bridge_dim)
            Spike-rate coded population activity from Timmy. This is the
            full-dimensional output of the source area, containing both
            predictive dimensions (what the kernel will hear) and private
            dimensions (internal processing invisible to the kernel).

        Returns
        -------
        coords_target : Tensor, shape (batch, coordinate_dim)
            The input to the hippocampal coordinate manifold. Only the
            predictive dimensions of Timmy's activity reach the kernel.
        """
        # Scale U_send columns by per-channel gains. When a gain is zero,
        # the corresponding column of U_send is zeroed out, and that
        # communication channel is closed.
        scaled_U = self.U_send * self.channel_gains.unsqueeze(0)  # (source, rank)

        # Two thin matrix multiplications enforce the rank constraint.
        # Cost: O(B * source * rank) + O(B * rank * target)
        # versus O(B * source * target) for a dense projection.
        # For source=496, rank=3, target=64: 1,488 + 192 = 1,680 MACs
        # versus 31,744 MACs dense. Factor of ~19x cheaper.
        subspace_activity = spike_rates_source @ scaled_U  # (batch, rank)
        coords_target = subspace_activity @ self.V_receive  # (batch, target)

        return coords_target

    def effective_weight(self) -> Tensor:
        """Compute the full (bridge_dim, coordinate_dim) projection matrix.

        The matrix W = (U_send * channel_gains) @ V_receive has rank at most
        comm_subspace_rank by construction. This method materializes the full
        matrix for analysis, rank verification, and visualization of the
        communication subspace geometry.

        Returns
        -------
        W : Tensor, shape (bridge_dim, coordinate_dim)
            The effective dense weight matrix. Rank is at most
            cfg.comm_subspace_rank.
        """
        scaled_U = self.U_send * self.channel_gains.unsqueeze(0)
        return scaled_U @ self.V_receive

    def effective_rank(self) -> float:
        """Compute the Shannon-entropy effective rank of the projection.

        An effective rank of k means the communication subspace is behaving
        as though k independent channels are open. If all channel_gains are
        equal, effective rank equals the exact rank. If one channel dominates,
        effective rank approaches 1.

        NOT a biological quantity. Diagnostic metric for monitoring whether
        the subspace is collapsing to fewer dimensions than intended.

        Returns
        -------
        eff_rank : float
            Shannon-entropy effective rank, in [1, comm_subspace_rank].
        """
        W = self.effective_weight()
        try:
            singular_values = torch.linalg.svdvals(W)
            s_norm = singular_values / (singular_values.sum() + 1e-8)
            entropy = -(s_norm * torch.log(s_norm + 1e-12)).sum()
            return torch.exp(entropy).item()
        except Exception:
            return float(self.cfg.comm_subspace_rank)

    def get_diagnostics(self) -> Dict[str, float]:
        """Return diagnostic information about the communication subspace.

        Returns
        -------
        diagnostics : dict
            channel_gains: current per-channel gain values
            effective_rank: Shannon-entropy rank
            exact_rank: numerical rank of the effective weight matrix
            frobenius_norm: norm of the full projection (proxy for signal strength)
        """
        W = self.effective_weight()
        return {
            "channel_gains": self.channel_gains.detach().tolist(),
            "effective_rank": self.effective_rank(),
            "exact_rank": torch.linalg.matrix_rank(W).item(),
            "frobenius_norm": torch.norm(W, p="fro").item(),
        }


# =============================================================================
# SMOKE TEST
# =============================================================================

def _smoke_test():
    """Verify the PerforantPathBridge meets all constraints."""
    print("PerforantPathBridge Smoke Test")
    print("=" * 50)

    cfg = HippocampalConfig()
    bridge = PerforantPathBridge(cfg)

    params = sum(p.numel() for p in bridge.parameters())
    print(f"Parameters: {params:,}")
    print(f"  U_send:        {cfg.bridge_dim} x {cfg.comm_subspace_rank} = "
          f"{cfg.bridge_dim * cfg.comm_subspace_rank:,}")
    print(f"  V_receive:     {cfg.comm_subspace_rank} x {cfg.coordinate_dim} = "
          f"{cfg.comm_subspace_rank * cfg.coordinate_dim:,}")
    print(f"  channel_gains: {cfg.comm_subspace_rank}")

    # Rank constraint
    W = bridge.effective_weight()
    exact_rank = torch.linalg.matrix_rank(W).item()
    print(f"\nEffective weight: {tuple(W.shape)}")
    print(f"Exact rank: {exact_rank} (must be <= {cfg.comm_subspace_rank})")
    assert exact_rank <= cfg.comm_subspace_rank, "Rank constraint violated!"

    # Forward pass
    batch = 4
    fake_input = torch.randn(batch, cfg.bridge_dim)
    output = bridge(fake_input)
    print(f"Input:  {tuple(fake_input.shape)}")
    print(f"Output: {tuple(output.shape)}")
    assert output.shape == (batch, cfg.coordinate_dim), "Output shape wrong!"

    # Channel gating: close one channel, verify rank drops
    with torch.no_grad():
        bridge.channel_gains[0] = 0.0
    W_gated = bridge.effective_weight()
    rank_gated = torch.linalg.matrix_rank(W_gated).item()
    print(f"\nAfter closing channel 0:")
    print(f"Rank: {rank_gated} (should be <= {cfg.comm_subspace_rank - 1})")
    assert rank_gated <= cfg.comm_subspace_rank - 1, "Gating did not reduce rank!"

    # Diagnostics
    diag = bridge.get_diagnostics()
    print(f"\nDiagnostics: {diag}")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _smoke_test()
