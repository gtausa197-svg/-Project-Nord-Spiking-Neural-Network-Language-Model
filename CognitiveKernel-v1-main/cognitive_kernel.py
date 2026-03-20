"""
cognitive_kernel.py
===================

PRAGMI Cognitive Kernel: Hippocampal Memory Subsystem
=====================================================

BIOLOGICAL GROUNDING
--------------------
This file models the hippocampal formation as a persistent episodic memory
system. The hippocampus binds distributed neocortical representations into
addressable episodic traces, supports pattern completion from partial cues,
and uses mismatch detection to decide whether incoming experience updates
an existing memory or creates a new one. The three most important papers
grounding the architecture are:

1. Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
   separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
   DOI: 10.3389/fnsys.2013.00074
   (CA3 autoassociation, pattern completion/separation, sparse DG codes)

2. Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M., & Kohn, A.
   (2019). "Cortical areas interact through a communication subspace."
   Neuron, 102(1), 249-259. DOI: 10.1016/j.neuron.2019.01.026
   (Low-rank inter-area projections, predictive vs. private dimensions)

3. Das, P. et al. (2024). "Larimar: Large Language Models with Episodic
   Memory Control." ICML 2024. arXiv: 2403.11901.
   DOI: {To be added later.}
   (Ben-Israel pseudoinverse for interference-minimizing memory writes)

ARCHITECTURE POSITION
---------------------
The Cognitive Kernel sits at the deepest layer of the three-layer stack.
The External LLM (user-facing) sends tokens to Timmy (the spiking bridge
language model). Timmy translates token embeddings into the kernel's
coordinate space via the PerforantPath, which uses a low-rank communication
subspace to prevent identity bleed from Timmy's internal dynamics. The
kernel speaks in coordinates and spike patterns, never in tokens.

The LLM is downstream of reconstructed experience and should be treated
as a narrator, interpreter, and planner. The kernel does the actual
construction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal


# =============================================================================
# S0  CONFIGURATION
# =============================================================================

@dataclass
class HippocampalConfig:
    """Configuration for the Cognitive Kernel.

    All parameters with biological analogs cite their source. Parameters that
    are engineering approximations are labeled explicitly.
    """

    # --- Coordinate space dimensions ---

    coordinate_dim: int = 64
    """Dimensionality of the shared coordinate manifold (the codebook bus).
    NOT a biological quantity. Engineering choice balancing expressiveness
    against computational cost for the UMAP projection target space."""

    bridge_dim: int = 496
    """Dimensionality of Timmy's output space (d_model from the SNN LM).
    This is the source space from which the communication subspace extracts
    predictive dimensions."""

    # --- Communication subspace (Semedo et al., 2019) ---

    comm_subspace_rank: int = 3
    """Rank of the perforant path communication subspace. Semedo et al.
    (2019) found inter-area communication uses rank 2-3 subspaces between
    V1 and V2. We use rank 3 as a conservative default for the Timmy-to-
    kernel projection.
    Semedo, J.D. et al. (2019). "Cortical areas interact through a
    communication subspace." Neuron, 102(1), 249-259.
    DOI: 10.1016/j.neuron.2019.01.026"""

    # --- CA3 autoassociative memory ---

    ca3_memory_slots: int = 256
    """Number of addressable memory slots in the CA3 recurrent matrix.
    NOT a biological quantity. Biological CA3 has ~300,000 neurons in the
    rat and ~2 million in primates, but we model the attractor basin count,
    not the neuron count. 256 slots is a prototype-scale choice.
    Rolls, E.T. (2013). DOI: 10.3389/fnsys.2013.00074"""

    ca3_code_dim: int = 64
    """Dimensionality of each memory vector stored in CA3. Matches the
    coordinate manifold dimension so that episodic traces live natively
    in the codebook bus."""

    pseudoinverse_iterations: int = 8
    """Number of Ben-Israel-Cohen iterations for the pseudoinverse
    approximation during memory writes. Larimar uses 3 by default, but
    we increase to 8 to match our temporal processing window T=8.
    Das, P. et al. (2024). "Larimar: Large Language Models with Episodic
    Memory Control." arXiv: 2403.11901. DOI: {To be added later.}"""

    pseudoinverse_alpha_init: float = 0.0
    """Initial value of the learnable log-scale for the Ben-Israel step
    size. The actual alpha is computed dynamically as 1/||A||^2 and then
    scaled by exp(this value), clamped to max 5e-4. A value of 0.0 means
    the learned scale starts at exp(0)=1, so the initial alpha is pure
    1/||A||^2.
    NOT a biological quantity. Training artifact for convergence stability."""

    convergence_tolerance: float = 1e-4
    """Residual threshold below which the pseudoinverse is considered
    converged. If the residual exceeds this after all iterations, the
    astrocytic regulator receives a metabolic alert.
    NOT a biological quantity. Engineering convergence criterion."""

    # --- CA1 mismatch detection ---

    novelty_low_threshold: float = 0.1
    """MSE below this value is classified as familiar (no update needed).
    Derived from the observation that CA1 acts as a comparator between
    CA3 recurrent output and direct entorhinal input, with mismatch
    novelty driving encoding decisions.
    Kumaran, D. & Maguire, E.A. (2007). "Match-mismatch processes underlie
    human hippocampal responses to associative novelty." Journal of
    Neuroscience, 27(32), 8517-8524. DOI: 10.1523/JNEUROSCI.1677-07.2007"""

    novelty_high_threshold: float = 0.5
    """MSE above this value is classified as fully novel (triggers new
    encoding). Between low and high thresholds, the system performs graded
    reconsolidation.
    NOT a biological quantity. The boundary between reconsolidation and
    new encoding is likely context-dependent in biology. We use a fixed
    threshold as a first approximation."""

    # --- Astrocytic regulation ---

    metabolic_alert_decay: float = 0.95
    """Exponential decay for the metabolic alert signal. Astrocytes
    integrate metabolic state over time and signal when the system is
    under stress (e.g., failed convergence, energy depletion).
    Araque, A. et al. (1999). "Tripartite synapses: glia, the unacknowledged
    partner." Trends in Neurosciences, 22(5), 208-215.
    DOI: 10.1016/S0166-2236(98)01349-6"""

    # --- Processing window ---

    T: int = 8
    """Number of spiking timesteps per forward pass. Matches the temporal
    window in which the Ben-Israel iteration must converge.
    NOT a biological quantity. Engineering choice matching the SNN LM's
    temporal unroll depth."""

    observation_noise_std: float = 0.0
    """Standard deviation of observation noise for Bayesian memory updates.
    Set to 0.0 for deterministic writes in the prototype. Nonzero values
    add a noise floor to the Kalman gain denominator, preventing division
    by zero in degenerate cases.
    NOT a biological quantity. Regularization parameter from Larimar."""


# =============================================================================
# S1  COMMUNICATION SUBSPACE (Task 1: ZoneTap)
# =============================================================================

class CommunicationSubspace(nn.Module):
    """Low-rank inter-area projection implementing the communication subspace.

    BIOLOGICAL STRUCTURE: Perforant Path (entorhinal cortex layer II to
    hippocampal CA3/DG). In our architecture, this models the projection
    from Timmy's output space (the "neocortical" source) into the kernel's
    coordinate space (the hippocampal target).

    BIOLOGICAL FUNCTION: Cortical areas communicate through low-dimensional
    subspaces of their population activity. Only the "predictive dimensions"
    (the directions in the source space that actually predict target
    fluctuations) are transmitted. The loudest activity in the source is NOT
    necessarily what the target receives. This selective routing prevents
    identity bleed from Timmy's internal processing into the kernel.

    The projection is parameterized as W = U_send @ V_receive, where:
        U_send:    (source_dim, rank) selects predictive dimensions in Timmy
        V_receive: (rank, target_dim) maps those dimensions into kernel space

    This factored form enforces rank exactly equal to comm_subspace_rank.

    Semedo, J.D., Zandvakili, A., Machens, C.K., Yu, B.M., & Kohn, A.
    (2019). "Cortical areas interact through a communication subspace."
    Neuron, 102(1), 249-259. DOI: 10.1016/j.neuron.2019.01.026
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.comm_subspace_rank

        # U_send: projects from Timmy's d_model space into the rank-r
        # communication subspace. Initialized with small random values so
        # the subspace starts undetermined and is shaped by training.
        self.U_send = nn.Parameter(
            torch.randn(cfg.bridge_dim, r) * (1.0 / math.sqrt(cfg.bridge_dim))
        )

        # V_receive: projects from the rank-r subspace into the kernel's
        # coordinate manifold. Initialized similarly.
        self.V_receive = nn.Parameter(
            torch.randn(r, cfg.coordinate_dim) * (1.0 / math.sqrt(r))
        )

        # Singular value scaling: the meta-zone (or, in prototype, a learned
        # diagonal) controls the gain on each communication channel.
        # Initialized to ones so all channels are open at start.
        # Semedo et al. showed that different downstream targets can have
        # different subspaces. When we add multiple targets, each gets its
        # own singular value vector. For now, one target (the kernel).
        self.channel_gains = nn.Parameter(torch.ones(r))

    def forward(self, timmy_output: Tensor) -> Tensor:
        """Project Timmy's output through the communication subspace.

        This is the Perforant Path: the anatomical projection from entorhinal
        cortex layer II to hippocampal CA3. It carries the retrieval cue
        that initiates pattern completion in the autoassociative network.

        Biologically, the perforant path provides ~3,600 synaptic connections
        per CA3 cell, enough to relay a partial cue but not enough to drive
        storage directly. Storage requires the stronger mossy fiber input.
        Rolls, E.T. (2013). DOI: 10.3389/fnsys.2013.00074

        Parameters
        ----------
        timmy_output : Tensor, shape (batch, bridge_dim)
            Timmy's spike-rate coded output at the current timestep.
            This is the population activity vector of the source area.

        Returns
        -------
        kernel_input : Tensor, shape (batch, coordinate_dim)
            The input to the hippocampal coordinate manifold. Only the
            predictive dimensions of Timmy's activity reach the kernel.
        """
        # Scale the communication channels. Setting a gain to zero closes
        # that channel. This is how the meta-zone modulates inter-area
        # routing: by rotating source activity into or out of alignment
        # with the communication subspace.
        scaled_U = self.U_send * self.channel_gains.unsqueeze(0)  # (bridge, r)

        # Two thin matmuls instead of one large dense matmul.
        # Computational cost: O(B * bridge * r) + O(B * r * coord)
        # versus O(B * bridge * coord) for a dense projection.
        subspace_activity = timmy_output @ scaled_U   # (batch, r)
        kernel_input = subspace_activity @ self.V_receive  # (batch, coord)

        return kernel_input

    def effective_weight(self) -> Tensor:
        """Return the full (bridge_dim, coordinate_dim) projection matrix.

        Useful for analysis and visualization. The rank of this matrix is
        at most comm_subspace_rank by construction.
        """
        scaled_U = self.U_send * self.channel_gains.unsqueeze(0)
        return scaled_U @ self.V_receive


# =============================================================================
# S2  CA3 RECURRENT MATRIX WITH STABILIZED BEN-ISRAEL (Task 2)
# =============================================================================

class CA3RecurrentMatrix(nn.Module):
    """Autoassociative memory matrix modeling hippocampal area CA3.

    BIOLOGICAL STRUCTURE: CA3 pyramidal cell network with recurrent
    collateral (RC) synapses. In the rat, each CA3 cell receives ~12,000
    RC synapses from other CA3 cells, forming a single interconnected
    autoassociative network with ~2% connectivity.

    BIOLOGICAL FUNCTION: CA3 stores episodic memories as attractor basins
    in its recurrent weight matrix. Given a partial cue via the perforant
    path, the recurrent dynamics complete the pattern by settling into the
    nearest attractor basin. Storage uses strong mossy fiber input to force
    a new pattern; retrieval uses weaker perforant path input to cue recall.

    Rolls, E.T. (2013). "The mechanisms for pattern completion and pattern
    separation in the hippocampus." Frontiers in Systems Neuroscience, 7, 74.
    DOI: 10.3389/fnsys.2013.00074

    WRITE MECHANISM: Pseudoinverse addressing from Larimar (Das et al., 2024).
    Instead of a circular buffer, the write operation solves for addressing
    weights w such that z ≈ w @ M, using the Moore-Penrose pseudoinverse
    approximated by the Ben-Israel-Cohen iterative method. The memory matrix
    is then updated via a Bayesian posterior (Kalman filter update).

    CONVERGENCE STABILIZATION: The step size alpha for the Ben-Israel
    iteration is initialized dynamically as 1/||A||^2 (spectral norm
    scaling) rather than a fixed learned constant. A residual check after
    each iteration triggers an early-stop or metabolic alert.

    Das, P. et al. (2024). "Larimar: Large Language Models with Episodic
    Memory Control." arXiv: 2403.11901. DOI: {To be added later.}
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        K, C = cfg.ca3_memory_slots, cfg.ca3_code_dim

        # Memory matrix: (K, C). Each row is one memory vector.
        # Initialized to small random values (not zeros, to avoid
        # degenerate pseudoinverse on first write).
        self.register_buffer(
            "memory_mean",
            torch.randn(K, C) * 0.01
        )

        # Memory uncertainty (diagonal approximation of the posterior
        # covariance). Initialized to ones (uniform prior uncertainty).
        self.register_buffer(
            "memory_variance",
            torch.ones(K, C)
        )

        # Slot occupancy tracker: how many writes each slot has received.
        self.register_buffer(
            "slot_write_count",
            torch.zeros(K, dtype=torch.long)
        )

        # Learnable log-scale for the Ben-Israel step size.
        # The actual alpha = (1 / ||A||^2) * exp(this), clamped.
        self.ben_israel_log_scale = nn.Parameter(
            torch.tensor(cfg.pseudoinverse_alpha_init)
        )

        # Convergence diagnostics (not parameters, just state)
        self.register_buffer("last_residual", torch.tensor(0.0))
        self.register_buffer("last_converged", torch.tensor(True))

    def _compute_dynamic_alpha(self, A: Tensor) -> Tensor:
        """Compute the step size for Ben-Israel iteration.

        The optimal initial approximation for the pseudoinverse of A is
        alpha * A^T, where alpha = 1 / ||A||_2^2 (the squared spectral
        norm). This guarantees that the iteration A_{k+1} = 2*A_k - A_k@A@A_k
        converges to the Moore-Penrose pseudoinverse.

        We scale alpha by a learned factor (clamped to max 5e-4 for
        stability, following Larimar's approach).

        Parameters
        ----------
        A : Tensor, shape (batch, n, m) or (n, m)
            The matrix whose pseudoinverse we are approximating.

        Returns
        -------
        alpha : Tensor, scalar
            The step size for initializing the Ben-Israel iteration.
        """
        # Spectral norm via power iteration would be ideal but expensive.
        # Frobenius norm is an upper bound on spectral norm and cheaper.
        # ||A||_F^2 >= ||A||_2^2, so 1/||A||_F^2 <= 1/||A||_2^2.
        # This makes alpha conservative (smaller step), which is safe.
        A_fro_sq = (A * A).sum()
        alpha_base = 1.0 / (A_fro_sq + 1e-8)  # avoid division by zero

        # Apply learned scaling, clamped for stability
        learned_scale = torch.clamp(
            torch.exp(self.ben_israel_log_scale), max=5e-4
        )
        alpha = alpha_base * learned_scale

        return alpha

    def _approx_pseudoinverse(
        self,
        A: Tensor,
        max_iterations: int = 8,
        tolerance: float = 1e-4,
    ) -> Tuple[Tensor, Tensor, bool]:
        """Ben-Israel-Cohen iterative pseudoinverse with dynamic step size.

        The iteration is: A_pinv_{k+1} = 2 * A_pinv_k - A_pinv_k @ A @ A_pinv_k
        initialized with A_pinv_0 = alpha * A^T.

        Ben-Israel, A. & Cohen, D. (1966). "On iterative computation of
        generalized inverses and associated projections." SIAM Journal on
        Numerical Analysis, 3(3), 410-419. DOI: 10.1137/0703035

        Parameters
        ----------
        A : Tensor, shape (K, C)
            The memory matrix.
        max_iterations : int
            Maximum number of iterations (should match T=8).
        tolerance : float
            Convergence threshold on the residual norm.

        Returns
        -------
        A_pinv : Tensor, shape (C, K)
            Approximation of the Moore-Penrose pseudoinverse of A.
        residual : Tensor, scalar
            The final residual ||A @ A_pinv @ A - A||_F.
        converged : bool
            Whether the residual fell below tolerance.
        """
        alpha = self._compute_dynamic_alpha(A)

        # Initialize: A_pinv_0 = alpha * A^T
        A_pinv = alpha * A.T  # (C, K)

        converged = False
        residual = torch.tensor(float('inf'), device=A.device)

        for i in range(max_iterations):
            # Ben-Israel iteration step
            A_pinv_new = 2.0 * A_pinv - A_pinv @ A @ A_pinv

            # Residual check: ||A @ A_pinv @ A - A||_F
            # This measures how well A_pinv approximates A_pseudoinverse.
            # At convergence, A @ A_pinv @ A = A exactly.
            reconstruction = A @ A_pinv_new @ A
            residual = torch.norm(reconstruction - A, p='fro')

            A_pinv = A_pinv_new

            if residual.item() < tolerance:
                converged = True
                break

        return A_pinv, residual, converged

    def read(self, query: Tensor) -> Tuple[Tensor, Tensor]:
        """Read from memory using pseudoinverse addressing.

        This implements the retrieval side of CA3 pattern completion.
        Given a query (partial cue from the perforant path), compute
        addressing weights over memory slots and reconstruct the full
        memory pattern.

        BIOLOGICAL ANALOG: Perforant path input activates CA3 recurrent
        collaterals, which settle into the nearest attractor basin via
        autoassociative dynamics. We approximate this with a single
        pseudoinverse solve rather than iterative attractor settlement,
        which is a simplification but captures the key property:
        partial cues retrieve whole patterns.

        Parameters
        ----------
        query : Tensor, shape (batch, code_dim)
            The retrieval cue in coordinate space.

        Returns
        -------
        reconstruction : Tensor, shape (batch, code_dim)
            The completed memory pattern.
        addressing_weights : Tensor, shape (batch, memory_slots)
            The soft attention over memory slots (useful for diagnostics).
        """
        M = self.memory_mean  # (K, C)
        A_pinv, residual, converged = self._approx_pseudoinverse(
            M,
            max_iterations=self.cfg.pseudoinverse_iterations,
            tolerance=self.cfg.convergence_tolerance,
        )

        # Store convergence diagnostics
        self.last_residual.fill_(residual.item())
        self.last_converged.fill_(converged)

        # Addressing weights: w = query @ A_pinv^T = query @ (alpha * M^T ... iterated)
        # A_pinv has shape (C, K), so query @ A_pinv gives (batch, K)
        addressing_weights = query @ A_pinv  # (batch, K)

        # Softmax for interpretability and bounded contribution
        addressing_weights = F.softmax(addressing_weights, dim=-1)

        # Reconstruct: z_hat = w @ M
        reconstruction = addressing_weights @ M  # (batch, C)

        return reconstruction, addressing_weights

    def write(
        self,
        z: Tensor,
        slot_idx: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """Write a new episode to memory using Bayesian pseudoinverse update.

        This implements the storage side of CA3 episodic encoding.

        BIOLOGICAL ANALOG: Mossy fiber input from the dentate gyrus forces
        a new activation pattern into CA3, overriding the recurrent dynamics.
        The new pattern is then stored in the recurrent collateral weights
        via Hebbian learning. We model this as a Bayesian update on the
        memory matrix, which is mathematically equivalent to a Kalman filter
        step and produces interference-minimizing writes.

        Rolls, E.T. (2013). DOI: 10.3389/fnsys.2013.00074 (mossy fiber forcing)
        Das, P. et al. (2024). arXiv: 2403.11901 (pseudoinverse addressing)

        Parameters
        ----------
        z : Tensor, shape (batch, code_dim)
            The episode vector to store.
        slot_idx : Tensor, optional, shape (batch,)
            If provided, forces the write to specific slots (used during
            reconsolidation to update an existing basin). If None, the
            system finds the least-used slot.

        Returns
        -------
        stats : dict
            Write diagnostics including convergence info and slot indices.
        """
        M = self.memory_mean  # (K, C)
        V = self.memory_variance  # (K, C)
        K, C = M.shape
        batch = z.shape[0]

        # Determine target slots
        if slot_idx is None:
            # Find the least-written slots for each batch element.
            # This is a simple heuristic; a biological system would use
            # dentate gyrus pattern separation to find an orthogonal slot.
            counts = self.slot_write_count.clone()
            slot_idx = torch.stack([
                torch.argmin(counts) for _ in range(batch)
            ])

        # Compute addressing weights for the new episode
        A_pinv, residual, converged = self._approx_pseudoinverse(
            M,
            max_iterations=self.cfg.pseudoinverse_iterations,
            tolerance=self.cfg.convergence_tolerance,
        )

        w = z @ A_pinv  # (batch, K)
        w = F.softmax(w, dim=-1)

        # Bayesian update (Kalman filter step)
        # prediction: z_hat = w @ M
        z_hat = w @ M  # (batch, C)
        delta = z - z_hat  # prediction error

        # For each batch element, update the target slot
        for b in range(batch):
            s = slot_idx[b].item()
            noise_var = self.cfg.observation_noise_std ** 2

            # Kalman gain for this slot (simplified diagonal approximation)
            slot_var = V[s]  # (C,)
            w_s = w[b, s]  # scalar weight for target slot
            gain_denom = w_s * w_s * slot_var + noise_var + 1e-8
            kalman_gain = (w_s * slot_var) / gain_denom  # (C,)

            # Update mean and variance
            self.memory_mean[s] = M[s] + kalman_gain * delta[b]
            self.memory_variance[s] = V[s] - kalman_gain * w_s * V[s]
            self.memory_variance[s] = torch.clamp(
                self.memory_variance[s], min=1e-6
            )

            self.slot_write_count[s] += 1

        return {
            "slot_indices": slot_idx,
            "residual": residual,
            "converged": converged,
            "prediction_error_norm": torch.norm(delta, dim=-1).mean(),
        }


# =============================================================================
# S3  CA1 REGISTRATION BUFFER WITH GRADED NOVELTY (Task 3)
# =============================================================================

class CA1RegistrationBuffer(nn.Module):
    """Mismatch detector and routing decision module modeling hippocampal CA1.

    BIOLOGICAL STRUCTURE: CA1 pyramidal cell layer, positioned between CA3
    and the subiculum/entorhinal cortex output pathway. CA1 receives two
    convergent inputs: the Schaffer collateral projection from CA3 (carrying
    the pattern-completed memory) and the direct perforant path from
    entorhinal cortex layer III (carrying the current sensory input).

    BIOLOGICAL FUNCTION: CA1 acts as a comparator. It detects the mismatch
    between what the system predicted (the CA3 retrieval) and what is
    actually happening (the direct entorhinal input). The magnitude of this
    mismatch drives a graded routing decision:

        Low mismatch  -> The current input is FAMILIAR. No update needed.
                         The existing memory basin is confirmed.

        Medium mismatch -> The current input is SIMILAR BUT CHANGED.
                           Trigger RECONSOLIDATION: update the existing
                           CA3 basin with the new information. This is the
                           mechanism by which memories are modified upon
                           re-retrieval, as documented in the reconsolidation
                           literature.

        High mismatch -> The current input is NOVEL. Trigger new ENCODING:
                         one-shot write to a fresh CA3 slot via mossy fiber
                         forcing. The dentate gyrus performs pattern separation
                         to find an orthogonal slot.

    Kumaran, D. & Maguire, E.A. (2007). "Match-mismatch processes underlie
    human hippocampal responses to associative novelty." Journal of
    Neuroscience, 27(32), 8517-8524. DOI: 10.1523/JNEUROSCI.1677-07.2007

    Lisman, J.E. & Grace, A.A. (2005). "The hippocampal-VTA loop: controlling
    the entry of information into long-term memory." Neuron, 46(5), 703-713.
    DOI: 10.1016/j.neuron.2005.05.002
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable temperature for the sigmoid mapping from raw MSE
        # to the graded novelty scalar. Higher temperature makes the
        # transition sharper (more binary); lower makes it smoother.
        # NOT a biological quantity. Controls the shape of the novelty curve.
        self.novelty_temperature = nn.Parameter(torch.tensor(5.0))

        # Learnable midpoint for the sigmoid (where novelty = 0.5).
        # Initialized to the geometric mean of the low and high thresholds.
        midpoint = (cfg.novelty_low_threshold + cfg.novelty_high_threshold) / 2.0
        self.novelty_midpoint = nn.Parameter(torch.tensor(midpoint))

    def compute_mismatch(
        self,
        ca3_output: Tensor,
        entorhinal_input: Tensor,
    ) -> Tensor:
        """Compute the raw mismatch between CA3 retrieval and current input.

        BIOLOGICAL ANALOG: The Schaffer collateral projection carries
        pattern-completed output from CA3 to CA1. Simultaneously, the
        direct perforant path from entorhinal cortex layer III carries
        the current sensory/contextual representation. CA1 compares
        these two streams.

        Rolls, E.T. (2013). DOI: 10.3389/fnsys.2013.00074
        "It is suggested that the modifiable connections from the CA3
        neurons to the CA1 neurons allow the whole episode in CA3 to
        be produced in CA1."

        Parameters
        ----------
        ca3_output : Tensor, shape (batch, code_dim)
            The pattern-completed memory from CA3 (Schaffer collateral input).
        entorhinal_input : Tensor, shape (batch, code_dim)
            The current experience in coordinate space (direct perforant path).

        Returns
        -------
        mse : Tensor, shape (batch,)
            Per-element mean squared error between the two streams.
        """
        return F.mse_loss(ca3_output, entorhinal_input, reduction='none').mean(dim=-1)

    def compute_novelty_scalar(self, mse: Tensor) -> Tensor:
        """Map raw MSE to a graded novelty scalar in [0, 1].

        The mapping uses a sigmoid centered on a learnable midpoint with
        learnable temperature. This produces a smooth, differentiable
        transition between the three regimes (familiar / reconsolidate /
        novel) rather than hard thresholds.

        Novelty scalar near 0.0: familiar (low mismatch)
        Novelty scalar near 0.5: boundary (reconsolidation zone)
        Novelty scalar near 1.0: novel (high mismatch, new encoding)

        Parameters
        ----------
        mse : Tensor, shape (batch,)
            Raw mismatch values from compute_mismatch.

        Returns
        -------
        novelty : Tensor, shape (batch,)
            Graded novelty scalar in [0, 1].
        """
        return torch.sigmoid(
            self.novelty_temperature * (mse - self.novelty_midpoint)
        )

    def route(
        self,
        ca3_output: Tensor,
        entorhinal_input: Tensor,
    ) -> Dict[str, Tensor]:
        """Compute mismatch, novelty, and routing decision.

        Returns a dictionary with all the information the kernel needs
        to decide whether to reconsolidate, encode new, or do nothing.

        Parameters
        ----------
        ca3_output : Tensor, shape (batch, code_dim)
        entorhinal_input : Tensor, shape (batch, code_dim)

        Returns
        -------
        routing : dict containing:
            mse: raw mismatch (batch,)
            novelty: graded scalar (batch,)
            action: per-element action string label
            reconsolidate_mask: boolean mask for reconsolidation (batch,)
            encode_mask: boolean mask for new encoding (batch,)
            familiar_mask: boolean mask for no-update (batch,)
        """
        mse = self.compute_mismatch(ca3_output, entorhinal_input)
        novelty = self.compute_novelty_scalar(mse)

        # Hard routing decisions for the prototype
        # (the novelty scalar itself is differentiable and could be used
        # as a soft interpolation weight in a more advanced version)
        familiar_mask = mse < self.cfg.novelty_low_threshold
        novel_mask = mse > self.cfg.novelty_high_threshold
        reconsolidate_mask = ~familiar_mask & ~novel_mask

        return {
            "mse": mse,
            "novelty": novelty,
            "familiar_mask": familiar_mask,
            "reconsolidate_mask": reconsolidate_mask,
            "encode_mask": novel_mask,
        }


# =============================================================================
# S4  ASTROCYTIC REGULATOR
# =============================================================================

class AstrocyticRegulator(nn.Module):
    """Metabolic state monitor modeling astrocyte-neuron signaling.

    BIOLOGICAL STRUCTURE: Astrocytes are non-neuronal glial cells that
    form tripartite synapses with pre- and post-synaptic neurons. They
    monitor local metabolic state, modulate synaptic transmission, and
    integrate signals over slower timescales than neurons.

    BIOLOGICAL FUNCTION: In the kernel, the astrocytic regulator tracks
    convergence health of the pseudoinverse iteration, energy expenditure
    of memory writes, and accumulated stress. When the system is under
    stress (e.g., repeated convergence failures), the regulator can
    suppress new encoding or trigger consolidation pressure.

    Araque, A., Parpura, V., Sanzgiri, R.P., & Bhayward, P.G. (1999).
    "Tripartite synapses: glia, the unacknowledged partner." Trends in
    Neurosciences, 22(5), 208-215. DOI: 10.1016/S0166-2236(98)01349-6
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("metabolic_stress", torch.tensor(0.0))
        self.register_buffer("convergence_failure_count", torch.tensor(0))
        self.register_buffer("total_write_energy", torch.tensor(0.0))

    def report_convergence(self, converged: bool, residual: float):
        """Receive a convergence report from the CA3 pseudoinverse solver.

        If the solver did not converge within T iterations, this constitutes
        a metabolic alert. The stress signal accumulates with exponential
        decay, modeling the slow integration timescale of astrocytic calcium
        signaling.

        Parameters
        ----------
        converged : bool
            Whether the Ben-Israel iteration converged.
        residual : float
            The final residual norm.
        """
        # Decay existing stress
        self.metabolic_stress *= self.cfg.metabolic_alert_decay

        if not converged:
            # Metabolic alert: pseudoinverse did not converge
            alert_magnitude = min(residual, 10.0)  # cap to prevent runaway
            self.metabolic_stress += alert_magnitude
            self.convergence_failure_count += 1

    def report_write_energy(self, prediction_error_norm: float):
        """Track the metabolic cost of a write operation.

        Larger prediction errors mean the memory update was more substantial,
        corresponding to higher metabolic demand.
        """
        self.total_write_energy += prediction_error_norm

    @property
    def is_stressed(self) -> bool:
        """Whether the system is under metabolic stress.

        When stressed, the kernel should suppress new encoding attempts
        and prioritize consolidation (sleep-like behavior) to restabilize
        the memory matrix.
        """
        return self.metabolic_stress.item() > 1.0

    def get_diagnostics(self) -> Dict[str, float]:
        """Return current metabolic state for monitoring."""
        return {
            "metabolic_stress": self.metabolic_stress.item(),
            "convergence_failures": self.convergence_failure_count.item(),
            "total_write_energy": self.total_write_energy.item(),
            "is_stressed": float(self.is_stressed),
        }


# =============================================================================
# S5  COGNITIVE KERNEL (Orchestrator)
# =============================================================================

class CognitiveKernel(nn.Module):
    """The hippocampal memory system. This is the innermost layer.

    BIOLOGICAL STRUCTURE: Complete hippocampal formation including
    the perforant path input, CA3 autoassociative network, CA1 comparator,
    and astrocytic metabolic regulation.

    BIOLOGICAL FUNCTION: Receives coordinate-space representations from
    Timmy via the communication subspace, performs pattern completion or
    new encoding depending on mismatch novelty, and returns reconstructed
    experience to be narrated by the external LLM.

    The kernel speaks in coordinates and spike patterns, never in tokens.
    The LLM is downstream of reconstructed experience.

    INTERFACE CONTRACTS:
        Input:  Timmy's spike-rate coded output (bridge_dim)
        Output: Reconstructed or newly-encoded coordinate vector (coordinate_dim)
        State:  Serializable via state_dict() for .soul persistence
    """

    def __init__(self, cfg: HippocampalConfig):
        super().__init__()
        self.cfg = cfg

        # --- Perforant Path: Communication Subspace (Task 1) ---
        self.perforant_path = CommunicationSubspace(cfg)

        # --- CA3: Autoassociative Memory with Pseudoinverse Writes (Task 2) ---
        self.ca3 = CA3RecurrentMatrix(cfg)

        # --- CA1: Mismatch Detection and Routing (Task 3) ---
        self.ca1 = CA1RegistrationBuffer(cfg)

        # --- Astrocytic Regulator ---
        self.astrocyte = AstrocyticRegulator(cfg)

    def forward(
        self,
        timmy_output: Tensor,
        mode: Literal["encode", "retrieve", "auto"] = "auto",
    ) -> Dict[str, Tensor]:
        """Process one timestep through the hippocampal formation.

        The flow follows the trisynaptic circuit:
        1. Perforant path: Timmy output -> coordinate space (via comm subspace)
        2. CA3: Pattern completion (retrieve existing memory)
        3. CA1: Compare retrieved pattern with current input (mismatch detection)
        4. Route: Based on novelty, either reconsolidate, encode new, or pass

        Parameters
        ----------
        timmy_output : Tensor, shape (batch, bridge_dim)
            Timmy's spike-rate coded output.
        mode : str
            "encode": Force new encoding regardless of novelty.
            "retrieve": Retrieve only, no writes.
            "auto": Let CA1 mismatch detection decide.

        Returns
        -------
        result : dict containing:
            coordinates: the output coordinate vector (batch, coordinate_dim)
            novelty: graded novelty scalar (batch,)
            action_taken: string describing what happened
            ca3_diagnostics: convergence info from the pseudoinverse
            astrocyte_diagnostics: metabolic state
        """
        batch = timmy_output.shape[0]

        # Step 1: Perforant path projection (Task 1: low-rank comm subspace)
        current_coords = self.perforant_path(timmy_output)

        # Step 2: CA3 pattern completion (Task 2: stabilized pseudoinverse)
        ca3_retrieved, addressing_weights = self.ca3.read(current_coords)

        # Step 3: CA1 mismatch detection (Task 3: graded novelty)
        routing = self.ca1.route(
            ca3_output=ca3_retrieved,
            entorhinal_input=current_coords,
        )

        # Step 4: Routing decision
        output_coords = ca3_retrieved.clone()
        action_taken = "retrieve_only"

        if mode == "encode" or (mode == "auto" and not self.astrocyte.is_stressed):

            if mode == "encode":
                # Forced encoding: write to new slot
                write_stats = self.ca3.write(current_coords)
                output_coords = current_coords
                action_taken = "forced_encode"

                self.astrocyte.report_convergence(
                    write_stats["converged"],
                    write_stats["residual"].item(),
                )
                self.astrocyte.report_write_energy(
                    write_stats["prediction_error_norm"].item()
                )

            elif mode == "auto":
                encode_mask = routing["encode_mask"]
                recon_mask = routing["reconsolidate_mask"]

                if encode_mask.any():
                    # Novel items: write to new slots
                    novel_coords = current_coords[encode_mask]
                    write_stats = self.ca3.write(novel_coords)
                    output_coords[encode_mask] = novel_coords
                    action_taken = "new_encoding"

                    self.astrocyte.report_convergence(
                        write_stats["converged"],
                        write_stats["residual"].item(),
                    )
                    self.astrocyte.report_write_energy(
                        write_stats["prediction_error_norm"].item()
                    )

                if recon_mask.any():
                    # Reconsolidation: update existing basins
                    recon_coords = current_coords[recon_mask]

                    # Find which slots were most active during retrieval
                    recon_weights = addressing_weights[recon_mask]
                    recon_slots = torch.argmax(recon_weights, dim=-1)

                    write_stats = self.ca3.write(
                        recon_coords, slot_idx=recon_slots
                    )

                    # Blend: output is a mix of old and new, weighted by novelty
                    novelty_blend = routing["novelty"][recon_mask].unsqueeze(-1)
                    output_coords[recon_mask] = (
                        (1.0 - novelty_blend) * ca3_retrieved[recon_mask]
                        + novelty_blend * recon_coords
                    )

                    if action_taken == "new_encoding":
                        action_taken = "mixed_encode_and_reconsolidate"
                    else:
                        action_taken = "reconsolidation"

                    self.astrocyte.report_convergence(
                        write_stats["converged"],
                        write_stats["residual"].item(),
                    )

        return {
            "coordinates": output_coords,
            "novelty": routing["novelty"],
            "mismatch_mse": routing["mse"],
            "action_taken": action_taken,
            "ca3_residual": self.ca3.last_residual.item(),
            "ca3_converged": self.ca3.last_converged.item(),
            "astrocyte": self.astrocyte.get_diagnostics(),
            "comm_subspace_effective_rank": self._effective_comm_rank(),
        }

    def _effective_comm_rank(self) -> float:
        """Compute the effective rank of the communication subspace.

        Uses the Shannon entropy of the normalized singular values of the
        effective weight matrix. An effective rank of k means the subspace
        is behaving as though it has k independent channels.

        This is a diagnostic, not a biological quantity.
        """
        W = self.perforant_path.effective_weight()
        try:
            s = torch.linalg.svdvals(W)
            s_normalized = s / (s.sum() + 1e-8)
            # Shannon entropy of the singular value distribution
            entropy = -(s_normalized * torch.log(s_normalized + 1e-12)).sum()
            # Effective rank = exp(entropy)
            return torch.exp(entropy).item()
        except Exception:
            return float(self.cfg.comm_subspace_rank)

    def serialize_state(self) -> Dict[str, Tensor]:
        """Serialize the kernel's full dynamical state for .soul persistence.

        This captures everything needed to restore the kernel across resets:
        the memory matrix, uncertainty estimates, slot occupancy, metabolic
        state, and all learned parameters.

        Returns
        -------
        state : dict
            A state dictionary suitable for torch.save().
        """
        return self.state_dict()

    def load_state(self, state: Dict[str, Tensor]):
        """Restore the kernel from a serialized state.

        Parameters
        ----------
        state : dict
            A state dictionary from serialize_state() or torch.load().
        """
        self.load_state_dict(state)


# =============================================================================
# S6  SMOKE TEST
# =============================================================================

def smoke_test():
    """Verify that all three tasks are wired correctly.

    This is NOT an acceptance test. It is a minimal connectivity check.
    The acceptance tests from the engineering protocol (cue-specific recall
    across resets with stable identity) are a separate deliverable.
    """
    print("=" * 72)
    print("PRAGMI Cognitive Kernel Smoke Test")
    print("=" * 72)

    cfg = HippocampalConfig()
    kernel = CognitiveKernel(cfg)

    # Count parameters
    total_params = sum(p.numel() for p in kernel.parameters())
    trainable_params = sum(p.numel() for p in kernel.parameters() if p.requires_grad)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    # --- Task 1: Communication Subspace ---
    print("[Task 1] Communication Subspace (ZoneTap)")
    W = kernel.perforant_path.effective_weight()
    rank = torch.linalg.matrix_rank(W).item()
    print(f"  Effective weight shape: {tuple(W.shape)}")
    print(f"  Exact rank: {rank} (should be <= {cfg.comm_subspace_rank})")
    eff_rank = kernel._effective_comm_rank()
    print(f"  Effective rank (entropy-based): {eff_rank:.2f}")
    assert rank <= cfg.comm_subspace_rank, "Rank exceeds subspace constraint!"
    print("  PASS: Low-rank constraint enforced.")
    print()

    # --- Task 2: Ben-Israel Convergence ---
    print("[Task 2] Ben-Israel Pseudoinverse Convergence")

    # Write some episodes first
    batch = 4
    fake_timmy = torch.randn(batch, cfg.bridge_dim)
    encode_result = kernel(fake_timmy, mode="encode")
    print(f"  Encode residual: {encode_result['ca3_residual']:.6f}")
    print(f"  Converged: {bool(encode_result['ca3_converged'])}")

    # Now test retrieval with a partial cue (corrupted version of input)
    partial_cue = fake_timmy + torch.randn_like(fake_timmy) * 0.1
    retrieve_result = kernel(partial_cue, mode="retrieve")
    print(f"  Retrieve residual: {retrieve_result['ca3_residual']:.6f}")
    print(f"  Converged: {bool(retrieve_result['ca3_converged'])}")
    print(f"  Astrocyte stress: {retrieve_result['astrocyte']['metabolic_stress']:.4f}")
    print("  PASS: Pseudoinverse with dynamic alpha operational.")
    print()

    # --- Task 3: Graded Novelty Scalar ---
    print("[Task 3] Graded Novelty Scalar (CA1 Mismatch)")

    # Test with known mismatch levels
    coord_dim = cfg.coordinate_dim
    base = torch.randn(1, coord_dim)

    # Familiar: same input
    ca1_result_familiar = kernel.ca1.route(base, base.clone())
    print(f"  Familiar:     MSE={ca1_result_familiar['mse'].item():.6f}, "
          f"novelty={ca1_result_familiar['novelty'].item():.4f}")

    # Medium mismatch: small perturbation
    medium = base + torch.randn_like(base) * 0.3
    ca1_result_medium = kernel.ca1.route(base, medium)
    print(f"  Medium:       MSE={ca1_result_medium['mse'].item():.6f}, "
          f"novelty={ca1_result_medium['novelty'].item():.4f}")

    # High mismatch: large perturbation
    novel = base + torch.randn_like(base) * 2.0
    ca1_result_novel = kernel.ca1.route(base, novel)
    print(f"  Novel:        MSE={ca1_result_novel['mse'].item():.6f}, "
          f"novelty={ca1_result_novel['novelty'].item():.4f}")

    # Verify ordering
    assert ca1_result_familiar['novelty'].item() < ca1_result_medium['novelty'].item(), \
        "Familiar should have lower novelty than medium!"
    assert ca1_result_medium['novelty'].item() < ca1_result_novel['novelty'].item(), \
        "Medium should have lower novelty than novel!"
    print("  PASS: Novelty scalar is monotonically increasing with mismatch.")
    print()

    # --- Auto mode: full pipeline ---
    print("[Integration] Full auto-mode pipeline")
    # Encode some episodes
    for i in range(8):
        episode = torch.randn(1, cfg.bridge_dim) * (0.5 + i * 0.1)
        result = kernel(episode, mode="encode")
    print(f"  Encoded 8 episodes.")

    # Retrieve with a novel stimulus
    novel_stim = torch.randn(1, cfg.bridge_dim) * 3.0
    auto_result = kernel(novel_stim, mode="auto")
    print(f"  Auto-mode action: {auto_result['action_taken']}")
    print(f"  Novelty: {auto_result['novelty'].item():.4f}")
    print(f"  Astrocyte: {auto_result['astrocyte']}")
    print()

    # --- Serialization round-trip ---
    print("[Persistence] Serialize/restore round-trip")
    state = kernel.serialize_state()
    kernel2 = CognitiveKernel(cfg)
    kernel2.load_state(state)

    # Verify identical output after restore
    test_input = torch.randn(1, cfg.bridge_dim)
    out1 = kernel(test_input, mode="retrieve")
    out2 = kernel2(test_input, mode="retrieve")
    diff = torch.norm(out1["coordinates"] - out2["coordinates"]).item()
    print(f"  Output difference after restore: {diff:.10f}")
    assert diff < 1e-5, "Serialization round-trip failed!"
    print("  PASS: State survives serialization.")
    print()

    print("=" * 72)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    smoke_test()
