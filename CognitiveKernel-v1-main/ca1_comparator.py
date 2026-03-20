"""
ca1_comparator.py
The CA1 Mismatch Detection and Gating System: Spiking Substrate + Readout

BIOLOGICAL GROUNDING:
This file models the CA1 subfield of the hippocampal formation, which operates
as a biological comparator distinguishing between internal memory reconstructions
and external sensory reality. CA1 pyramidal cells receive convergent input from
two anatomically distinct pathways:

    1. Schaffer collaterals (from CA3): carry the memory reconstruction signal.
       These synapse onto proximal apical dendrites of CA1 pyramidal cells.
    2. Temporoammonic path (from EC layer III): carry the current sensory reality.
       These synapse onto distal apical dendrites of CA1 pyramidal cells.

The mismatch between these two streams produces a prediction error signal that
gates downstream memory operations: encoding of novel episodes, reconsolidation
of partially-matching memories, or silence when experience matches prediction.

This file implements two computational layers that serve different consumers:

    Layer 1 (Spiking Substrate): Izhikevich neuron dynamics for the CA1 pyramidal
    population. Produces spike outputs consumed by the STDP eligibility trace
    system during wake and by the three-factor consolidation rule during NREM.

    Layer 2 (Mismatch Readout): Continuous dual-component mismatch signal
    (perceptual + semantic) consumed by the Allocortex memory controller to
    make graded encoding/reconsolidation decisions.

These are not competing implementations. They are two stages of a pipeline:
biophysical computation first (how the neurons actually process the inputs),
then readout (what the population-level result means for the memory system).

Key Grounding Papers:
1. Vinogradova, O.S. (2001). "Hippocampus as comparator: role of the two
   input and two output systems of the hippocampus." Hippocampus, 11(5),
   578-598. DOI: 10.1002/hipo.1073
   Establishes CA1 as a comparator between hippocampal and cortical streams.

2. Hasselmo, M.E. & Schnell, E. (1994). "Laminar selectivity of the
   cholinergic suppression of synaptic transmission in rat hippocampal
   region CA1." Journal of Neuroscience, 14(6), 3898-3914.
   DOI: 10.1523/JNEUROSCI.14-06-03898.1994
   Theta-phase modulation of the relative strength of the two CA1 inputs.

3. Izhikevich, E.M. (2003). "Simple model of spiking neurons." IEEE
   Transactions on Neural Networks, 14(6), 1569-1572.
   DOI: 10.1109/TNN.2003.820440
   The neuron model used for CA1 pyramidal cell dynamics.

4. Helfer, P. & Shultz, T.R. (2019). "A computational model of systems
   memory consolidation and reconsolidation." Hippocampus, 30(7), 659-677.
   DOI: 10.1002/hipo.23187
   Reconsolidation as graded prediction error: moderate mismatch updates,
   large mismatch creates new trace.

5. Amaral, D.G. & Witter, M.P. (1989). "The three-dimensional organization
   of the hippocampal formation: A review of anatomical data." Neuroscience,
   31(3), 571-591. DOI: 10.1016/0306-4522(89)90424-7
   Anatomy of Schaffer collateral and temporoammonic projections to CA1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import NamedTuple


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class CA1Config:
    """
    Configuration for the CA1 comparator module.

    Contains parameters for both the spiking substrate (Izhikevich dynamics)
    and the mismatch readout (dual-component comparison).
    """

    # ---- Dimensions ----

    # Dimensionality of the CA3 attractor output (Schaffer collateral input).
    ca3_dim: int = 512

    # Dimensionality of the entorhinal cortex input (temporoammonic input).
    # In PRAGMI, this is the full Isocortex bridge width (sum of zone_dims).
    entorhinal_dim: int = 1664

    # Number of CA1 pyramidal neurons in the model population.
    ca1_num_neurons: int = 512

    # ---- Izhikevich neuron parameters (CA1 pyramidal cells) ----
    # Source: hippocampus_genes.py neuron type "CA1", derived from:
    # Izhikevich (2003). DOI: 10.1109/TNN.2003.820440, Table 1.
    #
    # a: time scale of recovery variable u. Smaller a = slower recovery.
    # b: sensitivity of u to subthreshold fluctuations of v.
    # c: after-spike reset value of v (mV).
    # d: after-spike increment of u.
    #
    # Effective membrane time constant: Under these parameters (a=0.02,
    # b=0.2), the subthreshold dynamics produce an effective tau_m of
    # approximately 20-25 ms, consistent with rodent CA1 pyramidal cell
    # measurements (18-28 ms range depending on dorso-ventral position).
    # Reference: Staff, N.P. et al. (2000). "Resting and active properties
    # of pyramidal neurons in subiculum and CA1 of rat hippocampus."
    # Journal of Neurophysiology, 84(5), 2398-2408.
    # DOI: 10.1152/jn.2000.84.5.2398
    # Note: tau_m is not an explicit parameter in the Izhikevich model;
    # it emerges from the quadratic and linear voltage terms.
    izh_a: float = 0.02
    izh_b: float = 0.2
    izh_c: float = -65.0

    # d=6.0 is CA1-specific, lower than the standard RS value of d=8.0.
    # This produces less spike-frequency adaptation, consistent with CA1
    # pyramidal cells showing less burst adaptation than DG granule cells.
    # CA3 uses d=4.0 (more bursting), DG uses d=8.0 (strong adaptation).
    # Reference for CA1-specific Izhikevich fitting:
    # Ferguson, K.A. et al. (2014). "A simple, biologically-constrained
    # CA1 pyramidal cell model." {DOI to be added later.}
    izh_d: float = 6.0

    # Integration time step in milliseconds.
    # Must match the dt used in the rest of the spiking simulation.
    # NOT a biological time constant. Engineering choice for numerical
    # stability with the Izhikevich equations.
    dt: float = 0.5

    # ---- Mismatch readout parameters ----

    # Weighting between semantic and perceptual mismatch components.
    # NOT a biological quantity. Hyperparameter.
    alpha: float = 0.6

    # Dual thresholds for three-regime mismatch logic.
    # Biological basis: Helfer & Shultz (2019). DOI: 10.1002/hipo.23187.
    # Specific values are engineering choices.
    reconsolidation_threshold: float = 0.2
    encoding_threshold: float = 0.5


# =========================================================================
# Output types
# =========================================================================

class CA1SpikeOutput(NamedTuple):
    """Output of the CA1 spiking substrate. Consumed by STDP eligibility traces."""
    spikes: torch.Tensor          # (batch, ca1_num_neurons) binary
    v_membrane: torch.Tensor      # (batch, ca1_num_neurons) membrane potentials
    u_recovery: torch.Tensor      # (batch, ca1_num_neurons) recovery variables


class MismatchSignal(NamedTuple):
    """Output of the CA1 mismatch readout. Consumed by the Allocortex controller."""
    total_mismatch: torch.Tensor
    perceptual_error: torch.Tensor
    semantic_error: torch.Tensor
    action: str  # "none", "reconsolidate", or "encode"


# =========================================================================
# The CA1 Comparator
# =========================================================================

class CA1Comparator(nn.Module):
    """
    Biological Name: CA1 Pyramidal Layer with Mismatch Detection.

    Plain English: Integrates two convergent information streams (memory from
    CA3, reality from EC) through a population of spiking neurons, producing
    both spike outputs (for plasticity) and a continuous mismatch signal
    (for the memory controller).

    The spiking substrate (forward) runs every integration step. The mismatch
    readout (detect_mismatch) runs once per theta cycle after dynamics settle.
    Spikes feed the eligibility trace buffer. Mismatch feeds the allocortex
    controller. Both outputs are needed; they flow to different consumers.

    ANATOMICAL BOUNDARIES:
        Input 1 (ca3_input): Schaffer collateral projection.
            Sender: CA3 pyramidal cells.
            Receiver: CA1 proximal apical dendrites.
            Ref: Amaral & Witter (1989). DOI: 10.1016/0306-4522(89)90424-7

        Input 2 (ec_input): Temporoammonic pathway.
            Sender: EC layer III.
            Receiver: CA1 distal apical dendrites.
            Ref: Witter et al. (2000). DOI: 10.1111/j.1749-6632.2000.tb06716.x

    References:
        Vinogradova (2001). DOI: 10.1002/hipo.1073
        Izhikevich (2003). DOI: 10.1109/TNN.2003.820440
    """

    def __init__(self, cfg: CA1Config = None):
        super().__init__()
        if cfg is None:
            cfg = CA1Config()
        self.cfg = cfg

        # ==================================================================
        # Synaptic weight matrices (two convergent pathways)
        # ==================================================================

        # Schaffer collateral weights (CA3 -> CA1 proximal dendrites).
        # Fan-in normalized so expected magnitude of I_schaffer is O(1)
        # regardless of ca3_dim. Prevents pathway dominance at init.
        # NOT a biological quantity. Standard fan-in normalization.
        self.schaffer_collateral_weights = nn.Parameter(
            torch.randn(cfg.ca3_dim, cfg.ca1_num_neurons) / math.sqrt(cfg.ca3_dim)
        )

        # Temporoammonic weights (EC -> CA1 distal dendrites).
        # Same fan-in normalization.
        self.temporoammonic_weights = nn.Parameter(
            torch.randn(cfg.entorhinal_dim, cfg.ca1_num_neurons) / math.sqrt(cfg.entorhinal_dim)
        )

        # ==================================================================
        # Izhikevich neuron state
        # ==================================================================

        # v: membrane potential (mV). Resting at -65 mV.
        # Ref: Izhikevich (2003). DOI: 10.1109/TNN.2003.820440
        self.register_buffer("v", torch.full((cfg.ca1_num_neurons,), -65.0))

        # u: recovery variable. Initialized at b * v_rest.
        # Ref: Izhikevich (2003). DOI: 10.1109/TNN.2003.820440
        self.register_buffer("u", torch.full((cfg.ca1_num_neurons,), cfg.izh_b * -65.0))

        # Per-neuron heterogeneity in reset parameters (c, d).
        # CA1 pyramidal cells show electrophysiological diversity across
        # the deep/superficial and proximo-distal axes, including differences
        # in excitability, modulation, and burst propensity.
        # Ref: Graves, A.R. et al. (2012). "Hippocampal pyramidal neurons
        # comprise two distinct cell types that have different impacts on
        # network activity." Neuron, 31(3), 1002-1012.
        # DOI: 10.1016/j.neuron.2012.09.036
        #
        # Implementation matches hippocampus_genes.py: r ~ U(0,1),
        # c += 10*r^2, d -= 4*r^2. This spreads the reset voltage and
        # adaptation increment across the population, producing a mix
        # of regular spiking and weakly bursting phenotypes.
        r = torch.rand(cfg.ca1_num_neurons)
        self.register_buffer(
            "c_reset",
            torch.full((cfg.ca1_num_neurons,), cfg.izh_c) + 10.0 * r**2
        )
        self.register_buffer(
            "d_reset",
            torch.full((cfg.ca1_num_neurons,), cfg.izh_d) - 4.0 * r**2
        )

        # ==================================================================
        # Mismatch readout projectors (Layer 2)
        # ==================================================================

        # Separate projections for each stream into semantic comparison space.
        # NOT a direct biological mechanism. Engineering approximation of
        # CA1 dendritic integration.
        self.reality_semantic_proj = nn.Linear(cfg.entorhinal_dim, cfg.ca1_num_neurons, bias=False)
        self.memory_semantic_proj = nn.Linear(cfg.ca3_dim, cfg.ca1_num_neurons, bias=False)

        # Perceptual comparison projections (shared target dimensionality).
        self.reality_perceptual_proj = nn.Linear(cfg.entorhinal_dim, cfg.ca1_num_neurons, bias=False)
        self.memory_perceptual_proj = nn.Linear(cfg.ca3_dim, cfg.ca1_num_neurons, bias=False)

    # ==================================================================
    # Layer 1: Spiking Substrate
    # ==================================================================

    def forward(
        self,
        ca3_input: torch.Tensor,
        ec_input: torch.Tensor,
        theta_phase: float = None,
    ) -> CA1SpikeOutput:
        """
        Biological Name: CA1 Pyramidal Cell Izhikevich Dynamics.

        Plain English: Integrates memory prediction (CA3, Schaffer collaterals)
        and sensory reality (EC, temporoammonic path) as synaptic currents.
        Produces spike outputs for the STDP eligibility trace system.

        Theta-phase modulation suppresses memory input during encoding to
        prevent interference with new storage.
        Ref: Hasselmo & Schnell (1994). DOI: 10.1523/JNEUROSCI.14-06-03898.1994

        Args:
            ca3_input: (batch, ca3_dim) or (ca3_dim,) CA3 output.
            ec_input: (batch, entorhinal_dim) or (entorhinal_dim,) EC state.
            theta_phase: optional float in [0, 1] for gain modulation.

        Returns:
            CA1SpikeOutput with spikes, membrane potentials, recovery vars.
        """
        if ca3_input.dim() == 1:
            ca3_input = ca3_input.unsqueeze(0)
        if ec_input.dim() == 1:
            ec_input = ec_input.unsqueeze(0)

        batch_size = ca3_input.shape[0]

        # Expand neuron state to batch dimension.
        v = self.v.unsqueeze(0).expand(batch_size, -1).clone()
        u = self.u.unsqueeze(0).expand(batch_size, -1).clone()

        # ---- Synaptic currents ----

        # I_schaffer: memory prediction from CA3 via Schaffer collaterals.
        I_schaffer = torch.matmul(ca3_input, self.schaffer_collateral_weights)

        # Theta modulation of Schaffer gain.
        # Ref: Hasselmo & Schnell (1994). DOI: 10.1523/JNEUROSCI.14-06-03898.1994
        if theta_phase is not None:
            retrieval_gain = 0.5 * (1.0 + math.cos(2.0 * math.pi * theta_phase))
            I_schaffer = I_schaffer * retrieval_gain

        # I_temporoammonic: sensory reality from EC.
        I_temporoammonic = torch.matmul(ec_input, self.temporoammonic_weights)

        I_total = I_schaffer + I_temporoammonic

        # ---- Izhikevich dynamics ----
        # Ref: Izhikevich (2003). DOI: 10.1109/TNN.2003.820440
        dt = self.cfg.dt

        dv = 0.04 * v**2 + 5.0 * v + 140.0 - u + I_total
        v = v + dv * dt

        # Physiological voltage clamp: Na+ reversal ~+40mV, K+ ~-90mV.
        v = torch.clamp(v, -90.0, 40.0)

        du = self.cfg.izh_a * (self.cfg.izh_b * v - u)
        u = u + du * dt

        # Spike detection at +30 mV threshold.
        spikes = (v >= 30.0).float()

        # After-spike reset with per-neuron heterogeneity.
        c_exp = self.c_reset.unsqueeze(0).expand_as(v)
        d_exp = self.d_reset.unsqueeze(0).expand_as(u)
        v = torch.where(spikes > 0, c_exp, v)
        u = torch.where(spikes > 0, u + d_exp, u)

        # Persist the canonical neuron state (first batch element).
        # For single-episode rollout (batch=1), this is exact.
        # For multi-batch training (parallel independent episodes), this
        # persists only the first sample's state. Multi-batch use requires
        # either external state management per episode or separate CA1
        # instances. The .soul serialization captures this single-episode
        # canonical state.
        self.v.copy_(v[0].detach())
        self.u.copy_(u[0].detach())

        return CA1SpikeOutput(
            spikes=spikes,
            v_membrane=v.detach(),
            u_recovery=u.detach(),
        )

    # ==================================================================
    # Layer 2: Mismatch Readout
    # ==================================================================

    def detect_mismatch(
        self,
        ca3_reconstruction: torch.Tensor,
        ec_reality: torch.Tensor,
    ) -> MismatchSignal:
        """
        Biological Name: CA1 Population-Level Mismatch Readout.

        Plain English: Extracts a graded mismatch signal from the convergent
        CA3 (memory) and EC (reality) inputs. Returns both the continuous
        mismatch value and a three-way action decision for the Allocortex.

        Two components:
            1. Perceptual error (MSE): surface-level feature deviation.
            2. Semantic error (cosine distance): conceptual shift.

        Ref: Helfer & Shultz (2019). DOI: 10.1002/hipo.23187

        Args:
            ca3_reconstruction: (batch, ca3_dim) or (ca3_dim,) memory signal.
            ec_reality: (batch, entorhinal_dim) or (entorhinal_dim,) reality.

        Returns:
            MismatchSignal with total mismatch, components, and action.
        """
        if ca3_reconstruction.dim() == 1:
            ca3_reconstruction = ca3_reconstruction.unsqueeze(0)
        if ec_reality.dim() == 1:
            ec_reality = ec_reality.unsqueeze(0)

        # ---- Perceptual mismatch (MSE in shared projection space) ----
        reality_percept = self.reality_perceptual_proj(ec_reality)
        memory_percept = self.memory_perceptual_proj(ca3_reconstruction)
        perceptual_error = F.mse_loss(
            reality_percept, memory_percept, reduction='none'
        ).mean(dim=-1)

        # ---- Semantic mismatch (cosine distance in learned space) ----
        reality_semantic = self.reality_semantic_proj(ec_reality)
        memory_semantic = self.memory_semantic_proj(ca3_reconstruction)
        semantic_error = 1.0 - F.cosine_similarity(
            reality_semantic, memory_semantic, dim=-1
        )

        # ---- Weighted combination ----
        total_mismatch = (
            self.cfg.alpha * semantic_error
            + (1.0 - self.cfg.alpha) * perceptual_error
        )

        # ---- Dual-threshold action ----
        # Ref: Helfer & Shultz (2019). DOI: 10.1002/hipo.23187
        # Ref: Vinogradova (2001). DOI: 10.1002/hipo.1073
        #
        # Action is determined from the batch mean mismatch. For batch=1
        # (typical single-episode wake loop), this is exact. For batch>1,
        # the mean may mask per-sample variation. If per-sample actions are
        # needed (e.g., parallel independent episodes), threshold
        # total_mismatch elementwise and return a per-sample action tensor
        # instead of a single string.
        mean_mm = total_mismatch.mean().item()

        if mean_mm > self.cfg.encoding_threshold:
            action = "encode"
        elif mean_mm > self.cfg.reconsolidation_threshold:
            action = "reconsolidate"
        else:
            action = "none"

        return MismatchSignal(
            total_mismatch=total_mismatch,
            perceptual_error=perceptual_error,
            semantic_error=semantic_error,
            action=action,
        )

    # ==================================================================
    # Serialization
    # ==================================================================

    def get_state(self) -> dict:
        """Capture spiking neuron state for the .soul file."""
        return {
            "v": self.v.cpu().clone(),
            "u": self.u.cpu().clone(),
            "c_reset": self.c_reset.cpu().clone(),
            "d_reset": self.d_reset.cpu().clone(),
        }

    def restore_state(self, state: dict):
        """Restore neuron state from a .soul checkpoint."""
        self.v.copy_(state["v"])
        self.u.copy_(state["u"])
        self.c_reset.copy_(state["c_reset"])
        self.d_reset.copy_(state["d_reset"])

    def reset_dynamics(self):
        """
        Reset membrane potentials and recovery variables to resting state.
        Used during the glymphatic sweep phase to clear residual activation.
        NOT modeling actual glymphatic chemistry. Functional reset.
        """
        self.v.fill_(-65.0)
        self.u.copy_(self.cfg.izh_b * self.v)
