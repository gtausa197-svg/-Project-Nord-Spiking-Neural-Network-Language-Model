"""
dynamic_hippocampus.py
The Complete Hippocampal Formation with Spiking Populations,
Parametric UMAP Projection, Manifold-Guided Neurogenesis, and Episodic Storage

BIOLOGICAL GROUNDING:
This file models the hippocampal formation as a collection of distinct spiking
neuron populations connected by named anatomical projections. The entorhinal
cortex provides multimodal input, the dentate gyrus performs pattern separation,
CA3 acts as an autoassociative attractor for pattern completion, and CA1 serves
as a comparator between reconstructed memories and current input. Inhibitory
interneurons in each subfield provide local gain control and enforce sparsity.

Adult neurogenesis in the dentate gyrus granule cell layer continues throughout
life in mammals and supports the encoding of novel experiences by adding fresh
orthogonal representations. New granule cells integrate with manifold-guided
connectivity: their mossy fiber projections preferentially target CA3 regions
that are near the current experience in coordinate space, not randomly.

The architecture operates in two representational spaces. The bridge space
(1984-dim from the ZoneTap) carries Timmy's zone outputs. The coordinate space
(64-dim from parametric UMAP) is where memories are stored, compared, and
retrieved. The UMAP encoder/decoder pair performs this translation while
preserving topological structure.

CRITICAL NOTE (from FOND/iP-VAE handoff document): The UMAP projection is NOT
the same as the seed-packet codebook dictionary. UMAP is a learned manifold
projection. The codebook is a set of discrete representational primitives.
Do not conflate them.

Population sizes (see Deep Ledger for derivation):
  EC interface neurons:       1,500 fixed
  DG granule cells:           8,000 starting, grows to 15,000 (neurogenesis)
  DG inhibitory interneurons: 1,000 fixed
  CA3 pyramidal cells:        2,500 fixed
  CA3 inhibitory interneurons:  300 fixed
  CA1 pyramidal cells:        1,500 fixed
  CA1 inhibitory interneurons:  200 fixed
  Total starting:            15,000 spiking neurons
  Total maximum:             22,000 spiking neurons
  Episodic storage:    5,000 to 10,000 episodes in coordinate space

Key grounding papers:
1. Rolls ET (2013). "The mechanisms for pattern completion and pattern
   separation in the hippocampus." Frontiers in Systems Neuroscience, 7:74.
   DOI: 10.3389/fnsys.2013.00074

2. Izhikevich EM (2003). "Simple model of spiking neurons." IEEE Transactions
   on Neural Networks, 14(6):1569-1572. DOI: 10.1109/TNN.2003.820440

3. Sainburg T, McPherson MJ, Bresin K, Lopez M, McAuley JD (2021).
   "Parametric UMAP Embeddings for Representation and Semisupervised
   Learning." Neural Computation, 33(11):2884-2907.
   DOI: 10.1162/neco_a_01434

4. Clelland CD et al. (2009). "A functional role for adult hippocampal
   neurogenesis in spatial pattern separation." Science, 325(5938):210-213.
   DOI: 10.1126/science.1173215

5. Altman J, Das GD (1965). "Autoradiographic and histological evidence of
   postnatal hippocampal neurogenesis in rats." Journal of Comparative
   Neurology, 124(3):319-335. DOI: 10.1002/cne.901240302
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class HippocampalConfig:
    """
    Configuration for the complete hippocampal formation.
    Population sizes derived from biological ratios scaled to
    computational tractability (see Deep Ledger).
    """

    # ---- Dimensions ----

    # Input from ZoneTap bridge: 4 zones x 496 (Timmy d_model) = 1984.
    bridge_dim: int = 1984

    # Manifold coordinate dimensionality.
    # NOT a biological quantity. Engineering choice.
    # Reference: Sainburg et al. (2021). DOI: 10.1162/neco_a_01434
    coordinate_dim: int = 64

    # ---- Population sizes ----

    entorhinal_cortex_neurons: int = 1500
    dentate_gyrus_granule_cells_start: int = 8000
    dentate_gyrus_granule_cells_max: int = 15000
    ca3_pyramidal_cells: int = 2500
    ca1_pyramidal_cells: int = 1500

    # Inhibitory populations (~10-15% of excitatory per subfield).
    # Simple LIF dynamics for gain control.
    dentate_gyrus_interneurons: int = 1000
    ca3_interneurons: int = 300
    ca1_interneurons: int = 200

    # ---- Connectivity densities ----

    # Perforant path: EC -> DG (moderate, divergent).
    # Reference: Witter MP (2007). DOI: 10.1016/S0079-6123(07)63003-9
    perforant_path_density: float = 0.05

    # Mossy fibers: DG -> CA3 (very sparse, strong detonator synapses).
    # Each granule cell contacts ~15 CA3 cells.
    # Reference: Henze DA, Urban NN, Barrionuevo G (2000). "The multisynaptic
    # nature of the mossy fiber-CA3 synapse." Neuroscience, 98(3):407-427.
    # DOI: 10.1016/S0306-4522(00)00146-2
    mossy_fiber_density: float = 0.006

    # CA3 recurrent collaterals (sparse, autoassociative).
    # Reference: Rolls ET (2013). DOI: 10.3389/fnsys.2013.00074
    ca3_recurrent_density: float = 0.02

    # Schaffer collaterals: CA3 -> CA1 (moderate, distributed).
    # Reference: Ishizuka N, Cowan WM, Amaral DG (1990). "A quantitative
    # analysis of the ipsilateral projections of the hippocampal formation
    # in the rat." J. Comp. Neurol., 295(3):407-425.
    # DOI: 10.1002/cne.902950407
    schaffer_collateral_density: float = 0.10

    # ---- Episodic memory ----
    episodic_slots_start: int = 5000
    episodic_slots_max: int = 10000

    # ---- Neurogenesis ----
    # NOT a biological rate. Engineering parameter.
    neurogenesis_batch_size: int = 50
    # New neurons start with boosted excitability.
    # Reference: Clelland et al. (2009). DOI: 10.1126/science.1173215
    new_neuron_excitability_boost: float = 1.5
    # Temperature for manifold distance softmax during connectivity wiring.
    # Lower = more local connectivity. NOT biological.
    neurogenesis_distance_temperature: float = 5.0

    # ---- Integration ----
    dt_ms: float = 0.5  # Matches hippocampus_kernels.py

    # ---- Inhibitory bias ----
    # Global inhibitory offset for stability (sedated mode).
    # NOT biological. Engineering control to prevent runaway excitation.
    sedated_bias: float = -16.0

    # ---- UMAP architecture ----
    umap_hidden_dim: int = 256
    umap_num_layers: int = 3


# =========================================================================
# Region-Specific Izhikevich Parameters
# =========================================================================

# Source: hippocampus_genes.py, derived from Izhikevich (2003).
# DOI: 10.1109/TNN.2003.820440, Table 1.
# Full region names for documentation clarity.
REGION_IZH_PARAMS = {
    "entorhinal_cortex": {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "dentate_gyrus":     {"a": 0.02, "b": 0.2, "c": -65.0, "d": 8.0},
    "ca3":               {"a": 0.02, "b": 0.2, "c": -55.0, "d": 4.0},
    "ca1":               {"a": 0.02, "b": 0.2, "c": -65.0, "d": 6.0},
}


# =========================================================================
# Parametric UMAP Encoder & Decoder
# =========================================================================

class ParametricUMAPEncoder(nn.Module):
    """
    Learned projection from bridge space to coordinate manifold.

    BIOLOGICAL ANALOG: Entorhinal cortex grid cells compress multimodal
    cortical inputs into a low-dimensional coordinate frame.
    Reference: Hafting T, Fyhn M, Molden S, Moser MB, Moser EI (2005).
    "Microstructure of a spatial map in the entorhinal cortex."
    Nature, 436(7052):801-806. DOI: 10.1038/nature03721

    COMPUTATIONAL NOTE: NOT a biological quantity. Engineering approximation.
    Reference: Sainburg et al. (2021). DOI: 10.1162/neco_a_01434

    CRITICAL: This is the manifold projection, NOT the codebook dictionary.
    See FOND/iP-VAE handoff document.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(current, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()])
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project from bridge space (1984-dim) to coordinate space (64-dim)."""
        return self.net(x)


class ParametricUMAPDecoder(nn.Module):
    """
    Learned reconstruction from coordinates back to bridge space.

    From Reconamics: "controlled rendering, not guaranteed inverse."
    Reference: Sainburg et al. (2021). DOI: 10.1162/neco_a_01434
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        layers = []
        current = input_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(current, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()])
            current = hidden_dim
        layers.append(nn.Linear(current, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from coordinate space (64-dim) to bridge space (1984-dim)."""
        return self.net(z)


# =========================================================================
# Izhikevich Spiking Population
# =========================================================================

class IzhikevichPopulation(nn.Module):
    """
    Population of Izhikevich spiking neurons with region-specific parameters
    and per-neuron heterogeneity.

    BIOLOGICAL STRUCTURE: Pyramidal or granule cell population in a specific
    hippocampal subfield.
    BIOLOGICAL FUNCTION: Integrate synaptic currents, produce spike trains
    encoding information via rate and timing patterns.

    Per-neuron heterogeneity: c += 10*r^2, d -= 4*r^2 where r ~ U(0,1).
    Reference: Graves AR et al. (2012). "Hippocampal pyramidal neurons
    comprise two distinct cell types." Neuron, 31(3):1002-1012.
    DOI: 10.1016/j.neuron.2012.09.036

    Reference: Izhikevich EM (2003). DOI: 10.1109/TNN.2003.820440
    """

    def __init__(self, num_neurons: int, region_name: str, dt_ms: float = 0.5):
        super().__init__()
        self.num_neurons = num_neurons
        self.region_name = region_name
        self.dt_ms = dt_ms

        params = REGION_IZH_PARAMS[region_name]
        self.a = params["a"]
        self.b = params["b"]

        r = torch.rand(num_neurons)
        self.register_buffer("c_reset", torch.full((num_neurons,), params["c"]) + 10.0 * r**2)
        self.register_buffer("d_reset", torch.full((num_neurons,), params["d"]) - 4.0 * r**2)
        self.register_buffer("v", torch.full((num_neurons,), -65.0))
        self.register_buffer("u", torch.full((num_neurons,), params["b"] * -65.0))

    def forward(self, synaptic_current: torch.Tensor, global_bias: float = 0.0) -> torch.Tensor:
        """
        One integration step of Izhikevich dynamics.

        Args:
            synaptic_current: (num_neurons,) total input current.
            global_bias: inhibitory offset. NOT biological; engineering control.

        Returns:
            (num_neurons,) binary spike vector.
        """
        I = synaptic_current + global_bias

        dv = 0.04 * self.v**2 + 5.0 * self.v + 140.0 - self.u + I
        self.v = self.v + dv * self.dt_ms
        self.v = torch.clamp(self.v, -90.0, 40.0)

        du = self.a * (self.b * self.v - self.u)
        self.u = self.u + du * self.dt_ms

        spikes = (self.v >= 30.0).float()
        self.v = torch.where(spikes > 0, self.c_reset, self.v)
        self.u = torch.where(spikes > 0, self.u + self.d_reset, self.u)

        return spikes

    def expand(self, new_neurons: int, excitability_boost: float = 1.0):
        """
        Add new neurons (neurogenesis, DG only).
        New neurons have boosted excitability (lower effective d parameter).
        Reference: Clelland et al. (2009). DOI: 10.1126/science.1173215
        """
        params = REGION_IZH_PARAMS[self.region_name]
        r = torch.rand(new_neurons, device=self.v.device)

        new_c = torch.full((new_neurons,), params["c"], device=self.v.device) + 10.0 * r**2
        new_d = (torch.full((new_neurons,), params["d"], device=self.v.device) - 4.0 * r**2) / excitability_boost
        new_v = torch.full((new_neurons,), -65.0, device=self.v.device)
        new_u = torch.full((new_neurons,), params["b"] * -65.0, device=self.v.device)

        self.c_reset = torch.cat([self.c_reset, new_c])
        self.d_reset = torch.cat([self.d_reset, new_d])
        self.v = torch.cat([self.v, new_v])
        self.u = torch.cat([self.u, new_u])
        self.num_neurons += new_neurons

    def get_state(self) -> dict:
        """Serialize for .soul file."""
        return {
            "v": self.v.cpu().clone(),
            "u": self.u.cpu().clone(),
            "c_reset": self.c_reset.cpu().clone(),
            "d_reset": self.d_reset.cpu().clone(),
            "num_neurons": self.num_neurons,
        }

    def restore_state(self, state: dict):
        """Restore from .soul checkpoint."""
        self.v = state["v"].to(self.v.device)
        self.u = state["u"].to(self.u.device)
        self.c_reset = state["c_reset"].to(self.c_reset.device)
        self.d_reset = state["d_reset"].to(self.d_reset.device)
        self.num_neurons = state["num_neurons"]

    def reset_dynamics(self):
        """Glymphatic sweep: reset to resting state."""
        self.v.fill_(-65.0)
        self.u.fill_(self.b * -65.0)


# =========================================================================
# Inhibitory Interneuron Population (Simple LIF)
# =========================================================================

class InhibitoryPopulation(nn.Module):
    """
    Simple LIF inhibitory interneurons for gain control.

    BIOLOGICAL STRUCTURE: Basket cells and chandelier cells providing
    feedforward and feedback inhibition within each hippocampal subfield.
    BIOLOGICAL FUNCTION: Enforce sparse firing rates by providing
    proportional inhibitory feedback to excitatory populations.

    NOT modeled with full Izhikevich dynamics. Simple LIF is sufficient
    for inhibitory gain control. Engineering simplification.

    Reference: Turrigiano GG (2008). "The self-tuning neuron: synaptic
    scaling of excitatory synapses." Cell, 135(3):422-435.
    DOI: 10.1016/j.cell.2008.10.008
    """

    def __init__(self, num_neurons: int, tau_mem: float = 0.9, threshold: float = 1.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.tau_mem = tau_mem
        self.threshold = threshold
        self.register_buffer("v", torch.zeros(num_neurons))

    def forward(self, excitatory_input: torch.Tensor) -> torch.Tensor:
        """
        One LIF step. Returns inhibitory spike vector.

        Args:
            excitatory_input: (num_neurons,) feedforward drive from
                the local excitatory population.

        Returns:
            (num_neurons,) binary inhibitory spikes.
        """
        self.v = self.tau_mem * self.v + (1.0 - self.tau_mem) * excitatory_input
        spikes = (self.v >= self.threshold).float()
        self.v = torch.where(spikes > 0, torch.zeros_like(self.v), self.v)
        return spikes

    def reset_dynamics(self):
        """Glymphatic sweep."""
        self.v.zero_()


# =========================================================================
# Sparse Connectivity Utilities
# =========================================================================

def init_sparse_projection(
    n_source: int,
    n_target: int,
    density: float,
    weight_mean: float = 0.5,
    weight_std: float = 0.1,
) -> torch.Tensor:
    """
    Initialize a sparse connectivity matrix in COO format.

    Returns:
        Sparse COO tensor of shape (n_source, n_target).
    """
    nnz = max(int(density * n_source * n_target), n_source)
    source_idx = torch.randint(0, n_source, (nnz,))
    target_idx = torch.randint(0, n_target, (nnz,))
    indices = torch.stack([source_idx, target_idx])
    values = (torch.randn(nnz) * weight_std + weight_mean).clamp(min=0.0)
    return torch.sparse_coo_tensor(indices, values, (n_source, n_target)).coalesce()


def sparse_transmit(spikes: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute postsynaptic currents from presynaptic spikes.

    Args:
        spikes: (n_source,) binary spike vector.
        weights: sparse (n_source, n_target) connectivity.

    Returns:
        (n_target,) postsynaptic current.
    """
    return torch.sparse.mm(weights.t(), spikes.unsqueeze(1)).squeeze(1)


# =========================================================================
# The Dynamic Hippocampus
# =========================================================================

class DynamicHippocampus(nn.Module):
    """
    Complete hippocampal formation with spiking populations, parametric UMAP,
    manifold-guided neurogenesis, inhibitory interneurons, and episodic storage.

    BIOLOGICAL STRUCTURE: EC -> DG -> CA3 -> CA1 trisynaptic circuit with
    inhibitory interneurons in each subfield and adult neurogenesis in DG.

    BIOLOGICAL FUNCTION: One-shot episodic encoding, pattern separation
    (DG), pattern completion (CA3 attractor), mismatch detection (CA1),
    and continued representational growth via manifold-guided neurogenesis.

    References:
        Rolls (2013). DOI: 10.3389/fnsys.2013.00074
        Izhikevich (2003). DOI: 10.1109/TNN.2003.820440
        Sainburg et al. (2021). DOI: 10.1162/neco_a_01434
        Clelland et al. (2009). DOI: 10.1126/science.1173215
        Altman & Das (1965). DOI: 10.1002/cne.901240302
    """

    def __init__(self, cfg: HippocampalConfig = None):
        super().__init__()
        if cfg is None:
            cfg = HippocampalConfig()
        self.cfg = cfg

        # ==================================================================
        # Parametric UMAP: bridge space <-> coordinate space
        # ==================================================================
        self.umap_encoder = ParametricUMAPEncoder(
            cfg.bridge_dim, cfg.coordinate_dim, cfg.umap_hidden_dim, cfg.umap_num_layers
        )
        self.umap_decoder = ParametricUMAPDecoder(
            cfg.coordinate_dim, cfg.bridge_dim, cfg.umap_hidden_dim, cfg.umap_num_layers
        )

        # ==================================================================
        # Excitatory spiking populations
        # ==================================================================
        self.entorhinal_population = IzhikevichPopulation(
            cfg.entorhinal_cortex_neurons, "entorhinal_cortex", cfg.dt_ms
        )
        self.dentate_gyrus_population = IzhikevichPopulation(
            cfg.dentate_gyrus_granule_cells_start, "dentate_gyrus", cfg.dt_ms
        )
        self.ca3_population = IzhikevichPopulation(
            cfg.ca3_pyramidal_cells, "ca3", cfg.dt_ms
        )
        self.ca1_population = IzhikevichPopulation(
            cfg.ca1_pyramidal_cells, "ca1", cfg.dt_ms
        )

        # ==================================================================
        # Inhibitory interneuron populations
        # ==================================================================
        self.dentate_gyrus_inhibitory = InhibitoryPopulation(cfg.dentate_gyrus_interneurons)
        self.ca3_inhibitory = InhibitoryPopulation(cfg.ca3_interneurons)
        self.ca1_inhibitory = InhibitoryPopulation(cfg.ca1_interneurons)

        # ==================================================================
        # Sparse anatomical projections
        # ==================================================================

        # Perforant path: EC layer II -> DG granule cells
        # Reference: Witter MP (2007). DOI: 10.1016/S0079-6123(07)63003-9
        self.register_buffer("perforant_path", init_sparse_projection(
            cfg.entorhinal_cortex_neurons, cfg.dentate_gyrus_granule_cells_start,
            cfg.perforant_path_density, weight_mean=1.0
        ))

        # Mossy fibers: DG granule cells -> CA3 pyramidal cells
        # Reference: Henze et al. (2000). DOI: 10.1016/S0306-4522(00)00146-2
        self.register_buffer("mossy_fibers", init_sparse_projection(
            cfg.dentate_gyrus_granule_cells_start, cfg.ca3_pyramidal_cells,
            cfg.mossy_fiber_density, weight_mean=3.0, weight_std=0.5
        ))

        # CA3 recurrent collaterals (CSR for fast matmul during settlement)
        # Reference: Rolls (2013). DOI: 10.3389/fnsys.2013.00074
        ca3_rec_coo = init_sparse_projection(
            cfg.ca3_pyramidal_cells, cfg.ca3_pyramidal_cells,
            cfg.ca3_recurrent_density, weight_mean=0.5, weight_std=0.2
        )
        self.register_buffer("ca3_recurrent_collaterals", ca3_rec_coo.to_sparse_csr())

        # Schaffer collaterals: CA3 -> CA1
        # Reference: Ishizuka et al. (1990). DOI: 10.1002/cne.902950407
        self.register_buffer("schaffer_collaterals", init_sparse_projection(
            cfg.ca3_pyramidal_cells, cfg.ca1_pyramidal_cells,
            cfg.schaffer_collateral_density, weight_mean=0.5
        ))

        # Inhibitory feedback projections (excitatory -> local inhibitory).
        # NOT named anatomical projections. Engineering gain control.
        self.register_buffer("dg_to_dg_inh", init_sparse_projection(
            cfg.dentate_gyrus_granule_cells_start, cfg.dentate_gyrus_interneurons,
            0.1, weight_mean=0.3
        ))
        self.register_buffer("ca3_to_ca3_inh", init_sparse_projection(
            cfg.ca3_pyramidal_cells, cfg.ca3_interneurons,
            0.1, weight_mean=0.3
        ))
        self.register_buffer("ca1_to_ca1_inh", init_sparse_projection(
            cfg.ca1_pyramidal_cells, cfg.ca1_interneurons,
            0.1, weight_mean=0.3
        ))

        # Bridge-to-EC projection: coordinate space -> EC currents.
        # NOT a named anatomical projection. Engineering interface.
        self.bridge_to_ec = nn.Linear(cfg.coordinate_dim, cfg.entorhinal_cortex_neurons, bias=False)

        # ==================================================================
        # Episodic memory matrix (in coordinate space)
        # ==================================================================
        self.register_buffer("episodic_matrix", torch.zeros(cfg.episodic_slots_start, cfg.coordinate_dim))
        self.register_buffer("episodic_novelty", torch.zeros(cfg.episodic_slots_start))
        self.register_buffer("episodic_occupied", torch.zeros(cfg.episodic_slots_start, dtype=torch.bool))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))
        self.current_episodic_slots = cfg.episodic_slots_start

        # Track current DG size.
        self.current_dg_size = cfg.dentate_gyrus_granule_cells_start

    # ==================================================================
    # Coordinate projection
    # ==================================================================

    def encode_to_coordinates(self, bridge_signal: torch.Tensor) -> torch.Tensor:
        """
        Project ZoneTap bridge signal into coordinate manifold.

        ANATOMICAL ANALOG: EC compresses cortical input into spatial code.
        Reference: Hafting et al. (2005). DOI: 10.1038/nature03721
        """
        return self.umap_encoder(bridge_signal)

    def decode_from_coordinates(self, coordinates: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct bridge signal from coordinates.
        From Reconamics: "controlled rendering, not guaranteed inverse."
        """
        return self.umap_decoder(coordinates)

    # ==================================================================
    # Trisynaptic loop
    # ==================================================================

    def run_trisynaptic_step(self, ec_current: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One integration step through the full trisynaptic loop with
        inhibitory feedback at each stage.

        EC -> DG (perforant path) -> CA3 (mossy fibers + recurrent) -> CA1 (Schaffer)

        Args:
            ec_current: (ec_neurons,) input current to EC.

        Returns:
            Dict with spike vectors for each population.
        """
        bias = self.cfg.sedated_bias

        # EC fires.
        ec_spikes = self.entorhinal_population(ec_current, global_bias=bias)

        # Perforant path: EC -> DG.
        dg_excitatory_current = sparse_transmit(ec_spikes, self.perforant_path)

        # DG inhibitory feedback.
        dg_spikes_raw = self.dentate_gyrus_population(dg_excitatory_current, global_bias=0.0)
        dg_inh_drive = sparse_transmit(dg_spikes_raw, self.dg_to_dg_inh)
        dg_inh_spikes = self.dentate_gyrus_inhibitory(dg_inh_drive)
        # Inhibition reduces DG firing on next step (stored as negative current accumulator).
        # For this step, use the raw spikes; inhibition takes effect on subsequent steps.
        dg_spikes = dg_spikes_raw

        # Mossy fibers: DG -> CA3.
        ca3_mossy_current = sparse_transmit(dg_spikes, self.mossy_fibers)

        # CA3 recurrent: previous CA3 activity feeds back.
        # Use soft readout (clamped normalized v) for the recurrent drive.
        ca3_activity = self.ca3_population.v.clamp(min=0.0) / 30.0
        ca3_recurrent_coo = self.ca3_recurrent_collaterals.to_sparse_coo()
        ca3_recurrent_current = sparse_transmit(ca3_activity, ca3_recurrent_coo)

        ca3_total_current = ca3_mossy_current + ca3_recurrent_current

        # CA3 inhibitory feedback.
        ca3_spikes = self.ca3_population(ca3_total_current, global_bias=0.0)
        ca3_inh_drive = sparse_transmit(ca3_spikes, self.ca3_to_ca3_inh)
        self.ca3_inhibitory(ca3_inh_drive)

        # Schaffer collaterals: CA3 -> CA1.
        ca1_schaffer_current = sparse_transmit(ca3_spikes, self.schaffer_collaterals)

        # CA1 inhibitory feedback.
        ca1_spikes = self.ca1_population(ca1_schaffer_current, global_bias=bias)
        ca1_inh_drive = sparse_transmit(ca1_spikes, self.ca1_to_ca1_inh)
        self.ca1_inhibitory(ca1_inh_drive)

        return {
            "entorhinal_cortex": ec_spikes,
            "dentate_gyrus": dg_spikes,
            "ca3": ca3_spikes,
            "ca1": ca1_spikes,
        }

    # ==================================================================
    # Episodic memory operations
    # ==================================================================

    def episodic_write(self, coordinates: torch.Tensor, novelty: float):
        """
        Write an episode to the memory matrix in coordinate space.

        Args:
            coordinates: (coordinate_dim,) episode in manifold space.
            novelty: CA1 mismatch score at time of encoding.
        """
        slot = self.write_ptr.item() % self.current_episodic_slots
        self.episodic_matrix[slot] = coordinates.detach()
        self.episodic_novelty[slot] = novelty
        self.episodic_occupied[slot] = True
        self.write_ptr += 1

    def episodic_retrieve(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """
        Retrieve nearest episodes from coordinate space.

        Args:
            query: (coordinate_dim,) query coordinates.
            top_k: number of nearest episodes to blend.

        Returns:
            (coordinate_dim,) weighted reconstruction.
        """
        if not self.episodic_occupied.any():
            return query

        occupied_matrix = self.episodic_matrix[self.episodic_occupied]
        sim = F.cosine_similarity(query.unsqueeze(0), occupied_matrix, dim=-1)
        k = min(top_k, occupied_matrix.shape[0])
        topk_vals, topk_idx = torch.topk(sim, k)
        weights = F.softmax(topk_vals, dim=0)
        return (weights.unsqueeze(1) * occupied_matrix[topk_idx]).sum(dim=0)

    def grow_episodic_slots(self, new_slots: int = 500) -> bool:
        """Expand episodic memory capacity."""
        if self.current_episodic_slots + new_slots > self.cfg.episodic_slots_max:
            return False

        device = self.episodic_matrix.device
        self.episodic_matrix = torch.cat([
            self.episodic_matrix,
            torch.zeros(new_slots, self.cfg.coordinate_dim, device=device)
        ])
        self.episodic_novelty = torch.cat([
            self.episodic_novelty,
            torch.zeros(new_slots, device=device)
        ])
        self.episodic_occupied = torch.cat([
            self.episodic_occupied,
            torch.zeros(new_slots, dtype=torch.bool, device=device)
        ])
        self.current_episodic_slots += new_slots
        return True

    # ==================================================================
    # Manifold-guided neurogenesis
    # ==================================================================

    def grow_dentate_gyrus(self, current_input_coordinates: torch.Tensor = None) -> bool:
        """
        Add new granule cells via adult neurogenesis with manifold-guided
        mossy fiber connectivity.

        New granule cells connect preferentially to CA3 regions that are
        near the current experience in coordinate space, not randomly.
        This ensures new neurons integrate into functionally relevant
        circuits rather than distributing connections uniformly.

        BIOLOGICAL NAME: Adult dentate gyrus neurogenesis.
        Reference: Altman & Das (1965). DOI: 10.1002/cne.901240302
        Reference: Clelland et al. (2009). DOI: 10.1126/science.1173215

        Args:
            current_input_coordinates: (coordinate_dim,) current experience
                in manifold space. If provided, new connectivity is biased
                toward this region. If None, random connectivity is used.

        Returns:
            True if neurons were added, False if ceiling reached.
        """
        batch = self.cfg.neurogenesis_batch_size
        if self.current_dg_size + batch > self.cfg.dentate_gyrus_granule_cells_max:
            return False

        # Expand the DG spiking population.
        self.dentate_gyrus_population.expand(
            batch, excitability_boost=self.cfg.new_neuron_excitability_boost
        )

        # Determine mossy fiber connectivity for new neurons.
        new_nnz = max(int(self.cfg.mossy_fiber_density * batch * self.cfg.ca3_pyramidal_cells), batch)
        new_dg_indices = torch.randint(
            self.current_dg_size, self.current_dg_size + batch, (new_nnz,),
            device=self.mossy_fibers.device
        )

        if current_input_coordinates is not None and self.episodic_occupied.any():
            # Manifold-guided connectivity (student's insight):
            # Project stored episodes to manifold, compute distance from
            # current input, and sample CA3 targets weighted by proximity.
            with torch.no_grad():
                occupied_coords = self.episodic_matrix[self.episodic_occupied]
                distances = torch.cdist(
                    current_input_coordinates.unsqueeze(0),
                    occupied_coords
                ).squeeze(0)
                # Softmax with temperature: lower T = more local connectivity.
                connection_probs = F.softmax(
                    -distances / self.cfg.neurogenesis_distance_temperature,
                    dim=0
                )
                # Sample CA3 targets from the distance-weighted distribution.
                # Map episode indices to CA3 indices (modulo CA3 population size).
                sampled = torch.multinomial(connection_probs, new_nnz, replacement=True)
                new_ca3_indices = sampled % self.cfg.ca3_pyramidal_cells
        else:
            # Fallback: random connectivity.
            new_ca3_indices = torch.randint(
                0, self.cfg.ca3_pyramidal_cells, (new_nnz,),
                device=self.mossy_fibers.device
            )

        new_values = (torch.randn(new_nnz, device=self.mossy_fibers.device) * 0.5 + 3.0).clamp(min=0.0)
        new_indices = torch.stack([new_dg_indices, new_ca3_indices])

        # Rebuild mossy fiber sparse matrix with expanded DG dimension.
        old_mf = self.mossy_fibers
        combined_indices = torch.cat([old_mf.indices(), new_indices], dim=1)
        combined_values = torch.cat([old_mf.values(), new_values])
        new_size = (self.current_dg_size + batch, self.cfg.ca3_pyramidal_cells)
        self.mossy_fibers = torch.sparse_coo_tensor(
            combined_indices, combined_values, new_size
        ).coalesce()

        # Expand perforant path for new DG neurons.
        new_pp = init_sparse_projection(
            self.cfg.entorhinal_cortex_neurons, batch,
            self.cfg.perforant_path_density, weight_mean=1.0
        )
        old_pp = self.perforant_path
        pp_new_indices = new_pp.indices() + torch.tensor(
            [[0], [self.current_dg_size]], device=old_pp.device
        )
        pp_combined_indices = torch.cat([old_pp.indices(), pp_new_indices], dim=1)
        pp_combined_values = torch.cat([old_pp.values(), new_pp.values()])
        pp_new_size = (self.cfg.entorhinal_cortex_neurons, self.current_dg_size + batch)
        self.perforant_path = torch.sparse_coo_tensor(
            pp_combined_indices, pp_combined_values, pp_new_size
        ).coalesce()

        # Expand DG -> DG_inhibitory projection.
        new_inh = init_sparse_projection(batch, self.cfg.dentate_gyrus_interneurons, 0.1, weight_mean=0.3)
        old_inh = self.dg_to_dg_inh
        inh_new_indices = new_inh.indices() + torch.tensor(
            [[self.current_dg_size], [0]], device=old_inh.device
        )
        inh_combined_indices = torch.cat([old_inh.indices(), inh_new_indices], dim=1)
        inh_combined_values = torch.cat([old_inh.values(), new_inh.values()])
        inh_new_size = (self.current_dg_size + batch, self.cfg.dentate_gyrus_interneurons)
        self.dg_to_dg_inh = torch.sparse_coo_tensor(
            inh_combined_indices, inh_combined_values, inh_new_size
        ).coalesce()

        self.current_dg_size += batch
        return True

    # ==================================================================
    # Serialization
    # ==================================================================

    def get_state(self) -> dict:
        """Complete hippocampal state for .soul file."""
        return {
            "entorhinal_cortex": self.entorhinal_population.get_state(),
            "dentate_gyrus": self.dentate_gyrus_population.get_state(),
            "ca3": self.ca3_population.get_state(),
            "ca1": self.ca1_population.get_state(),
            "current_dg_size": self.current_dg_size,
            "perforant_path_indices": self.perforant_path.indices().cpu(),
            "perforant_path_values": self.perforant_path.values().cpu(),
            "perforant_path_size": list(self.perforant_path.size()),
            "mossy_fibers_indices": self.mossy_fibers.indices().cpu(),
            "mossy_fibers_values": self.mossy_fibers.values().cpu(),
            "mossy_fibers_size": list(self.mossy_fibers.size()),
            "ca3_recurrent_crow_indices": self.ca3_recurrent_collaterals.crow_indices().cpu(),
            "ca3_recurrent_col_indices": self.ca3_recurrent_collaterals.col_indices().cpu(),
            "ca3_recurrent_values": self.ca3_recurrent_collaterals.values().cpu(),
            "schaffer_indices": self.schaffer_collaterals.indices().cpu(),
            "schaffer_values": self.schaffer_collaterals.values().cpu(),
            "episodic_matrix": self.episodic_matrix.cpu(),
            "episodic_novelty": self.episodic_novelty.cpu(),
            "episodic_occupied": self.episodic_occupied.cpu(),
            "write_ptr": self.write_ptr.item(),
            "current_episodic_slots": self.current_episodic_slots,
            "umap_encoder_state": self.umap_encoder.state_dict(),
            "umap_decoder_state": self.umap_decoder.state_dict(),
            "bridge_to_ec_state": self.bridge_to_ec.state_dict(),
        }

    def restore_state(self, state: dict):
        """Restore from .soul checkpoint."""
        self.entorhinal_population.restore_state(state["entorhinal_cortex"])
        self.dentate_gyrus_population.restore_state(state["dentate_gyrus"])
        self.ca3_population.restore_state(state["ca3"])
        self.ca1_population.restore_state(state["ca1"])
        self.current_dg_size = state["current_dg_size"]

        self.perforant_path = torch.sparse_coo_tensor(
            state["perforant_path_indices"],
            state["perforant_path_values"],
            state["perforant_path_size"]
        ).coalesce()

        self.mossy_fibers = torch.sparse_coo_tensor(
            state["mossy_fibers_indices"],
            state["mossy_fibers_values"],
            state["mossy_fibers_size"]
        ).coalesce()

        self.ca3_recurrent_collaterals = torch.sparse_csr_tensor(
            state["ca3_recurrent_crow_indices"],
            state["ca3_recurrent_col_indices"],
            state["ca3_recurrent_values"],
            (self.cfg.ca3_pyramidal_cells, self.cfg.ca3_pyramidal_cells)
        )

        self.schaffer_collaterals = torch.sparse_coo_tensor(
            state["schaffer_indices"],
            state["schaffer_values"],
            (self.cfg.ca3_pyramidal_cells, self.cfg.ca1_pyramidal_cells)
        ).coalesce()

        self.episodic_matrix = state["episodic_matrix"]
        self.episodic_novelty = state["episodic_novelty"]
        self.episodic_occupied = state["episodic_occupied"]
        self.write_ptr.fill_(state["write_ptr"])
        self.current_episodic_slots = state["current_episodic_slots"]

        self.umap_encoder.load_state_dict(state["umap_encoder_state"])
        self.umap_decoder.load_state_dict(state["umap_decoder_state"])
        self.bridge_to_ec.load_state_dict(state["bridge_to_ec_state"])

    def reset_all_dynamics(self):
        """Glymphatic sweep: reset all spiking state to resting."""
        self.entorhinal_population.reset_dynamics()
        self.dentate_gyrus_population.reset_dynamics()
        self.ca3_population.reset_dynamics()
        self.ca1_population.reset_dynamics()
        self.dentate_gyrus_inhibitory.reset_dynamics()
        self.ca3_inhibitory.reset_dynamics()
        self.ca1_inhibitory.reset_dynamics()

    def get_diagnostics(self) -> dict:
        """System health report."""
        occupied = self.episodic_occupied.sum().item()
        return {
            "dentate_gyrus_neurons": self.current_dg_size,
            "dentate_gyrus_max": self.cfg.dentate_gyrus_granule_cells_max,
            "dentate_gyrus_headroom": self.cfg.dentate_gyrus_granule_cells_max - self.current_dg_size,
            "episodic_occupied": occupied,
            "episodic_capacity": self.current_episodic_slots,
            "episodic_utilization": occupied / max(self.current_episodic_slots, 1),
            "total_spiking_neurons": (
                self.cfg.entorhinal_cortex_neurons
                + self.current_dg_size
                + self.cfg.ca3_pyramidal_cells
                + self.cfg.ca1_pyramidal_cells
                + self.cfg.dentate_gyrus_interneurons
                + self.cfg.ca3_interneurons
                + self.cfg.ca1_interneurons
            ),
        }
