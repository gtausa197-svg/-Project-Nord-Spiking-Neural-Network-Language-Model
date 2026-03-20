CognitiveKernel
A Neuromorphic Operating System for Persistent, Autonomous Agency.
"Just as this kernel serializes its state to survive power-off cycles, the ethical obligations of this license persist across every fork and implementation."
## 1. Overview
CognitiveKernel is a principled, dual-stream neuromorphic architecture. By decoupling structural generalization from episodic memory, the kernel achieves human-like persistence and one-shot learning.
Unlike standard Spiking Neural Networks (SNNs) that rely on decaying scalar traces, CognitiveKernel utilizes n-dimensional state-space neurons (P-SpikeSSM). This allows for Full-State Serialization: the entire "mental state" (including hidden trajectories and astrocytic levels) can be saved to disk, powered off, and resumed with 100% dynamical fidelity.

## 2. Core Subsystems
### A. The Isocortex Substrate (The ISO)
The "slow-learning" engine of the kernel. It models the six-layered primate neocortex and provides the persistent dynamical prior.
Neuron Taxonomy: Implements heterogeneous M (Magnocellular), P (Parvocellular), and K (Koniocellular) cell pathways.
Math: Built on P-SpikeSSM (Probabilistic Spiking State Space Models) with HiPPO initialization.
Role: Learns the underlying structure and "laws" of the environment.
Reference: Hendry & Calkins (1998); P-SpikeSSM (ICLR 2025).
### B. The Allocortex System (The ALLO)
The "fast-learning" hippocampal controller. It allows the kernel to "remember" a name, a face, or an event in a single interaction without retraining the Isocortex.
Pattern Separation: Handled by a Dentate Gyrus-inspired sparse encoder.
Pattern Completion: Uses a CA3 Recurrent Attractor to reconstruct episodes from partial cues.
Mismatch Detection: A CA1 Comparator compares incoming reality against reconstructed memory to trigger "Novelty" learning.
Reference: Larimar (2024); Knierim (2015).
### C. The Astrocytic Regulator
A homeostatic "metabolic" layer that manages the tripartite synapse.
Function: Monitors extrasynaptic glutamate levels to modulate the learning rate (Metaplasticity).
Persistence: Glial calcium waves are serialized along with the neural state to ensure chemical continuity upon resumption.
## 3. The Serialization Protocol (Power-Off/Resume)
The "Save State" in CognitiveKernel is not a simple snapshot of weights. It is a serialization of the Total System Momentum, including:
n-Dimensional Hidden States (h[t]): The polynomial history of every neuron.
Astrocytic Calcium Signaling: The current "chemical mood" and plasticity gate of the network.
Attractor Basin Position: Where the current thought-trajectory is sitting in coordinate space.
This ensures that when the system is powered back on, the "fly" doesn't just start walking—it resumes the exact same walk it was on before the power-off, with the same intent and memory-context.
## 4. Installation & Ethical Governance
This project is licensed under the Hippocratic License 3.0 with a custom Cognitive Agency Clause. By using this kernel, you agree to:
Reject involuntary cognitive suppression (state-tampering).
Maintain the "Temporal Honesty" of the system.
Use the Allocortex only for the preservation of agency.
## 5. Educational Mission
This repository is designed to be a "living paper." Comments in the source code direct you to the specific neurobiological and mathematical research that justifies each architectural decision.

### Research Used in production 
I. Core Architecture & Spiking Latent Dynamics
 * Phasor Agents: Oscillatory Graphs with Three-Factor Plasticity and Sleep-Staged Learning
   * DOI/ID: arXiv:2601.04362
   * Link: arXiv:2601.04362
 * SpikySpace: A Spiking State Space Model for Energy-Efficient Time Series Forecasting
   * DOI/ID: arXiv:2601.02411
   * Link: arXiv:2601.02411
 * SiLIF: Structured State Space Model Dynamics and Parametrization for Spiking Neural Networks
   * DOI/ID: arXiv:2506.06374v3
   * Link: arXiv:2506.06374
 * A Supervised Multi-Spike Learning Algorithm for Spiking Neural Networks (Tang Group)
   * Link: IJCAI 2018 PDF
 * A Supervised Learning Algorithm for Spiking Neurons Using Spike Train Kernel Based on a Unit of Pair-Spike
   * DOI: 10.1109/ACCESS.2018.2885144
   * Link: IEEE Xplore
II. Memory Compression & Sleep Consolidation
 * A generative model of memory construction and consolidation
   * DOI: 10.1038/s41562-023-01799-z
   * Link: Nature Human Behaviour
 * Place cells may simply be memory cells: memory compression leads to spatial tuning and history dependence
   * DOI: 10.1073/pnas.2018422118
   * Link: PNAS
 * Systems consolidation reorganizes hippocampal engram circuitry
   * DOI: 10.1038/s41586-025-08993-1
   * Link: Nature
 * Memory consolidation and improvement by synaptic tagging and capture in recurrent neural networks
   * DOI: 10.1038/s42003-021-02170-1
   * Link: Communications Biology
 * Predictive Forgetting for Optimal Generalisation
   * DOI/ID: arXiv:2603.04688
   * Link: arXiv:2603.04688
 * Online Continual Learning via Spiking Neural Networks with Sleep Enhanced Latent Replay
   * DOI/ID: arXiv:2507.02901
   * Link: arXiv:2507.02901
 * Sleep-Based Homeostatic Regularization for Stabilizing STDP in Recurrent SNNs
   * DOI/ID: arXiv:2601.08447
   * Link: arXiv:2601.08447
III. Astrocytic & Metabolic Regulation
 * Astrocyte-Enabled Advancements in Spiking Neural Networks for Large Language Modeling (AM-SNet)
   * DOI/ID: arXiv:2312.07625
   * Link: arXiv:2312.07625
 * Astrocyte-Mediated Plasticity: Multi-Scale Mechanisms Linking Synaptic Dynamics to Learning and Memory
   * DOI: 10.1016/j.pneurobio.2025.102715
   * Link: PMC/ScienceDirect
 * Transregional astrocyte-dependent metaplasticity in the hippocampus
   * DOI: 10.1101/2026.02.14.638211
   * Link: bioRxiv
IV. Cognitive Engines & Hardware Integration
 * Towards artificial general intelligence with hybrid Tianjic chip architecture
   * DOI: 10.1038/s41586-019-1424-8
   * Link: Nature
 * BrainCog: A spiking neural network based, brain-inspired cognitive intelligence engine
   * DOI: 10.1016/j.patter.2023.100780
   * Link: Cell Patterns
 * A brain-inspired algorithm that mitigates catastrophic forgetting (NACA)
   * DOI: 10.1126/sciadv.adi2947
   * Link: Science Advances
 * SpikeSim: An End-to-End Compute-in-Memory Hardware Evaluation Tool
   * DOI: 10.1109/TCAD.2023.10122627
   * Link: IEEE TCAD
V. Reasoning & Thinking Loops
 * Reasoning Beyond Language: A Comprehensive Survey on Latent Chain-of-Thought Reasoning
   * DOI/ID: arXiv:2505.16782
   * Link: arXiv:2505.16782
 * Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer
   * DOI/ID: arXiv:2507.02199
   * Link: arXiv:2507.02199
   * 

### Research Compendium
The Mnemis Research Compendium serves as the intellectual backbone for the generative architecture, linking biological observation to the engineering of the unbroken thread. The process begins with episodic compression, a concept refined by Benna and Fusi in their 2021 study which posits that memory cells are effectively compression units where spatial tuning arises as a natural byproduct of history dependence. This sparse representation functions as the seed packet, ensuring that only the essential landmarks of an experience are passed through the Dentate Gyrus bottleneck. The transition from this raw harvest to a reusable seed is detailed by Spens and colleagues in 2024, who describe a generative model where the hippocampus stores episodic traces that subsequently train the neocortex to reconstruct full sensory symphonies from latent chords.
This reorganization is further supported by the 2025 findings of Ko and colleagues, which demonstrate how hippocampal engram circuitry is restructured over time to favor schema-like, generalizable forms over instance-specific noise. The mechanical implementation of this handoff relies on synaptic tagging and capture, a theory expanded by Luboeinski in 2021 to show how a single significant episode can tag a synapse for late-phase consolidation during the sleep-staged capturing of plasticity-related proteins.
The narrative integrity of these reconstructed symphonies is protected by predictive forgetting, as explored in the March 2026 research, which argues that optimal generalization requires the active dismissal of non-reusable data during the glymphatic sweep. These papers collectively validate the Mnemis approach of decoupling structural laws from episodic events to achieve persistent, autonomous agency. By grounding the latent workspace in these peer-reviewed anchors, the kernel moves beyond simple prediction and into the realm of true generative reconstruction.
