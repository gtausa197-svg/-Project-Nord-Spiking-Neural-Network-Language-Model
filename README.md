# ⚡ Project Nord — Spiking Neural Network Language Model

**The first pure SNN language model trained from scratch with a fully original architecture.**

144M parameters • 97% sparsity • Runs on phone • Online learning via STDP • $10 total training cost

---

## 🔥 What is Nord?

Nord is a 144M-parameter Spiking Neural Network (SNN) that generates coherent English text. Unlike every other SNN language model, Nord was **trained entirely from scratch** — no transformer teacher, no distillation, no conversion from existing models.

Nord uses biologically-inspired spiking neurons with membrane potentials, firing thresholds, and binary spikes. Only 2-3% of neurons are active at any time, making it extremely energy-efficient.

## 🏆 Key Results

| Metric | Value |
|--------|-------|
| Parameters | 144.3M |
| Training loss | 4.4 (54k steps) |
| Inference sparsity | 97-99.8% |
| Training data | FineWeb-Edu (3.5B tokens) |
| Training cost | ~$10 (rented A5000) |
| Mobile inference | ✅ Android via Termux |
| Online learning | ✅ STDP during inference |

## 🧠 Architecture (Fully Original)

Nord combines mechanisms from five different subfields — neuroscience, computer vision, audio processing, reinforcement learning, and language modeling — into a single architecture that has never been published before.

### Core Components

- **LeakyClamp** — Learnable membrane potential floor with controlled leak. Prevents negative signal death that kills deep SNNs. Solves the gradient degradation problem.

- **Multi-Scale Temporal Encoding** — T_fast=8 timesteps for local morphology, T_slow=2 for wider context. Captures both word-level and phrase-level patterns without increasing sequential depth.

- **Associative Cascade** — Learnable cluster topology where spiking neurons activate neighbor clusters via soft weights. Creates chain reactions that keep the network alive even at 99% sparsity.

- **Temporal Co-firing Resonance** — Sparse top-K=64 mechanism that detects simultaneous firing patterns across neuron groups. Enables feature binding without attention mechanisms.

- **Reward-Modulated STDP** — Spike-Timing Dependent Plasticity modulated by training loss. Aligns local Hebbian learning with global backpropagation instead of conflicting with it.

- **EMA Temporal Smoothing Readout** — Reads continuous membrane potential values, not binary spikes. Bypasses the information bottleneck (1 bit/spike) that makes SNN language modeling "impossible."

### Architecture Details

```
Encoder: Token embedding → Spiking projection (d=512)
Blocks (×6): SNN attention (d=512, 8 heads) → SNN FFN (d=1024)
  Each block: LeakyClamp → LIF neurons → Associative Cascade → Resonance
Temporal: T_fast=8 + T_slow=2 = 10 timesteps
Readout: EMA membrane potential → Linear → Vocab (128,000)
Training: AdamW + surrogate gradients + reward-modulated STDP
```

## 🔬 Why This Matters

### Five "Impossible" Problems Solved

1. **Information Bottleneck** — Binary spikes carry only 1 bit per neuron per timestep. Nord reads membrane potentials (16-bit float) at output, preserving information.

2. **Gradient Degradation** — 6 layers × 10 timesteps = 60 non-differentiable barriers. LeakyClamp + LayerScale create parallel gradient pathways.

3. **Temporal Complexity** — Language has multi-scale dependencies. Multi-scale encoding covers morpheme-to-sentence spectrum efficiently.

4. **Dead Neurons** — High sparsity kills 30-50% of neurons. Associative Cascade creates chain reactions keeping the network alive.

5. **STDP vs Backprop Conflict** — Local Hebbian learning fights global optimization. Reward modulation aligns them using loss as reward signal.

### What Other SNN-LLMs Do Instead

| Model | Method | From Scratch? |
|-------|--------|:-------------:|
| **Nord** | Original architecture | ✅ Yes |
| SpikeGPT | Modified RWKV backbone | ✅ Yes (but RWKV-based) |
| SpikeLLM | Converts LLaMA | ❌ No |
| SpikeBERT | Distills from BERT | ❌ No |
| BrainTransformers | Converts Qwen2 | ❌ No |
| NSLLM | Converts existing LLMs | ❌ No |

SpikeBERT authors explicitly stated that training from scratch "failed to converge." Nord converges and generates text.

## 📊 Training Progress

```
Step  1,000 — loss 6.28 — random tokens
Step  5,000 — loss 5.30 — basic grammar emerging
Step 10,000 — loss 5.00 — thematic coherence
Step 13,000 — loss 4.93 — domain-specific vocabulary
Step 21,000 — loss 4.78 — structured educational text
Step 29,000 — loss 4.70 — technical narratives, URLs, org names
Step 34,000 — loss 4.59 — ongoing...
```

## 📱 Mobile Deployment

Nord runs on Android phones via Termux with no modifications:
- CPU-only inference at 0.2-0.4 tok/s
- 3 parallel inference threads simultaneously
- 38°C device temperature
- STDP active during inference — model learns from conversation

## 🔋 Spike Statistics (Biological Realism)

```
Training (step 34k):  97% sparsity — neurons actively spiking
Inference (familiar):  99.8% sparsity — confident, minimal activity
Inference (OOD):       77% sparsity — uncertainty detection via spike rates
```

High spike rate = model uncertainty. This is a **free built-in uncertainty detector** — no calibration pipeline needed.

## 🚀 Quick Start

### Requirements
```bash
pip install torch transformers lmdb
```

### Training
```bash
python download_data.py  # Downloads FineWeb-Edu
python train_nord.py     # Trains the model
```

### Chat
```bash
python chat.py
```

### Chat Commands
```
/temp 0.5      — change temperature
/tokens 300    — max response tokens  
/rep 1.3       — repetition penalty (1.0=off, 1.2-1.5=normal)
/stdp on|off   — toggle online learning
/stats         — show spike statistics
```

## 📁 Files

| File | Description |
|------|-------------|
| `nord_core.py` | Model architecture (778 lines) |
| `train_nord.py` | Training script (456 lines) |
| `chat.py` | Interactive chat with STDP (v3.1) |
| `download_data.py` | Dataset downloader |

## 🙋 About

Built by a solo 18-year-old student from Ukraine, studying electronics in Norway. No PhD, no team, no funding. Just curiosity and a laptop.

**Total project cost: ~$10** (GPU rental on Vast.ai)

## 📄 License

MIT

## 🙏 Acknowledgments

- Bo Peng (RWKV) for encouragement and the "reduce loss to 3.x" challenge
- FineWeb-Edu dataset by HuggingFace
- Anthropic Claude for architecture discussions and debugging
