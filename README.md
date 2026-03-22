<p align="center">
  <img src="assets/nord_banner.png" alt="Project Nord" width="800"/>
</p>

<h1 align="center">⚡ Project Nord v4.2</h1>

<p align="center">
  <b>Brain-Inspired Spiking Neural Network Language Model</b><br>
  <i>Spike-Driven MoE · Zonal Specialization · 93% Sparsity · 618M Parameters · Instruction-Tuned</i>
</p>

<p align="center">
  <a href="https://www.reddit.com/r/LocalLLaMA/"><img src="https://img.shields.io/badge/r%2FLocalLLaMA-51K%20views-orange?logo=reddit" alt="Reddit"/></a>
  <img src="https://img.shields.io/badge/Top%205%25-Poster-gold" alt="Top 5%"/>
  <img src="https://img.shields.io/badge/Parameters-618.8M-blue" alt="Params"/>
  <img src="https://img.shields.io/badge/Sparsity-93%25-green" alt="Sparsity"/>
  <img src="https://img.shields.io/badge/Loss-3.65-brightgreen" alt="Loss"/>
  <img src="https://img.shields.io/badge/License-Apache%202.0-red" alt="License"/>
</p>

---

## 🔥 What is Nord?

Nord is a **spiking neural network (SNN) language model** that processes text using biologically-inspired spike patterns instead of continuous activations. Unlike standard transformers where 100% of neurons are active for every token, Nord activates only **7-17% of neurons** at any time — with different brain-inspired zones specializing in different functions.

This is not a fine-tuned LLM. Nord is trained **from scratch** with a novel architecture that combines:

- **Leaky Integrate-and-Fire (LIF) neurons** with surrogate gradients
- **Spike-Driven Mixture of Experts (MoE)** routing
- **Brain-inspired zonal organization** (Sensory → Association → Memory → Executive)
- **Temporal spike coding** across multiple timesteps
- **Instruction tuning** (OpenHermes 2.5) — first SNN to attempt chat-style responses
- **87-93% average sparsity** during both training and inference

## 🏆 Key Results

| Metric | 140M (v4.2) | 618M (v4.2) |
|---|---|---|
| Parameters | 139.9M | **618.8M** |
| Training loss | 4.30 | **3.65** |
| Sparsity | 91% | **87-93%** |
| Architecture | d=512, 6 blocks | **d=1536, 10 blocks (3S+3A+4E)** |
| Training | FineWeb-Edu | **FineWeb-Edu + OpenHermes 2.5** |
| Inference speed | 7.3 tok/s | **6.8 tok/s (RTX 4090 Ti)** |
| Training cost | ~$15 | **~$260** |
| Instruction tuning | No | **Yes (first SNN with instruction tuning)** |

## 🧠 Why Spikes?

| | Transformer | Nord SNN |
|---|---|---|
| **Active params per token** | 100% | 7-17% |
| **Computation** | Dense matrix multiply | Sparse spike events |
| **Energy model** | GPU-optimized | Neuromorphic-compatible |
| **Biological similarity** | Low | High |

If SNN language models can match transformer quality at scale, they could run 86B-parameter models with the compute cost of a 3-4B model on neuromorphic hardware.

## 🏗️ Architecture (618M)

```
┌─────────────────────────────────────────────────────────────┐
│                    TEMPORAL SPIKE ENCODER                    │
│         Token → 8 fast + 2 slow timestep currents           │
│         d_model=1536, vocab=128K (Llama-3.2 tokenizer)      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐   Spike rates: 3-7%                       │
│  │  SENSORY    │   3 blocks, FFN + LIF (66.3M params)      │
│  │  ZONE       │   Feature extraction (quiet)               │
│  └──────┬──────┘                                            │
│  ┌──────▼──────┐   Spike rates: 4-12%                      │
│  │ ASSOCIATION │   3 blocks, Spike-Driven MoE (66.4M)       │
│  │    ZONE     │   4 experts, top-2 routing                 │
│  └──────┬──────┘                                            │
│  ┌──────▼──────┐   Memory neurons: 256                     │
│  │   MEMORY    │   τ=0.99, gated temporal attention (1.3M)  │
│  │   CORTEX    │   Multi-head readout, 8 read heads         │
│  └──────┬──────┘                                            │
│  ┌──────▼──────┐   Spike rates: 4-33%                      │
│  │ EXECUTIVE   │   4 blocks, FFN + LIF (88.4M params)      │
│  │    ZONE     │   Decision & output (loudest zone)         │
│  └──────┬──────┘                                            │
│  ┌──────▼──────┐                                            │
│  │  READOUT    │   EMA over membrane potential              │
│  │  + LM HEAD  │   → vocabulary logits (128K)               │
│  └─────────────┘                                            │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Breakdown (618M)

| Component | Parameters |
|-----------|-----------|
| Sensory Zone | 66.3M (3 blocks) |
| Association Zone | 66.4M (3 blocks, MoE) |
| Memory Cortex | 1.3M |
| Executive Zone | 88.4M (4 blocks) |
| Encoder + Readout + LM Head | ~396.4M |
| **Total** | **618.8M** |

## 📊 Emergent Zonal Specialization

Zonal specialization **survives and evolves at 4.4x scale**:

**140M zones:**
```
Sensory:      8-10%   (quiet)
Association:  10-14%  (moderate, MoE routing)
Memory:       0.5-1%  (very selective)
Executive:    11-26%  (loud, decision-making)
```

**618M zones:**
```
Sensory:      3-7%    (even quieter at scale)
Association:  4-12%   (MoE routing)
Memory:       39%     (dramatically more active)
Executive:    4-33%   (Executive[3]=33%, still loudest)
```

Key discovery: **Memory Cortex became 39x more active** at 618M. The model learned that persistent memory is more valuable at larger scale. Not programmed — emergent.

## 📈 Training Progression

### Phase 1: FineWeb-Edu (base language, 40GB)

| Step | Loss | Sparsity | Note |
|------|------|----------|------|
| 0 | 13.4 | 68% | Training start |
| 5,000 | 5.30 | 87% | Basic grammar |
| 10,000 | 5.00 | 91% | Thematic coherence |
| 22,000 | 4.90 | 86% | End of base training |

### Phase 2: OpenHermes 2.5 (instruction tuning, 1M examples)

| Step | Loss | Sparsity | Note |
|------|------|----------|------|
| 22,200 | 4.76 | 85% | Learning instruction format |
| 23,000 | 4.20 | 85% | Structure emerging |
| 25,000 | 3.89 | 86% | Topic relevance improving |
| 27,200 | **3.65** | **93%** | Structured responses |

## 💬 Before vs After Instruction Tuning

**Before OpenHermes (loss 4.9):**
```
You: What is water?
Nord: Water are non-renewable. Water has long been a subject 
to health care for centuries... Malt IVs' goals involve 
improving fertility at all levels...
```

**After OpenHermes (loss 3.65):**
```
You: What is a computer?
Nord: A computer science degree plays an important role in 
the development of software and system application. It will 
help to get rid of a recording process by creating computing 
elements... In this article we discuss how the ideal simplest, 
the more normal solution of the structure...
```

Still not ChatGPT — but the transformation is clear. First SNN to demonstrate instruction-following behavior.

## ⚡ Live Spike Visualization

```
┌──────────────────────────────────────────────────────┐
│ Neural Activity                                      │
├──────────────────────────────────────────────────────┤
│ ⚡ Sensory     ███······················   6.0% │
│ ⚡ Association █████····················   9.2% │
│ ⚡ Memory      ████████████████████████·  38.7% │
│ ⚡ Executive   ██████████···············  17.6% │
├──────────────────────────────────────────────────────┤
│ Sparsity: 83% silent  (17% neurons active per token) │
└──────────────────────────────────────────────────────┘
```

## 🔬 Comparison with Other SNN Language Models

| Feature | Nord | SpikeGPT | SpikeLLM | SpikingLLM |
|---|---|---|---|---|
| Trained from scratch | ✅ | ✅ (RWKV-based) | ❌ (converts LLaMA) | Uses KD teacher |
| Max params (from scratch) | **618M** | 216M | N/A | Paper withdrawn |
| Instruction tuning | **✅** | ❌ | ❌ | ❌ |
| Zonal specialization | **✅** | ❌ | ❌ | ❌ |
| Memory cortex | **✅** | ❌ | ❌ | ❌ |
| Spike-driven MoE | **✅** | ❌ | ❌ | ❌ |
| Multi-dataset training | **✅** | ❌ | ❌ | ❌ |
| Open source | ✅ | ✅ | Partial | No code |

## 🚀 Quick Start

```bash
pip install torch transformers lmdb numpy
```

### Training

```bash
# Base training
python train_nord_tpu_700m.py --dataset data.jsonl

# Instruction tuning (continued)
python train_nord_tpu_700m.py --dataset openhermes.jsonl --continued

# Scale to 1B
python train_nord_tpu_700m.py --dataset data.jsonl --preset 1b
```

### Chat

```bash
python chat.py
# Commands: /tokens, /temp, /rep, /live, /stats, /memory, /expert, /help
```

## 📁 Project Structure

```
nord-ai/
├── nord_core_700m.py          # Core architecture v4.2 (618M)
├── train_nord_tpu_700m.py     # Training script (CUDA + TPU support)
├── chat.py                    # Interactive chat with streaming + spike viz
├── build_lmdb.py              # Fast LMDB tokenizer
├── download_data.py           # Dataset downloader (8 datasets)
└── README.md
```

## 🗺️ What's Next

- **OpenWebMath** — arithmetic and reasoning
- **StarCoder** — code generation (SNN writing Python = first ever)
- **Scaling to 1B** — architecture ready via `--preset 1b`
- **NeurIPS 2026** — paper submission (deadline May 2026)
- **Neuromorphic deployment** — Intel Loihi / BrainChip Akida

## 🤝 Community & Support

Nord is fully open-source, built with zero institutional funding. Total cost: **$260** out of pocket.

- **Discord**: [Join our server](https://discord.gg/cFghG89P) — live updates, architecture discussion
- **Website**: https://www.nord-ai.net
- **Wiki**: https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model/wiki

Every contribution goes directly to GPU rental for scaling to 1B.

## 📖 Citation

```bibtex
@software{nord2026,
  title={Project Nord: Brain-Inspired Spiking Neural Network Language Model},
  author={Makarenko, Volodymyr},
  year={2026},
  url={https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model}
}
```

## License

Apache 2.0

## Acknowledgments

- FineWeb-Edu & OpenHermes 2.5 by HuggingFace / Teknium
- Bo Peng (RWKV) for encouragement
- Prof. Nikola Kasabov (KEDRI/AUT) for feedback
- Visual presentation by [mnbnkr](https://mnbnkr.github.io/-Project-Nord-Spiking-Neural-Network-Language-Model/)
- HuggingFace: https://huggingface.co/zerdovzad/Nord-AI
- By me cooffe https://buymeacoffee.com/zerdovzad?new=1

---

<p align="center">
  <i>⚡ "Only 7-17% of neurons fire at any time — just like a real brain."</i><br>
  <b>Built solo. 18 years old. Ukraine → Norway. $260 total.</b>
</p>
