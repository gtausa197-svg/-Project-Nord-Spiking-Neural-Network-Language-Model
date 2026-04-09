"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PROJECT NORD — Core Engine v5.0 Genesis Autogenic (700M)         ║
║          Spiking Neural Network LLM with Brain-Inspired Architecture       ║
║                                                                            ║
║  Genesis Curriculum (implementable core, offline):                         ║
║    • Identity: I_t = f(I_{t-1}, M_t)  →  GRU over sequence on M_t = mean_t(x)║
║    • Dual / multi memory → Triple purpose banks (structural / personal / aux)║
║    • Purpose-filtered fusion → softmax router w(s,p,a), entropy regularize ║
║    • Recursive Archive Grid (RAG) → learned K slots + attention read       ║
║  Enable: NordConfig.genesis_autogenic_v5=True (new weights; memory_size=256)║
║                                                                            ║
║  v4.1 CRITICAL FIXES (from code review):                                   ║
║    FIX A: Vectorized MoE dispatch — no Python loops over experts           ║
║    FIX B: Temporal attention memory — multi-head read over ALL timesteps   ║
║    FIX C: Differentiable spike loss — proper gradient flow                 ║
║    FIX D: LIF stability — clamped tau/threshold, warmup freeze             ║
║    FIX E: Temporal mixing in attention (no naive T*Dh flattening)         ║
║    FIX F: STDP isolation — only executive zone, bounded magnitude          ║
║    FIX G: MoE load balancing loss — prevents expert collapse               ║
║    FIX H: Gradient checkpointing support — VRAM control                    ║
║    FIX I: Fused LIF operations — reduced kernel launch overhead            ║
║    FIX J: Realistic training estimates in docs                             ║
║                                                                            ║
║  v4.2 FIXES (from 13K step training analysis):                             ║
║    FIX K: Block outputs spike-only — clamp negative before spike_ts        ║
║    FIX L: Stronger spike regulator — adaptive weight, per-layer targeting  ║
║    FIX M: Executive clamp floor=0 — prevent negative spike propagation     ║
║  §7d Paper stack (optional, Genesis v5): Liu V1–MT DoG + streams; PopSAN    ║
║      bottleneck; Susi/FNS LIFL-style input gain — see PaperCorticalStackV5 ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Union, cast

# ═══════════════════════════════════════════════════════════════════════════════
# §0  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class NordConfig:
    tokenizer_id:str="meta-llama/Llama-3.2-1B"
    # ── 700M Architecture ──
    vocab_size:int=128_256; d_model:int=1536; n_heads:int=24; n_layers:int=10
    d_ff:int=4096; max_seq_len:int=512
    T:int=8; T_slow:int=2; persistent_mem:bool=True
    # LIF — FIX D: constrained ranges
    tau_mem:float=0.9; tau_mem_min:float=0.8; tau_mem_max:float=0.98
    tau_syn:float=0.50; v_threshold:float=0.12
    v_thresh_min:float=0.05; v_thresh_max:float=0.5
    v_reset:float=-0.1; refractory_t:int=2; threshold_lr:float=0.01
    lif_freeze_steps:int=1000
    n_clusters:int=128; cascade_radius:int=3; cascade_gain:float=0.8
    # STDP — FIX F: bounded
    stdp_a_plus:float=0.005; stdp_a_minus:float=0.005
    stdp_tau_plus:float=20.0; stdp_tau_minus:float=20.0
    stdp_w_max:float=0.5; stdp_w_min:float=-0.15
    stdp_reward_scale:float=1.0; stdp_layers:Optional[List[str]]=None
    resonance_top_k:int=64; clamp_floor:float=-0.1; surrogate_alpha:float=4.0
    rope_theta:float=10000.0
    # Use PyTorch scaled_dot_product_attention (Flash/mem-efficient when available); False = explicit matmul + top-k
    use_sdpa_attention:bool=True
    # MoE — FIX A+G
    n_experts:int=4; top_k_experts:int=2; moe_capacity_factor:float=1.25
    moe_load_balance_weight:float=0.01; moe_route_temperature:float=1.0
    # Spike loss — FIX L
    target_spike_rate:float=0.03; spike_loss_weight:float=0.5
    # Zones: 3 sensory + 3 association(MoE) + 4 executive = 10
    sensory_layers:int=3; association_layers:int=3; executive_layers:int=4
    # Memory — FIX B
    memory_tau_mem:float=0.99; memory_size:int=256
    memory_gate_threshold:float=0.3; memory_n_read_heads:int=8
    # Genesis curriculum — core dual track (structural vs personal LIF banks). New weights; not loadable into single-bank ckpt.
    genesis_dual_memory:bool=False
    genesis_balance_loss_weight:float=0.02
    # NORD 5.0 — full autogenic stack (supersedes genesis_dual_memory; mutually exclusive in __post_init__)
    genesis_autogenic_v5:bool=False
    genesis_archive_slots:int=32
    genesis_purpose_entropy_weight:float=0.02
    genesis_identity_reg_weight:float=0.001
    genesis_archive_entropy_weight:float=0.01
    # NORD 5.0+ paper-inspired cortical augmentation (single stack; requires genesis_autogenic_v5):
    #   • Liu et al. IEEE TNNLS — V1/MT-style multistream mixing + center-surround (DoG) along sequence
    #   • Tang et al. CoRL (PopSAN) — Gaussian population coding on a low-d bottleneck
    #   • Susi et al. Sci Rep (FNS/LIFL) — input-magnitude gain as soft latency/threshold interaction
    paper_cortical_stack_v5:bool=False
    paper_surround_radius:int=2
    paper_pop_bottleneck:int=32
    paper_pop_per_dim:int=8
    paper_latency_bias_strength:float=0.12
    # Readout balancer: fuses each position with sequence-mean gist before LM head (train with instruct/chat mix).
    conversational_balancer:bool=False
    conversational_balancer_hidden:int=0  # 0 → min(d_ff, max(d_model, 1024))
    conversational_balancer_max_scale:float=1.0
    # Bistable context / continual-learning hooks (optional; new checkpoint shapes when enabled)
    context_lif_enabled:bool=False
    context_lif_gate:float=0.1
    context_inhibitory_neurons:int=0
    # STDP neuromodulation & protection (call set_output_entropy from trainer / runtime when using STDP)
    stdp_neuromodulation:bool=True
    stdp_entropy_high_threshold:float=2.5
    stdp_read_only:bool=False
    stdp_skeleton_scale:float=0.0
    stdp_metaplastic:bool=False
    stdp_metaplastic_gamma:float=0.99
    stdp_metaplastic_rigidity_max:float=10.0
    # Cortical motifs (optional)
    dendritic_input_enabled:bool=False
    dendritic_n_branches:int=4
    predictive_coding_enabled:bool=False
    predictive_coding_residual:float=0.35
    cls_fast_slow_enabled:bool=False
    cls_hippo_frac:int=4
    oscillatory_gating_enabled:bool=False
    # FIX H
    gradient_checkpointing:bool=False
    # Training
    batch_size:int=1; grad_accum:int=32; lr:float=2e-4; min_lr:float=1e-5
    weight_decay:float=0.01; warmup_steps:int=1000; max_steps:int=50_000
    save_every:int=1000; log_every:int=10; max_grad_norm:float=1.0
    dtype:torch.dtype=torch.float16; device:str="cuda"
    scale_preset_used:str="700m"
    @property
    def T_total(self)->int: return self.T+self.T_slow
    @property
    def n_layers_total(self)->int: return self.sensory_layers+self.association_layers+self.executive_layers
    def __post_init__(self):
        if self.stdp_layers is None:
            self.stdp_layers=[f"executive_{i}" for i in range(self.executive_layers)]
        if self.genesis_autogenic_v5 and self.genesis_dual_memory:
            raise ValueError("Use genesis_autogenic_v5 alone (includes triple purpose memory).")
        if self.genesis_autogenic_v5:
            assert self.memory_size==256,(
                "genesis_autogenic_v5: use memory_size=256 (banks 96+96+64 for heads divisible by 8).")
            assert 96%self.memory_n_read_heads==0 and 64%self.memory_n_read_heads==0
        if self.genesis_dual_memory:
            half=self.memory_size//2
            assert self.memory_size%2==0 and half>0 and half%self.memory_n_read_heads==0,(
                "genesis_dual_memory: memory_size must be even and memory_size/2 divisible by memory_n_read_heads")
        if self.paper_cortical_stack_v5 and not self.genesis_autogenic_v5:
            raise ValueError("paper_cortical_stack_v5 requires genesis_autogenic_v5=True")


# ~1B-parameter scale (SNN + 8-way MoE). Apply via apply_nord_scale_preset(cfg, "1b").
_PRESET_SCALE_1B: Dict[str, Union[int, float]] = {
    "d_model": 2048,
    "n_heads": 32,
    "d_ff": 7168,
    "sensory_layers": 4,
    "association_layers": 4,
    "executive_layers": 4,
    "n_experts": 8,
    "top_k_experts": 2,
    "n_clusters": 128,
}

# ~1.1B-class training profile: same width/depth/MoE as 1b, shorter context + LR schedule for VRAM/stability.
_PRESET_TRAIN_11B: Dict[str, Union[int, float, bool]] = {
    "max_seq_len": 512,
    "gradient_checkpointing": False,
    "genesis_autogenic_v5": True,
    "memory_size": 256,
    "use_sdpa_attention": True,
    "batch_size": 1,
    "grad_accum": 64,
    "lr": 1.0e-04,
    "warmup_steps": 2000,
    "max_steps": 100_000,
}


def apply_nord_scale_preset(cfg: NordConfig, preset: str) -> None:
    """Mutates cfg: '700m', '1b' (~1.09e9 params), or '1.1b' (1b arch + 384 ctx + Genesis v5 train defaults)."""
    name = (preset or "700m").strip().lower()
    if name in ("700m", "618m", "default", "", "m700", "0.7b", "small"):
        cfg.scale_preset_used = "700m"
        return
    if name in ("1.1b", "1.1_b", "11b", "1100m", "1.10b"):
        for k, v in _PRESET_SCALE_1B.items():
            setattr(cfg, k, cast(Union[int, float], v))
        cfg.stdp_layers = [f"executive_{i}" for i in range(cfg.executive_layers)]
        for k, v in _PRESET_TRAIN_11B.items():
            setattr(cfg, k, v)
        cfg.scale_preset_used = "1.1b"
        return
    if name in ("1b", "1.0b", "1000m", "b1", "1_0b", "large"):
        for k, v in _PRESET_SCALE_1B.items():
            setattr(cfg, k, cast(Union[int, float], v))
        cfg.stdp_layers = [f"executive_{i}" for i in range(cfg.executive_layers)]
        cfg.scale_preset_used = "1b"
        return
    raise ValueError(f"Unknown Nord scale preset {preset!r} — use '700m', '1b', or '1.1b'")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  SURROGATE GRADIENT
# ═══════════════════════════════════════════════════════════════════════════════
class ATanSurrogate(torch.autograd.Function):
    alpha=2.0
    @staticmethod
    def forward(ctx,membrane:Tensor,threshold:Tensor)->Tensor:
        ctx.save_for_backward(membrane,threshold)
        return(membrane>=threshold).to(membrane.dtype)
    @staticmethod
    def backward(ctx,grad_output:Tensor)->Tuple[Tensor,Tensor]:
        membrane,threshold=ctx.saved_tensors
        x=(membrane.float()-threshold.float())
        grad=ATanSurrogate.alpha/(2.0*math.pi*(1.0+(ATanSurrogate.alpha*x)**2))
        grad_v=(grad_output.float()*grad).to(membrane.dtype)
        return grad_v,-grad_v

def spike_fn(v:Tensor,th:Tensor,alpha:float=2.0)->Tensor:
    ATanSurrogate.alpha=alpha; return ATanSurrogate.apply(v,th)

# ═══════════════════════════════════════════════════════════════════════════════
# §2  ASSOCIATIVE LIF — FIX D: Stability + FIX I: Fused ops
# ═══════════════════════════════════════════════════════════════════════════════
class AssociativeLIF(nn.Module):
    def __init__(self,d:int,cfg:NordConfig,persistent:bool=False,
                tau_mem_override:Optional[float]=None):
        super().__init__()
        self.cfg=cfg; self.d=d; self.persistent=persistent
        self.threshold_raw=nn.Parameter(torch.full((d,),cfg.v_threshold))
        tau_mem=tau_mem_override if tau_mem_override is not None else cfg.tau_mem
        self.beta_mem_raw=nn.Parameter(torch.tensor(math.log(tau_mem/(1-tau_mem+1e-6))))
        self.beta_syn_raw=nn.Parameter(torch.tensor(math.log(cfg.tau_syn/(1-cfg.tau_syn+1e-6))))
        nc=cfg.n_clusters
        self.register_buffer("cluster_ids",torch.arange(d)%nc)
        r=cfg.cascade_radius; idx=torch.arange(nc)
        iw=torch.zeros(nc,nc)
        for offset in range(-r,r+1):
            if offset!=0: iw[idx,(idx+offset)%nc]=1.0-abs(offset)/(r+1)
        self.neighbor_weights=nn.Parameter(iw)
        self.cluster_gain=nn.Parameter(torch.full((nc,),cfg.cascade_gain))
        if persistent:
            self.register_buffer("_v_mem_state",torch.zeros(1,d))
            self.register_buffer("_i_syn_state",torch.zeros(1,d))
        self.register_buffer("_firing_rate_ema",torch.full((d,),cfg.target_spike_rate))
        self.register_buffer("_step_counter",torch.tensor(0,dtype=torch.long))

    @property
    def threshold(self)->Tensor:
        return self.threshold_raw.clamp(self.cfg.v_thresh_min,self.cfg.v_thresh_max)
    @property
    def beta_mem(self)->Tensor:
        return torch.sigmoid(self.beta_mem_raw).clamp(self.cfg.tau_mem_min,self.cfg.tau_mem_max)
    @property
    def beta_syn(self)->Tensor: return torch.sigmoid(self.beta_syn_raw)

    def _cascade_amplify(self,spikes:Tensor)->Tensor:
        B,D=spikes.shape; nc=self.cfg.n_clusters
        cid=self.cluster_ids.unsqueeze(0).expand(B,-1)
        cf=torch.zeros(B,nc,device=spikes.device,dtype=spikes.dtype)
        cf.scatter_add_(1,cid,spikes); cf=cf/max(D//nc,1)
        W=torch.sigmoid(self.neighbor_weights)
        ns=(W.to(cf.dtype)@cf.T).T*self.cluster_gain.to(cf.dtype).unsqueeze(0)
        return ns.gather(1,cid)

    def reset_state(self):
        if self.persistent: self._v_mem_state.zero_(); self._i_syn_state.zero_()

    def forward(self,current_in:Tensor)->Tuple[Tensor,Tensor]:
        T,B,D=current_in.shape; device=current_in.device; dtype=current_in.dtype
        bm=self.beta_mem; bs=self.beta_syn; thresh=self.threshold
        if self.persistent and self._v_mem_state.shape[0]==B:
            v_mem=self._v_mem_state.clone(); i_syn=self._i_syn_state.clone()
        else:
            v_mem=torch.zeros(B,D,device=device,dtype=dtype)
            i_syn=torch.zeros(B,D,device=device,dtype=dtype)
            if self.persistent:
                self._v_mem_state=torch.zeros(B,D,device=device,dtype=dtype)
                self._i_syn_state=torch.zeros(B,D,device=device,dtype=dtype)
        refrac=torch.zeros(B,D,device=device,dtype=torch.int32)
        spikes_out=[]; v_trace=[]
        refractory_val=torch.full_like(v_mem,self.cfg.v_reset)
        ref_t=self.cfg.refractory_t; alpha=self.cfg.surrogate_alpha
        for t in range(T):
            i_syn=bs*i_syn+current_in[t]
            rmask=(refrac>0)
            new_v=bm*v_mem+(1.0-bm)*i_syn
            v_mem=torch.where(rmask,refractory_val,new_v)
            s=spike_fn(v_mem,thresh,alpha)
            # Vectorized: cascade is zero when s=0; no Python branch (was if s.sum()>0)
            i_syn=i_syn+self._cascade_amplify(s)
            v_mem=v_mem-s*thresh.detach()
            refrac=torch.where(s.bool(),torch.full_like(refrac,ref_t),(refrac-1).clamp(min=0))
            spikes_out.append(s); v_trace.append(v_mem)
        if self.persistent:
            self._v_mem_state=v_mem.detach(); self._i_syn_state=i_syn.detach()
        ss=torch.stack(spikes_out)
        with torch.no_grad():
            self._firing_rate_ema.lerp_(ss.mean(dim=(0,1)),0.01)
            self._step_counter+=1
        return ss,torch.stack(v_trace)

# ═══════════════════════════════════════════════════════════════════════════════
# §3  TEMPORAL ENCODER
# ═══════════════════════════════════════════════════════════════════════════════
class TemporalSpikeEncoder(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.cfg=cfg; D=cfg.d_model
        self.embed=nn.Embedding(cfg.vocab_size,D)
        nn.init.kaiming_uniform_(self.embed.weight,a=math.sqrt(5))
        self.temporal_proj=nn.Linear(D,D,bias=False)
        self.drive_scale=nn.Parameter(torch.tensor(25.0))
        self.fast_basis=nn.Parameter(torch.randn(cfg.T,D)*0.02)
        self.slow_basis=nn.Parameter(torch.randn(cfg.T_slow,D)*0.02)
        self.slow_scale=nn.Parameter(torch.tensor(8.0))
    def forward(self,token_ids:Tensor)->Tensor:
        B,S=token_ids.shape; D=self.cfg.d_model
        x=self.temporal_proj(self.embed(token_ids)).reshape(B*S,D)
        fast=torch.sigmoid(self.fast_basis).unsqueeze(1)*x.unsqueeze(0)*self.drive_scale
        slow=torch.sigmoid(self.slow_basis).unsqueeze(1)*x.unsqueeze(0)*self.slow_scale
        return torch.cat([fast,slow],dim=0)

# ═══════════════════════════════════════════════════════════════════════════════
# §4  RoPE
# ═══════════════════════════════════════════════════════════════════════════════
class RotaryPositionEmbedding(nn.Module):
    def __init__(self,dim:int,max_seq_len:int=2048,theta:float=10000.0):
        super().__init__()
        inv_freq=1.0/(theta**(torch.arange(0,dim,2).float()/dim))
        self.register_buffer("inv_freq",inv_freq)
        t=torch.arange(max_seq_len).float(); freqs=torch.outer(t,inv_freq)
        self.register_buffer("cos_cached",freqs.cos())
        self.register_buffer("sin_cached",freqs.sin())
    def forward(self,x:Tensor,seq_len:int)->Tuple[Tensor,Tensor]:
        return self.cos_cached[:seq_len].to(x.dtype),self.sin_cached[:seq_len].to(x.dtype)

def apply_rope(x:Tensor,cos:Tensor,sin:Tensor)->Tensor:
    d=cos.shape[-1]; x1=x[...,:d]; x2=x[...,d:2*d]
    c=cos.unsqueeze(0).unsqueeze(0); s=sin.unsqueeze(0).unsqueeze(0)
    rot=torch.cat([x1*c-x2*s,x1*s+x2*c],dim=-1)
    return torch.cat([rot,x[...,2*d:]],dim=-1) if x.shape[-1]>2*d else rot

# ═══════════════════════════════════════════════════════════════════════════════
# §5  SYNAPTIC RESONANCE — FIX E: Temporal mixing (not flattening)
# ═══════════════════════════════════════════════════════════════════════════════
class SpikingSynapticResonance(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.cfg=cfg
        self.n_heads=cfg.n_heads; self.d_head=cfg.d_model//cfg.n_heads
        self.top_k=cfg.resonance_top_k; D=cfg.d_model; T_t=cfg.T_total
        self.W_q=nn.Linear(D,D,bias=False); self.W_k=nn.Linear(D,D,bias=False)
        self.W_v=nn.Linear(D,D,bias=False); self.W_o=nn.Linear(D,D,bias=False)
        self.lif_q=AssociativeLIF(D,cfg); self.lif_k=AssociativeLIF(D,cfg)
        self.resonance_temp=nn.Parameter(torch.tensor(1.0/math.sqrt(self.d_head)))
        # FIX E: Learned temporal mixing weights (not concatenation)
        self.temporal_mix_q=nn.Parameter(torch.ones(T_t)/T_t)
        self.temporal_mix_k=nn.Parameter(torch.ones(T_t)/T_t)
        self.rope=RotaryPositionEmbedding(self.d_head,cfg.max_seq_len,cfg.rope_theta)

    def forward(self,x_spikes:Tensor)->Tensor:
        T_t,B,S,D=x_spikes.shape; H=self.n_heads; Dh=self.d_head
        xf=x_spikes.reshape(T_t*B*S,D)
        qc=self.W_q(xf).reshape(T_t,B*S,D)
        kc=self.W_k(xf).reshape(T_t,B*S,D)
        vr=self.W_v(xf).reshape(T_t,B,S,D)
        qs,_=self.lif_q(qc); ks,_=self.lif_k(kc)
        qs=qs.reshape(T_t,B,S,H,Dh); ks=ks.reshape(T_t,B,S,H,Dh)
        # FIX E: Weighted sum over time, preserves spike timing semantics
        twq=F.softmax(self.temporal_mix_q,dim=0).reshape(T_t,1,1,1,1)
        twk=F.softmax(self.temporal_mix_k,dim=0).reshape(T_t,1,1,1,1)
        qm=(qs*twq).sum(0).permute(0,2,1,3) # (B,H,S,Dh)
        km=(ks*twk).sum(0).permute(0,2,1,3)
        cos,sin=self.rope(qm,S)
        qm=apply_rope(qm,cos,sin); km=apply_rope(km,cos,sin)
        vm=vr.mean(dim=0).reshape(B,S,H,Dh).permute(0,2,1,3)
        if self.cfg.use_sdpa_attention:
            # Match old logits (qm·km^T)*resonance_temp with SDPA's (q·k^T)/sqrt(dh): scale q only.
            rt=self.resonance_temp.to(qm.dtype).view(1,1,1,1)
            q_s=qm*(rt*math.sqrt(float(self.d_head)))
            ctx=F.scaled_dot_product_attention(
                q_s, km, vm, attn_mask=None, dropout_p=0.0, is_causal=True,
            )
        else:
            res=torch.matmul(qm,km.transpose(-2,-1))*self.resonance_temp
            cmask=torch.triu(torch.ones(S,S,device=x_spikes.device,dtype=torch.bool),diagonal=1)
            res.masked_fill_(cmask.unsqueeze(0).unsqueeze(0),float("-inf"))
            K=min(self.top_k,S)
            if K<S:
                tv,ti=torch.topk(res,K,dim=-1)
                sr=torch.full_like(res,float("-inf")); sr.scatter_(-1,ti,tv); res=sr
            attn=F.softmax(res.float(),dim=-1).to(res.dtype)
            ctx=torch.matmul(attn,vm)
        ctx=ctx.permute(0,2,1,3).reshape(B,S,D)
        return self.W_o(ctx).unsqueeze(0).expand(T_t,-1,-1,-1)

# ═══════════════════════════════════════════════════════════════════════════════
# §6  SPIKE-DRIVEN MoE — FIX A: Vectorized + FIX G: Load Balance
# ═══════════════════════════════════════════════════════════════════════════════
class SpikingExpertGroup(nn.Module):
    """FIX A: Memory-efficient expert dispatch using per-expert Linear + masking.
    Instead of bmm with (N,ef,D) tensors, we loop over experts (not tokens).
    With 4 experts this is 4 iterations — much better than 2048-token bmm."""
    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.n_experts=cfg.n_experts; self.expert_ff=cfg.d_ff//cfg.n_experts
        D=cfg.d_model; ef=self.expert_ff
        # Standard Linear layers per expert — memory efficient
        self.up=nn.ModuleList([nn.Linear(D,ef,bias=False) for _ in range(cfg.n_experts)])
        self.down=nn.ModuleList([nn.Linear(ef,D,bias=False) for _ in range(cfg.n_experts)])
        self.lif1=AssociativeLIF(ef,cfg); self.lif2=AssociativeLIF(D,cfg)

    def forward(self,x:Tensor,expert_indices:Tensor,expert_weights:Tensor)->Tensor:
        """x:(T,N,D), expert_indices:(N,top_k), expert_weights:(N,top_k)"""
        T,N,D=x.shape; top_k=expert_indices.shape[1]
        output=torch.zeros_like(x)
        # Loop over experts (4 iterations), not tokens (2048)
        for e in range(self.n_experts):
            # Find which tokens use this expert and with what weight
            mask=torch.zeros(N,device=x.device,dtype=x.dtype)
            for k in range(top_k):
                is_e=(expert_indices[:,k]==e).to(x.dtype)
                mask=mask+is_e*expert_weights[:,k]
            if mask.sum()==0: continue
            # Which tokens actually route here
            active=(mask>0)
            if not active.any(): continue
            # Extract active tokens across all timesteps
            active_x=x[:,active,:] # (T, n_active, D)
            Ta,Na,Da=active_x.shape
            # Up projection + LIF
            h=self.up[e](active_x.reshape(Ta*Na,Da)).reshape(Ta,Na,-1)
            h,_=self.lif1(h)
            # Down projection + LIF
            o=self.down[e](h.reshape(Ta*Na,-1)).reshape(Ta,Na,Da)
            o,_=self.lif2(o)
            # Weighted scatter back
            w=mask[active].unsqueeze(0).unsqueeze(-1) # (1,n_active,1)
            output[:,active,:]+=o*w
        return output

class SpikeDrivenMoE(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.cfg=cfg
        self.n_experts=cfg.n_experts; self.top_k=cfg.top_k_experts
        self.clusters_per_expert=cfg.n_clusters//cfg.n_experts
        self.expert_group=SpikingExpertGroup(cfg)
        self.route_lif=AssociativeLIF(cfg.d_model,cfg)
        self.expert_bias=nn.Parameter(torch.zeros(cfg.n_experts))
        self.register_buffer("expert_counts_ema",torch.ones(cfg.n_experts)/cfg.n_experts)

    def _compute_expert_scores(self,spikes:Tensor)->Tensor:
        fr=spikes.mean(dim=0); N,D=fr.shape; nc=self.cfg.n_clusters
        cid=torch.arange(D,device=fr.device)%nc
        cr=torch.zeros(N,nc,device=fr.device,dtype=fr.dtype)
        cr.scatter_add_(1,cid.unsqueeze(0).expand(N,-1),fr)
        cr=cr/max(D//nc,1)
        es=cr.reshape(N,self.n_experts,self.clusters_per_expert).mean(dim=-1)
        es=es/max(self.cfg.moe_route_temperature,0.01)
        return es+self.expert_bias.to(es.dtype)

    def _load_balance_loss(self,scores:Tensor,top_idx:Tensor)->Tensor:
        N=scores.shape[0]
        ef=torch.zeros(self.n_experts,device=scores.device)
        for e in range(self.n_experts):
            ef[e]=(top_idx==e).float().sum()/(N*self.top_k)
        rp=F.softmax(scores,dim=-1).mean(dim=0)
        loss=self.n_experts*(ef*rp).sum()
        with torch.no_grad(): self.expert_counts_ema.lerp_(ef,0.01)
        return loss

    def forward(self,x:Tensor)->Tuple[Tensor,Dict]:
        T,B,S,D=x.shape; N=B*S
        xf=x.reshape(T,N,D); rs,_=self.route_lif(xf)
        es=self._compute_expert_scores(rs)
        ts,ti=torch.topk(es,self.top_k,dim=-1)
        tw=F.softmax(ts.float(),dim=-1).to(x.dtype)
        output=self.expert_group(xf,ti,tw).reshape(T,B,S,D)
        lb=self._load_balance_loss(es,ti)
        stats={"moe_route_entropy":-(F.softmax(es,dim=-1)*F.log_softmax(es+1e-8,dim=-1)).sum(-1).mean().item(),
            "moe_load_balance_loss":lb}
        with torch.no_grad():
            for e in range(self.n_experts): stats[f"expert_{e}_load"]=self.expert_counts_ema[e].item()
        return output,stats

# ═══════════════════════════════════════════════════════════════════════════════
# §7  MEMORY CORTEX — FIX B: Temporal attention readout
# ═══════════════════════════════════════════════════════════════════════════════
class MemoryCortex(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.cfg=cfg; D=cfg.d_model; M=cfg.memory_size
        self.to_memory=nn.Linear(D,M,bias=False)
        self.from_memory=nn.Linear(M,D,bias=False)
        self.memory_lif=AssociativeLIF(M,cfg,persistent=True,tau_mem_override=cfg.memory_tau_mem)
        self.gate_lif=AssociativeLIF(M,cfg)
        self.gate_proj=nn.Linear(D,M,bias=False)
        self.gate_threshold=nn.Parameter(torch.tensor(cfg.memory_gate_threshold))
        # FIX B: Multi-head temporal attention for memory readout
        H=cfg.memory_n_read_heads; hd=M//H
        self.n_read_heads=H
        self.read_query=nn.Parameter(torch.randn(H,hd)*0.02)
        self.read_key_proj=nn.Linear(M,M,bias=False)
        self.read_scale=1.0/math.sqrt(hd)
        self.mem_norm=nn.LayerNorm(D)
        self.memory_mix=nn.Parameter(torch.tensor(0.1))

    def reset_state(self): self.memory_lif.reset_state()

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        T,B,S,D=x.shape; M=self.cfg.memory_size; N=B*S; H=self.n_read_heads; hd=M//H
        xf=x.reshape(T,N,D)
        mi=self.to_memory(xf.reshape(T*N,D)).reshape(T,N,M)
        ms,mv=self.memory_lif(mi)
        gi=self.gate_proj(xf.reshape(T*N,D)).reshape(T,N,M)
        gs,_=self.gate_lif(gi)
        gate_sig=gs.mean(dim=0)
        gate_mask=torch.sigmoid((gate_sig-self.gate_threshold)*10.0)
        # FIX B: Temporal attention over ALL timesteps
        mvh=mv.reshape(T,N,H,hd)
        mk=self.read_key_proj(mv.reshape(T*N,M)).reshape(T,N,H,hd)
        q=self.read_query.unsqueeze(0).unsqueeze(0) # (1,1,H,hd)
        attn_s=(q*mk).sum(-1)*self.read_scale # (T,N,H)
        attn_w=F.softmax(attn_s.float(),dim=0).to(mv.dtype) # (T,N,H)
        mem_read=(mvh*attn_w.unsqueeze(-1)).sum(0).reshape(N,M) # (N,M)
        mem_read=mem_read*gate_mask
        mem_out=self.mem_norm(self.from_memory(mem_read).float()).to(x.dtype)
        mix=torch.sigmoid(self.memory_mix)
        x_e=x+mix*mem_out.reshape(1,B,S,D).expand_as(x)
        stats={"memory_spike_rate":ms.mean().item(),"gate_activity":gate_sig.mean().item(),
            "memory_mix":mix.item(),
            "memory_attn_entropy":-(attn_w.float()*(attn_w.float()+1e-8).log()).sum(0).mean().item()}
        return x_e,stats

# ═══════════════════════════════════════════════════════════════════════════════
# §7b GENESIS DUAL MEMORY — two LIF banks (structural / personal), parallel fusion
# ═══════════════════════════════════════════════════════════════════════════════
class GenesisMemoryBank(nn.Module):
    """One spiking memory bank: returns residual contribution (no add-to-x). M must divide memory_n_read_heads."""

    def __init__(self,cfg:NordConfig,M:int,name:str):
        super().__init__()
        self.cfg=cfg; self.M=M; self.name=name; D=cfg.d_model
        self.to_memory=nn.Linear(D,M,bias=False)
        self.from_memory=nn.Linear(M,D,bias=False)
        self.memory_lif=AssociativeLIF(M,cfg,persistent=True,tau_mem_override=cfg.memory_tau_mem)
        self.gate_lif=AssociativeLIF(M,cfg)
        self.gate_proj=nn.Linear(D,M,bias=False)
        self.gate_threshold=nn.Parameter(torch.tensor(cfg.memory_gate_threshold))
        H=cfg.memory_n_read_heads; assert M%H==0; hd=M//H
        self.n_read_heads=H
        self.read_query=nn.Parameter(torch.randn(H,hd)*0.02)
        self.read_key_proj=nn.Linear(M,M,bias=False)
        self.read_scale=1.0/math.sqrt(hd)
        self.mem_norm=nn.LayerNorm(D)

    def reset_state(self)->None:
        self.memory_lif.reset_state()

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        T,B,S,D=x.shape; M=self.M; N=B*S; H=self.n_read_heads; hd=M//H
        xf=x.reshape(T,N,D)
        mi=self.to_memory(xf.reshape(T*N,D)).reshape(T,N,M)
        ms,mv=self.memory_lif(mi)
        gi=self.gate_proj(xf.reshape(T*N,D)).reshape(T,N,M)
        gs,_=self.gate_lif(gi)
        gate_sig=gs.mean(dim=0)
        gate_mask=torch.sigmoid((gate_sig-self.gate_threshold)*10.0)
        mvh=mv.reshape(T,N,H,hd)
        mk=self.read_key_proj(mv.reshape(T*N,M)).reshape(T,N,H,hd)
        q=self.read_query.unsqueeze(0).unsqueeze(0)
        attn_s=(q*mk).sum(-1)*self.read_scale
        attn_w=F.softmax(attn_s.float(),dim=0).to(mv.dtype)
        mem_read=(mvh*attn_w.unsqueeze(-1)).sum(0).reshape(N,M)
        mem_read=mem_read*gate_mask
        mem_out=self.mem_norm(self.from_memory(mem_read).float()).to(x.dtype)
        contrib=mem_out.reshape(1,B,S,D).expand_as(x)
        stats={
            f"{self.name}_memory_spike_rate":ms.mean().item(),
            f"{self.name}_gate_activity":gate_sig.mean().item(),
            f"{self.name}_memory_attn_entropy":-(attn_w.float()*(attn_w.float()+1e-8).log()).sum(0).mean().item(),
        }
        return contrib,stats

class DualGenesisMemoryCortex(nn.Module):
    """Maps Genesis dual external memory to two parallel trainable cortical banks + fused residual."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.cfg=cfg
        half=cfg.memory_size//2
        self.structural=GenesisMemoryBank(cfg,half,"struct")
        self.personal=GenesisMemoryBank(cfg,half,"pers")
        self.mix_structural=nn.Parameter(torch.tensor(0.2))
        self.mix_personal=nn.Parameter(torch.tensor(0.2))

    def reset_state(self)->None:
        self.structural.reset_state(); self.personal.reset_state()

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        c_s,st_s=self.structural(x)
        c_p,st_p=self.personal(x)
        ms=torch.sigmoid(self.mix_structural)
        mp=torch.sigmoid(self.mix_personal)
        x_e=x+ms*c_s+mp*c_p
        stats={**st_s,**st_p}
        stats["memory_mix_structural"]=float(ms.item())
        stats["memory_mix_personal"]=float(mp.item())
        stats["memory_spike_rate"]=(stats["struct_memory_spike_rate"]+stats["pers_memory_spike_rate"])*0.5
        stats["gate_activity"]=(stats["struct_gate_activity"]+stats["pers_gate_activity"])*0.5
        stats["memory_mix"]=(ms+mp).item()*0.5
        ent=(stats["struct_memory_attn_entropy"]+stats["pers_memory_attn_entropy"])*0.5
        stats["memory_attn_entropy"]=ent
        stats["genesis_dual_balance"]=(ms-mp).pow(2)
        return x_e,stats

# ═══════════════════════════════════════════════════════════════════════════════
# §7c NORD 5.0 — Genesis Autogenic Core (triple memory + identity GRU + archive RAG)
# ═══════════════════════════════════════════════════════════════════════════════
class GenesisTriplePurposeMemory(nn.Module):
    """Structural / Personal / Auxiliary (interests·unresolved) banks + purpose-filtered softmax fusion.
    Curriculum: memory as selection — router assigns per-position weights over containers."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.cfg=cfg
        Ms,Mp,Ma=96,96,64
        assert Ms+Mp+Ma==cfg.memory_size
        self.structural=GenesisMemoryBank(cfg,Ms,"struct")
        self.personal=GenesisMemoryBank(cfg,Mp,"pers")
        self.auxiliary=GenesisMemoryBank(cfg,Ma,"aux")
        self.purpose_router=nn.Linear(cfg.d_model,3,bias=False)
        nn.init.normal_(self.purpose_router.weight,std=0.02)

    def reset_state(self)->None:
        self.structural.reset_state(); self.personal.reset_state(); self.auxiliary.reset_state()

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        c_s,st_s=self.structural(x)
        c_p,st_p=self.personal(x)
        c_a,st_a=self.auxiliary(x)
        # Banks return contrib expanded to x: (T,B,S,D). F.layer_norm(..., (D,)) normalizes ONLY the
        # feature axis d_model — not time T, batch B, or sequence S (PyTorch normalized_shape suffix).
        D=self.cfg.d_model
        assert c_s.shape == x.shape and c_p.shape == x.shape and c_a.shape == x.shape
        assert c_s.shape[-1] == D
        c_s=F.layer_norm(c_s.float(),(D,),eps=1e-5).to(dtype=x.dtype)
        c_p=F.layer_norm(c_p.float(),(D,),eps=1e-5).to(dtype=x.dtype)
        c_a=F.layer_norm(c_a.float(),(D,),eps=1e-5).to(dtype=x.dtype)
        assert c_s.shape == x.shape
        h=x.mean(dim=0)
        w=F.softmax(self.purpose_router(h),dim=-1)
        stack=torch.stack([c_s,c_p,c_a],dim=-2)
        w_exp=w.view(1,h.shape[0],h.shape[1],3,1).to(dtype=x.dtype)
        fused=(stack*w_exp).sum(dim=-2)
        x_e=x+fused
        ent=-(w*(w+1e-8).log()).sum(dim=-1).mean()
        stats={**st_s,**st_p,**st_a}
        stats["memory_spike_rate"]=(st_s["struct_memory_spike_rate"]+st_p["pers_memory_spike_rate"]+st_a["aux_memory_spike_rate"])/3.0
        stats["gate_activity"]=(st_s["struct_gate_activity"]+st_p["pers_gate_activity"]+st_a["aux_gate_activity"])/3.0
        stats["memory_mix"]=float(w.mean().item())
        stats["memory_attn_entropy"]=(st_s["struct_memory_attn_entropy"]+st_p["pers_memory_attn_entropy"]+st_a["aux_memory_attn_entropy"])/3.0
        stats["genesis_purpose_entropy"]=ent
        stats["genesis_purpose_balance"]=w.std(dim=-1).mean().pow(2)
        return x_e,stats

class GenesisIdentityTrack(nn.Module):
    r"""Recursive identity along sequence: implements discrete-time update
    I_t \approx GRU(M_t, I_{t-1}) with M_t = pooled cortical input at t (mean over spike-time)."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.d_id=max(64,cfg.d_model//8)
        self.gru=nn.GRU(cfg.d_model,self.d_id,batch_first=True)
        self.to_d=nn.Linear(self.d_id,cfg.d_model,bias=False)
        self.gate_raw=nn.Parameter(torch.tensor(0.0))

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,Tensor]]:
        h=x.mean(dim=0)
        out,_=self.gru(h)
        inj=self.to_d(out)
        g=torch.sigmoid(self.gate_raw)
        x_e=x+g*inj.unsqueeze(0).expand_as(x)
        stats={"identity_hidden_norm":out.pow(2).mean()}
        return x_e,stats

class GenesisArchiveGrid(nn.Module):
    """Recursive Archive Grid (learned): K static slots + softmax attention = differentiable RAG prior."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.k=cfg.genesis_archive_slots
        D=cfg.d_model
        self.keys=nn.Parameter(torch.randn(self.k,D)*0.02)
        self.values=nn.Parameter(torch.randn(self.k,D)*0.02)
        self.gamma_raw=nn.Parameter(torch.tensor(0.0))
        self.scale=1.0/math.sqrt(D)

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,Tensor]]:
        h=x.mean(dim=0)
        logits=torch.matmul(h,self.keys.T)*self.scale
        attn=F.softmax(logits,dim=-1)
        ctx=torch.matmul(attn,self.values)
        g=torch.sigmoid(self.gamma_raw)
        x_e=x+g*ctx.unsqueeze(0).expand_as(x)
        ent=-(attn*(attn+1e-8).log()).sum(dim=-1).mean()
        stats={"genesis_archive_attn_entropy":ent}
        return x_e,stats

# ═══════════════════════════════════════════════════════════════════════════════
# §7d  PAPER-INSPIRED CORTICAL STACK (Genesis v5 only) — consolidated in-core
#  Liu et al. TNNLS 2017 (V1–MT): parallel motion-like streams + center–surround along sequence
#  Tang et al. CoRL 2020 (PopSAN): population coding with learnable Gaussian RFs on a bottleneck
#  Susi et al. Sci Rep 2021 (FNS/LIFL): input-gated multiplicative gain (latency/threshold metaphor)
# ═══════════════════════════════════════════════════════════════════════════════
class PaperCorticalStackV5(nn.Module):
    """One differentiable stack: DoG sequence filter → softmax-mixed directional streams → PopSAN bottleneck → gain."""

    def __init__(self, cfg: NordConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.d_model
        r = max(1, int(cfg.paper_surround_radius))
        self.r = r
        t = torch.arange(-r, r + 1, dtype=torch.float32)
        sig_c = max(r / 3.0, 0.5)
        sig_s = max(r * 0.7, 0.5)
        c = torch.exp(-0.5 * (t / sig_c) ** 2)
        s = torch.exp(-0.5 * (t / sig_s) ** 2)
        dog = c / c.sum() - 0.6 * (s / s.sum())
        self.register_buffer("dog_weight", dog.view(1, 1, -1).expand(D, 1, 2 * r + 1).contiguous())
        self.mix_dog = nn.Parameter(torch.tensor(0.2))

        self.stream_gate = nn.Linear(D, 3, bias=False)
        nn.init.normal_(self.stream_gate.weight, std=0.02)

        b = max(4, int(cfg.paper_pop_bottleneck))
        k = max(2, int(cfg.paper_pop_per_dim))
        self.bottleneck = b
        self.pop_k = k
        self.down = nn.Linear(D, b, bias=False)
        self.mu = nn.Parameter(torch.linspace(-1.5, 1.5, k).unsqueeze(0).expand(b, k).clone())
        self.log_sigma = nn.Parameter(torch.zeros(b, k))
        self.up = nn.Linear(b * k, D, bias=False)
        self.mix_pop = nn.Parameter(torch.tensor(0.15))

        self.latency_raw = nn.Parameter(torch.tensor(0.0))
        self.lat_strength = float(cfg.paper_latency_bias_strength)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        Tn, B, S, D = x.shape
        xb = x.permute(0, 1, 3, 2).reshape(Tn * B, D, S)
        dog = F.conv1d(xb, self.dog_weight, padding=self.r, groups=D)
        dog = dog.view(Tn, B, D, S).permute(0, 1, 3, 2)
        x1 = x + torch.tanh(self.mix_dog) * dog

        xs = torch.roll(x1, shifts=1, dims=2)
        xs[:, :, 0, :] = x1[:, :, 0, :]
        xa = torch.roll(x1, shifts=-1, dims=2)
        xa[:, :, -1, :] = x1[:, :, -1, :]
        w = F.softmax(self.stream_gate(x1), dim=-1)
        x2 = w[..., 0:1] * x1 + w[..., 1:2] * xs + w[..., 2:3] * xa

        flat = x2.reshape(Tn * B * S, D)
        z = self.down(flat)
        sigma = F.softplus(self.log_sigma) + 1e-2
        diff = (z.unsqueeze(-1) - self.mu.view(1, self.bottleneck, self.pop_k)) / sigma.view(1, self.bottleneck, self.pop_k)
        act = torch.exp(-0.5 * diff * diff)
        pop_flat = act.reshape(Tn * B * S, self.bottleneck * self.pop_k)
        delta = self.up(pop_flat).reshape(Tn, B, S, D)
        x3 = x2 + torch.tanh(self.mix_pop) * delta

        mag = x3.norm(dim=-1, keepdim=True).clamp(max=12.0)
        gain = 1.0 + self.lat_strength * torch.sigmoid(self.latency_raw) * torch.tanh(mag)
        x4 = x3 * gain

        st: Dict[str, Tensor] = {
            "paper_dog_mix": self.mix_dog.detach().reshape(1),
            "paper_pop_mix": self.mix_pop.detach().reshape(1),
            "paper_latency_gate": torch.sigmoid(self.latency_raw).detach().reshape(1),
        }
        return x4, st

# ═══════════════════════════════════════════════════════════════════════════════
# §8  BLOCKS — FIX H: Gradient checkpointing
# ═══════════════════════════════════════════════════════════════════════════════
class SpikingFeedForward(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.up=nn.Linear(cfg.d_model,cfg.d_ff,bias=False)
        self.down=nn.Linear(cfg.d_ff,cfg.d_model,bias=False)
        self.lif1=AssociativeLIF(cfg.d_ff,cfg); self.lif2=AssociativeLIF(cfg.d_model,cfg)
    def forward(self,x:Tensor)->Tensor:
        T,B,S,D=x.shape
        h=self.up(x.reshape(T*B*S,D)).reshape(T,B*S,-1); h,_=self.lif1(h)
        h=self.down(h.reshape(T*B*S,-1)).reshape(T,B*S,D); h,_=self.lif2(h)
        return h.reshape(T,B,S,D)

class LeakyClamp(nn.Module):
    def __init__(self,d:int,floor_init:float=-0.1,leak_init:float=0.1,force_nonneg:bool=False):
        super().__init__()
        # FIX M: force_nonneg=True for executive blocks — no negative spikes
        self.force_nonneg=force_nonneg
        if force_nonneg:
            floor_init=0.0
        self.floor=nn.Parameter(torch.full((d,),floor_init))
        self.leak_raw=nn.Parameter(torch.full((d,),math.log(leak_init/(1-leak_init+1e-6))))
    @property
    def leak(self)->Tensor: return torch.sigmoid(self.leak_raw)
    def forward(self,x:Tensor)->Tensor:
        if self.force_nonneg:
            # Executive: no negative values allowed
            return F.relu(x)
        return torch.where(x>=0,x,(self.leak*x).clamp(min=self.floor))

class NordBlock(nn.Module):
    def __init__(self,cfg:NordConfig,layer_idx:int=0,use_moe:bool=False,zone:str="sensory"):
        super().__init__(); D=cfg.d_model; self.use_moe=use_moe; self.zone=zone
        self.layer_idx=layer_idx; self.use_checkpoint=cfg.gradient_checkpointing
        self.norm1=nn.LayerNorm(D); self.norm2=nn.LayerNorm(D)
        self.resonance=SpikingSynapticResonance(cfg)
        if use_moe: self.moe=SpikeDrivenMoE(cfg)
        else: self.ffn=SpikingFeedForward(cfg)
        sc=0.1/max(cfg.n_layers_total,1)
        self.gamma_attn=nn.Parameter(torch.full((D,),sc))
        self.gamma_ffn=nn.Parameter(torch.full((D,),sc))
        # FIX M: Executive blocks force non-negative output
        self.clamp=LeakyClamp(D,floor_init=cfg.clamp_floor,
                            force_nonneg=(zone=="executive"))
    @staticmethod
    def _sn(nl:nn.LayerNorm,x:Tensor)->Tensor:
        od=x.dtype
        return F.layer_norm(x.float(),nl.normalized_shape,
            nl.weight.float() if nl.weight is not None else None,
            nl.bias.float() if nl.bias is not None else None,nl.eps).to(od)
    def _forward_inner(self,x:Tensor)->Tuple[Tensor,Dict]:
        stats={}
        x=x+self.gamma_attn*self.resonance(self._sn(self.norm1,x))
        xn=self._sn(self.norm2,x)
        if self.use_moe: fo,ms=self.moe(xn); stats.update(ms)
        else: fo=self.ffn(xn)
        return self.clamp(x+self.gamma_ffn*fo),stats
    def forward(self,x:Tensor)->Tuple[Tensor,Dict]:
        if self.use_checkpoint and self.training:
            x=grad_checkpoint(lambda inp:self._forward_inner(inp)[0],x,use_reentrant=False)
            return x,{}
        return self._forward_inner(x)

# ═══════════════════════════════════════════════════════════════════════════════
# §9  SPIKE REGULATOR — FIX C: Differentiable
# ═══════════════════════════════════════════════════════════════════════════════
class AuxiliarySpikeRegulator(nn.Module):
    """FIX L: Adaptive spike regulator.
    - Stronger weight (0.5 default)
    - Extra penalty when any layer drops below min_rate (anti-death)
    - Asymmetric: penalizes too-low firing 3x more than too-high"""
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.target=cfg.target_spike_rate
        self.weight=cfg.spike_loss_weight
        self.min_rate=0.01  # absolute minimum — below this = dead layer
    def forward(self,spike_tensors:List[Tensor])->Tensor:
        if not spike_tensors:
            return torch.tensor(0.0, dtype=torch.float32)
        dev=spike_tensors[0].device
        loss=torch.zeros((),device=dev,dtype=torch.float32)
        t_rate=float(self.target)
        for s in spike_tensors:
            rate=s.float().clamp(min=0).mean()
            diff=t_rate-rate
            w_asym=torch.where(diff>0,diff.new_tensor(3.0),diff.new_tensor(1.0))
            loss=loss+w_asym*diff*diff
            loss=loss+10.0*torch.relu(self.min_rate-rate).pow(2)
        return self.weight*loss/len(spike_tensors)

# ═══════════════════════════════════════════════════════════════════════════════
# §10  STDP — FIX F + neuromodulation / skeleton / metaplasticity
# ═══════════════════════════════════════════════════════════════════════════════
class STDPEngine:
    """STDP with optional 'dopamine' gate (output entropy), skeleton weights, metaplastic rigidity."""

    def __init__(self,cfg:NordConfig):
        self.cfg=cfg; self.a_plus=cfg.stdp_a_plus; self.a_minus=cfg.stdp_a_minus
        self.tau_plus=cfg.stdp_tau_plus; self.tau_minus=cfg.stdp_tau_minus
        self.w_max=cfg.stdp_w_max; self.w_min=cfg.stdp_w_min
        self.reward_scale=cfg.stdp_reward_scale
        self.allowed=set(cfg.stdp_layers or [])
        self._loss_ema=10.0; self._ema_decay=0.99; self.max_update_norm=0.01
        self._last_output_entropy:Optional[float]=None
        self._plasticity_trace:Dict[str,Tensor]={}

    def set_output_entropy(self,entropy:Optional[float])->None:
        """Low entropy ⇒ confident prediction ⇒ neuromodulation can block STDP updates."""
        self._last_output_entropy=entropy

    def should_apply_plasticity(self)->bool:
        if self.cfg.stdp_read_only:
            return False
        if not self.cfg.stdp_neuromodulation:
            return True
        if self._last_output_entropy is None:
            return True
        return float(self._last_output_entropy)>float(self.cfg.stdp_entropy_high_threshold)

    def update_reward(self,cl:float): self._loss_ema=self._ema_decay*self._loss_ema+(1-self._ema_decay)*cl
    def _compute_reward(self,cl:float)->float:
        return float(torch.sigmoid(torch.tensor((self._loss_ema-cl)*self.reward_scale)).item())
    def is_allowed(self,name:str)->bool: return name in self.allowed

    @torch.no_grad()
    def compute_stdp_update(self,pre:Tensor,post:Tensor)->Tensor:
        T=pre.shape[0]; d=pre.device
        tp=torch.zeros_like(pre[0]); tpo=torch.zeros_like(post[0])
        dp=math.exp(-1.0/self.tau_plus); dm=math.exp(-1.0/self.tau_minus)
        dW=torch.zeros(post.shape[1],pre.shape[1],device=d,dtype=pre.dtype)
        for t in range(T):
            tp=tp*dp+pre[t]; tpo=tpo*dm+post[t]
            if post[t].any(): dW+=self.a_plus*torch.outer(post[t],tp)
            if pre[t].any(): dW-=self.a_minus*torch.outer(tpo,pre[t])
        n=dW.norm()
        if n>self.max_update_norm: dW=dW*(self.max_update_norm/n)
        return dW

    @torch.no_grad()
    def apply_to_layer(self,layer:nn.Linear,pre:Tensor,post:Tensor,
                    cl:Optional[float]=None,name:str=""):
        if not self.should_apply_plasticity():
            return
        if name and not self.is_allowed(name): return
        if pre.dim()==3: pre=pre.mean(dim=1)
        if post.dim()==3: post=post.mean(dim=1)
        dW=self.compute_stdp_update(pre,post)
        if cl is not None:
            r=self._compute_reward(cl); dW=dW*(2.0*r-1.0); self.update_reward(cl)
        o,i=layer.weight.shape; dW=dW[:o,:i]
        w=layer.weight.data
        sk=float(self.cfg.stdp_skeleton_scale)
        if sk>0:
            dW=dW/(1.0+sk*w.abs())
        key=name or str(id(layer))
        if self.cfg.stdp_metaplastic:
            if key not in self._plasticity_trace or self._plasticity_trace[key].shape!=w.shape:
                self._plasticity_trace[key]=torch.ones_like(w)
            rig=self._plasticity_trace[key].clamp(1.0,float(self.cfg.stdp_metaplastic_rigidity_max))
            dW=dW/rig
        layer.weight.data=(w+dW).clamp(self.w_min,self.w_max)
        if self.cfg.stdp_metaplastic:
            g=float(self.cfg.stdp_metaplastic_gamma)
            self._plasticity_trace[key]=g*self._plasticity_trace[key]+(1.0-g)*dW.abs()

# ═══════════════════════════════════════════════════════════════════════════════
# §10b  CONVERSATIONAL READOUT BALANCER — coherence / dialogue-shaped readout
# ═══════════════════════════════════════════════════════════════════════════════
class ConversationalReadoutBalancer(nn.Module):
    """Learned adapter before ``lm_head``: LN(local) + sequence-mean (gist) → MLP residual.

    Not a separate \"language model\" — one CE loss trains the whole stack. Works best when the
    corpus includes instructions / multi-turn text so logits align with fluent, on-topic replies.
    """

    def __init__(self, cfg: NordConfig):
        super().__init__()
        D = cfg.d_model
        if cfg.conversational_balancer_hidden > 0:
            H = cfg.conversational_balancer_hidden
        else:
            H = min(cfg.d_ff, max(D, 1024))
        self.norm = nn.LayerNorm(D)
        self.fuse = nn.Linear(2 * D, H, bias=False)
        self.proj = nn.Linear(H, D, bias=False)
        self.gate_raw = nn.Parameter(torch.tensor(0.0))
        self.register_buffer("max_scale", torch.tensor(float(cfg.conversational_balancer_max_scale)))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, S, D = x.shape
        xl = self.norm(x.float()).to(x.dtype)
        g = x.mean(dim=1, keepdim=True)
        xg = self.norm(g.float()).to(x.dtype).expand(B, S, D)
        h = torch.cat([xl, xg], dim=-1)
        delta = self.proj(F.gelu(self.fuse(h)))
        scale = torch.sigmoid(self.gate_raw) * self.max_scale.to(dtype=x.dtype, device=x.device)
        return x + scale * delta, scale

# ═══════════════════════════════════════════════════════════════════════════════
# §10c–h  Optional cortical motifs: context LIF, dendrites, predictive coding, CLS, oscillations
# ═══════════════════════════════════════════════════════════════════════════════
class ContextLIF(nn.Module):
    """Slow, almost bistable integration over batch-mean input (working-memory style context).
    Wang XJ-style mnemonic reverberation approximated with tau≈1 and soft reset; topic shift damps state."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        self.cfg=cfg
        D=cfg.d_model
        self.gate=float(cfg.context_lif_gate)
        self.tau_context_raw=nn.Parameter(torch.tensor(math.log(0.999/(1-0.999+1e-6))))
        self.threshold_raw=nn.Parameter(torch.full((D,),0.22))
        self.topic_proj=nn.Linear(D,64,bias=False)
        self.inhib_thresh=nn.Parameter(torch.tensor(0.45))
        self.register_buffer("_topic_gist",torch.zeros(1,64))
        self.register_buffer("_context_charge",torch.zeros(1,D))
        ni=cfg.context_inhibitory_neurons
        if ni>0:
            self.inhib_in=nn.Linear(D,ni,bias=False)
            self.inhib_lif=AssociativeLIF(ni,cfg,persistent=False)
        else:
            self.inhib_in=None
            self.inhib_lif=None

    @property
    def tau_context(self)->Tensor:
        return torch.sigmoid(self.tau_context_raw).clamp(0.999,1.0)

    @property
    def threshold(self)->Tensor:
        return self.threshold_raw.clamp(0.12,0.55)

    def reset_context(self)->None:
        self._topic_gist.zero_()
        self._context_charge.zero_()

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        T,B,S,D=x.shape
        device,dtype=x.device,x.dtype
        x_mean=x.mean(dim=(0,2))
        current_gist=self.topic_proj(x_mean.mean(dim=0,keepdim=True))
        if self._topic_gist.abs().sum()<1e-8:
            with torch.no_grad():
                self._topic_gist.copy_(current_gist.detach())
            topic_shift=torch.zeros((),device=device,dtype=dtype)
        else:
            sim=F.cosine_similarity(current_gist,self._topic_gist,dim=-1).clamp(0,1)
            topic_shift=(1.0-sim).squeeze(0)
            with torch.no_grad():
                self._topic_gist.mul_(0.9).add_(0.1*current_gist.detach())
        inh_gate=torch.sigmoid(self.inhib_thresh)
        charge=self._context_charge.to(dtype)
        if charge.shape[0]!=B:
            charge=torch.zeros(B,D,device=device,dtype=dtype)
        if topic_shift>inh_gate:
            with torch.no_grad():
                strength=((topic_shift-inh_gate)/(1.0-inh_gate+1e-6)).clamp(0,1)
                charge=charge*(1.0-0.5*strength)
        if self.inhib_lif is not None:
            xi=x.mean(dim=2)
            h=self.inhib_in(xi.reshape(T*B,D)).reshape(T,B,-1)
            inh,_=self.inhib_lif(h)
            decay=1.0-0.25*inh.mean(dim=-1,keepdim=True)
            charge=charge*decay.mean(dim=0)
        tau=self.tau_context.to(dtype)
        thr=self.threshold.to(dtype)
        ctx_spikes:List[Tensor]=[]
        for _ in range(T):
            charge=tau*charge+(1.0-tau)*x_mean
            s=spike_fn(charge,thr,self.cfg.surrogate_alpha)
            charge=charge-s*thr.detach()*0.1
            ctx_spikes.append(s)
        with torch.no_grad():
            if self._context_charge.shape==(B,D):
                self._context_charge.copy_(charge.detach())
            else:
                self._context_charge=charge.detach().clone()
        ctx=torch.stack(ctx_spikes,dim=0).unsqueeze(2).expand(T,B,S,D)
        x_e=x+self.gate*ctx
        sr=torch.stack(ctx_spikes).float().mean()
        st={
            "context_spike_rate":float(sr.item()),
            "context_topic_shift":float(topic_shift.item()),
            "context_inhibitory_gate":float(inh_gate.item()),
        }
        return x_e,st


class DendriticLIF(nn.Module):
    """Parallel dendritic branches (Gidon et al. 2020-style nonlinearity) → soma → AssociativeLIF."""

    def __init__(self,d:int,cfg:NordConfig,n_dendrites:int=4):
        super().__init__()
        self.d=d; self.n=max(2,n_dendrites)
        db=max(1,d//self.n)
        self.db=db
        self.branches=nn.ModuleList([nn.Linear(d,db,bias=False) for _ in range(self.n)])
        self.branch_steep=nn.Parameter(torch.ones(self.n)*2.0)
        soma_in=self.n*db
        self.soma_proj=nn.Linear(soma_in,d,bias=False)
        self.lif=AssociativeLIF(d,cfg,persistent=cfg.persistent_mem)

    def forward(self,current_in:Tensor)->Tuple[Tensor,Tensor]:
        T,N,D=current_in.shape
        outs:List[Tensor]=[]
        flat=current_in.reshape(T*N,D)
        for i,br in enumerate(self.branches):
            o=br(flat)
            o=torch.sigmoid(self.branch_steep[i]*o)
            outs.append(o)
        comb=torch.cat(outs,dim=-1)
        som=self.soma_proj(comb).reshape(T,N,D)
        return self.lif(som+current_in)

    def reset_state(self)->None:
        self.lif.reset_state()


class PredictiveCodingLayer(nn.Module):
    """Sparse prediction error along sequence (Friston-style); feeds AssociativeLIF."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        D=cfg.d_model
        self.predictor=nn.Linear(D,D,bias=False)
        self.err_raw=nn.Parameter(torch.tensor(0.0))
        self.lif=AssociativeLIF(D,cfg,persistent=False)
        self.res=float(cfg.predictive_coding_residual)

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        T,B,S,D=x.shape
        prev=torch.roll(x,1,dims=2)
        prev[:,:,0,:]=0
        pred=self.predictor(prev.reshape(-1,D)).reshape(T,B,S,D)
        err=x-pred
        mag=err.abs().mean(-1,keepdim=True)
        thr=F.softplus(self.err_raw)+0.05
        gate=(mag>thr).to(err.dtype)
        filtered=err*gate
        sp,_=self.lif(filtered.reshape(T,B*S,D))
        sp=sp.reshape(T,B,S,D)
        out=x+self.res*sp
        st={"pc_prediction_error":float(mag.mean().item()),"pc_suppression_rate":float((1.0-gate.mean()).item())}
        return out,st


class ComplementaryLearningSystems(nn.Module):
    """Fast hippocampal-like LIF on low-d projection + blend with cortical stream (Kumaran et al. CLS)."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        D=cfg.d_model
        self.d_hip=max(32,D//max(2,cfg.cls_hippo_frac))
        self.hip_in=nn.Linear(D,self.d_hip,bias=False)
        self.hip_lif=AssociativeLIF(self.d_hip,cfg,persistent=False,tau_mem_override=0.55)
        self.hip_out=nn.Linear(self.d_hip,D,bias=False)
        self.cortex_gate_raw=nn.Parameter(torch.tensor(0.0))
        self.hippo_gate_raw=nn.Parameter(torch.tensor(0.0))

    def forward(self,x:Tensor)->Tuple[Tensor,Dict[str,float]]:
        T,B,S,D=x.shape
        xi=x.reshape(T*B*S,D)
        h=self.hip_in(xi).reshape(T,B*S,self.d_hip)
        sp,_=self.hip_lif(h)
        hip=self.hip_out(sp.reshape(-1,self.d_hip)).reshape(T,B,S,D)
        cg=torch.sigmoid(self.cortex_gate_raw)
        hg=torch.sigmoid(self.hippo_gate_raw)
        fused=(cg*x+hg*hip)/(cg+hg+1e-6)
        st={"cls_cortex_gate":float(cg.item()),"cls_hippo_gate":float(hg.item()),"cls_hip_spike":float(sp.float().mean().item())}
        return fused,st


class OscillatoryGating(nn.Module):
    """Multiplicative rhythm gate on (T,B,S,D) — Muller-style traveling-wave metaphor (simplified)."""

    def __init__(self,cfg:NordConfig):
        super().__init__()
        D=cfg.d_model
        self.gamma_raw=nn.Parameter(torch.tensor(0.08))
        self.theta_raw=nn.Parameter(torch.tensor(0.018))
        self.phase=nn.Parameter(torch.randn(D)*0.02)

    def forward(self,x:Tensor,step:int)->Tensor:
        t=float(step); Dm=x.shape[-1]
        g=torch.sigmoid(self.gamma_raw)
        th=torch.sigmoid(self.theta_raw)
        rhythm=(torch.sin(t*g*6.283+self.phase)+(torch.sin(t*th*6.283+self.phase)))/2.0
        gate=torch.sigmoid(rhythm).view(1,1,1,Dm).to(dtype=x.dtype,device=x.device)
        return x*gate

# ═══════════════════════════════════════════════════════════════════════════════
# §11  NORD MODEL v5.0 / v4.2
# ═══════════════════════════════════════════════════════════════════════════════
class NordModel(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.cfg=cfg
        self.encoder=TemporalSpikeEncoder(cfg)
        if cfg.dendritic_input_enabled:
            self.input_lif=DendriticLIF(cfg.d_model,cfg,cfg.dendritic_n_branches)
        else:
            self.input_lif=AssociativeLIF(cfg.d_model,cfg,persistent=cfg.persistent_mem)
        self.context_lif=ContextLIF(cfg) if cfg.context_lif_enabled else None
        self.pc_layer=PredictiveCodingLayer(cfg) if cfg.predictive_coding_enabled else None
        self.cls_module=ComplementaryLearningSystems(cfg) if cfg.cls_fast_slow_enabled else None
        self.osc_gate=OscillatoryGating(cfg) if cfg.oscillatory_gating_enabled else None
        self.register_buffer("_osc_step",torch.tensor(0,dtype=torch.long))
        self.sensory_blocks=nn.ModuleList([NordBlock(cfg,i,False,zone="sensory") for i in range(cfg.sensory_layers)])
        self.association_blocks=nn.ModuleList([NordBlock(cfg,cfg.sensory_layers+i,True,zone="association") for i in range(cfg.association_layers)])
        if cfg.genesis_autogenic_v5:
            self.memory_cortex=GenesisTriplePurposeMemory(cfg)
            self.genesis_identity=GenesisIdentityTrack(cfg)
            self.genesis_archive=GenesisArchiveGrid(cfg)
        elif cfg.genesis_dual_memory:
            self.memory_cortex=DualGenesisMemoryCortex(cfg)
            self.genesis_identity=None
            self.genesis_archive=None
        else:
            self.memory_cortex=MemoryCortex(cfg)
            self.genesis_identity=None
            self.genesis_archive=None
        self.paper_stack=PaperCorticalStackV5(cfg) if (cfg.genesis_autogenic_v5 and cfg.paper_cortical_stack_v5) else None
        self.executive_blocks=nn.ModuleList([NordBlock(cfg,cfg.sensory_layers+cfg.association_layers+i,False,zone="executive") for i in range(cfg.executive_layers)])
        self.readout_lif=AssociativeLIF(cfg.d_model,cfg,persistent=cfg.persistent_mem)
        self.readout_ema_raw=nn.Parameter(torch.tensor(1.4))
        self.readout_norm=nn.LayerNorm(cfg.d_model)
        self.readout_balancer=ConversationalReadoutBalancer(cfg) if cfg.conversational_balancer else None
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.stdp=STDPEngine(cfg); self._last_loss=None
        self.spike_regulator=AuxiliarySpikeRegulator(cfg)

    @property
    def readout_ema_decay(self)->Tensor: return torch.sigmoid(self.readout_ema_raw)
    def reset_state(self):
        self.input_lif.reset_state(); self.readout_lif.reset_state()
        self.memory_cortex.reset_state()
        if self.context_lif is not None:
            self.context_lif.reset_context()
        self._osc_step.zero_()

    def forward(self,token_ids:Tensor,enable_stdp:bool=False)->Tuple[Tensor,Dict]:
        B,S=token_ids.shape; T_t=self.cfg.T_total; D=self.cfg.d_model
        cur=self.encoder(token_ids); isp,_=self.input_lif(cur)
        isp=isp.reshape(T_t,B,S,D)
        spike_ts=[isp]; stats={}; moe_lb=torch.tensor(0.0,device=token_ids.device)

        x=isp
        for i,bl in enumerate(self.sensory_blocks):
            x,bs=bl(x); spike_ts.append(x)
            for k,v in bs.items(): stats[f"sensory_{i}_{k}"]=v

        for i,bl in enumerate(self.association_blocks):
            x,bs=bl(x); spike_ts.append(x)
            lb=bs.pop("moe_load_balance_loss",None)
            if lb is not None: moe_lb=moe_lb+lb
            for k,v in bs.items(): stats[f"assoc_{i}_{k}"]=v

        x,ms=self.memory_cortex(x); stats.update(ms)
        if self.context_lif is not None:
            x,ctxs=self.context_lif(x); stats.update(ctxs)
        if self.pc_layer is not None:
            x,pcs=self.pc_layer(x); stats.update(pcs)
        if self.cls_module is not None:
            x,clss=self.cls_module(x); stats.update(clss)
        if self.genesis_identity is not None:
            x,ids=self.genesis_identity(x)
            stats.update(ids)

        if self.paper_stack is not None:
            x,pst=self.paper_stack(x)
            for pk,pv in pst.items():
                stats[pk]=float(pv.item()) if pv.numel()==1 else pv

        if self.osc_gate is not None:
            x=self.osc_gate(x,int(self._osc_step.item()))
            self._osc_step+=1

        for i,bl in enumerate(self.executive_blocks):
            x,bs=bl(x); spike_ts.append(x)
            for k,v in bs.items(): stats[f"exec_{i}_{k}"]=v

        if self.genesis_archive is not None:
            x,ars=self.genesis_archive(x)
            stats.update(ars)

        xf=x.reshape(T_t,B*S,D); rsp,vm=self.readout_lif(xf)
        a=self.readout_ema_decay
        ema=torch.zeros(B*S,D,device=x.device,dtype=vm.dtype)
        for t in range(T_t): ema=a*ema+(1-a)*vm[t]
        vs=ema.reshape(B,S,D)
        sm=rsp.mean(dim=0).reshape(B,S,D)
        ro=vs+sm
        xn=F.layer_norm(ro.float(),self.readout_norm.normalized_shape,
            self.readout_norm.weight.float() if self.readout_norm.weight is not None else None,
            self.readout_norm.bias.float() if self.readout_norm.bias is not None else None,
            self.readout_norm.eps).to(ro.dtype)
        bal_scale = torch.tensor(0.0, device=xn.device, dtype=torch.float32)
        if self.readout_balancer is not None:
            xn, bal_scale = self.readout_balancer(xn)
        logits=self.lm_head(xn)

        out_rate=rsp.detach().mean().item()
        # FIX K: clamp negatives — spike rates cannot be negative
        sr=[s.detach().clamp(min=0).mean().item() for s in spike_ts]

        # Convert ALL stats to tensors for DataParallel gather compatibility
        dev = token_ids.device
        tensor_stats = {}
        tensor_stats["sparsity"] = torch.tensor(1.0 - out_rate, device=dev)
        tensor_stats["avg_spike_rate"] = torch.tensor(sum(sr)/len(sr), device=dev)
        tensor_stats["spike_loss"] = self.spike_regulator(spike_ts)
        tensor_stats["moe_lb_loss"] = moe_lb
        tensor_stats["readout_balancer_gate"] = bal_scale.detach().reshape(1).to(dev)
        # Pack spike_rates as a single tensor
        tensor_stats["spike_rates_tensor"] = torch.tensor(sr, device=dev)
        # Convert any float stats from blocks/memory to tensors
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                tensor_stats[k] = torch.tensor(v, device=dev)
            elif isinstance(v, torch.Tensor):
                tensor_stats[k] = v.to(dev) if v.device != dev else v
            # skip lists and other non-tensor types
        return logits, tensor_stats

    def set_last_loss(self,l:float): self._last_loss=l
    def count_params(self)->str:
        total=sum(p.numel() for p in self.parameters())
        train=sum(p.numel() for p in self.parameters() if p.requires_grad)
        se=sum(p.numel() for n,p in self.named_parameters() if 'sensory' in n)
        a=sum(p.numel() for n,p in self.named_parameters() if 'association' in n)
        m=sum(p.numel() for n,p in self.named_parameters() if 'memory' in n)
        e=sum(p.numel() for n,p in self.named_parameters() if 'executive' in n)
        return(f"Total: {total/1e6:.1f}M | Trainable: {train/1e6:.1f}M\n"
            f"  Sensory:     {se/1e6:.1f}M ({self.cfg.sensory_layers} blocks)\n"
            f"  Association: {a/1e6:.1f}M ({self.cfg.association_layers} blocks, MoE)\n"
            f"  Memory:      {m/1e6:.1f}M\n"
            f"  Executive:   {e/1e6:.1f}M ({self.cfg.executive_layers} blocks)")