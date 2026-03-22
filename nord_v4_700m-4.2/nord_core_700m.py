"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                PROJECT NORD — Core Engine  v4.2  (700M)                 ║
║          Spiking Neural Network LLM with Brain-Inspired Architecture       ║
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
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

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
    # FIX H
    gradient_checkpointing:bool=False
    # Training
    batch_size:int=1; grad_accum:int=32; lr:float=2e-4; min_lr:float=1e-5
    weight_decay:float=0.01; warmup_steps:int=1000; max_steps:int=50_000
    save_every:int=1000; log_every:int=10; max_grad_norm:float=1.0
    dtype:torch.dtype=torch.float16; device:str="cuda"
    @property
    def T_total(self)->int: return self.T+self.T_slow
    @property
    def n_layers_total(self)->int: return self.sensory_layers+self.association_layers+self.executive_layers
    def __post_init__(self):
        if self.stdp_layers is None:
            self.stdp_layers=[f"executive_{i}" for i in range(self.executive_layers)]

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
            if s.sum()>0: i_syn=i_syn+self._cascade_amplify(s)
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
        res=torch.matmul(qm,km.transpose(-2,-1))*self.resonance_temp
        cmask=torch.triu(torch.ones(S,S,device=x_spikes.device,dtype=torch.bool),diagonal=1)
        res.masked_fill_(cmask.unsqueeze(0).unsqueeze(0),float("-inf"))
        K=min(self.top_k,S)
        if K<S:
            tv,ti=torch.topk(res,K,dim=-1)
            sr=torch.full_like(res,float("-inf")); sr.scatter_(-1,ti,tv); res=sr
        attn=F.softmax(res.float(),dim=-1).to(res.dtype)
        vm=vr.mean(dim=0).reshape(B,S,H,Dh).permute(0,2,1,3)
        ctx=torch.matmul(attn,vm).permute(0,2,1,3).reshape(B,S,D)
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
        if not spike_tensors: return torch.tensor(0.0)
        loss=torch.tensor(0.0,device=spike_tensors[0].device,dtype=torch.float32)
        for s in spike_tensors:
            # FIX K: Only count non-negative values as spikes
            rate=s.float().clamp(min=0).mean()
            diff=self.target-rate
            # Asymmetric: penalize too-low firing 3x more
            if diff>0:
                loss=loss+3.0*diff**2
            else:
                loss=loss+diff**2
            # Anti-death penalty: heavy penalty if rate < min_rate
            if rate<self.min_rate:
                loss=loss+10.0*(self.min_rate-rate)**2
        return self.weight*loss/len(spike_tensors)

# ═══════════════════════════════════════════════════════════════════════════════
# §10  STDP — FIX F: Bounded + Isolated
# ═══════════════════════════════════════════════════════════════════════════════
class STDPEngine:
    def __init__(self,cfg:NordConfig):
        self.cfg=cfg; self.a_plus=cfg.stdp_a_plus; self.a_minus=cfg.stdp_a_minus
        self.tau_plus=cfg.stdp_tau_plus; self.tau_minus=cfg.stdp_tau_minus
        self.w_max=cfg.stdp_w_max; self.w_min=cfg.stdp_w_min
        self.reward_scale=cfg.stdp_reward_scale
        self.allowed=set(cfg.stdp_layers or [])
        self._loss_ema=10.0; self._ema_decay=0.99; self.max_update_norm=0.01

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
        if name and not self.is_allowed(name): return
        if pre.dim()==3: pre=pre.mean(dim=1)
        if post.dim()==3: post=post.mean(dim=1)
        dW=self.compute_stdp_update(pre,post)
        if cl is not None:
            r=self._compute_reward(cl); dW=dW*(2.0*r-1.0); self.update_reward(cl)
        o,i=layer.weight.shape; dW=dW[:o,:i]
        layer.weight.data=(layer.weight.data+dW).clamp(self.w_min,self.w_max)

# ═══════════════════════════════════════════════════════════════════════════════
# §11  NORD MODEL v4.1
# ═══════════════════════════════════════════════════════════════════════════════
class NordModel(nn.Module):
    def __init__(self,cfg:NordConfig):
        super().__init__(); self.cfg=cfg
        self.encoder=TemporalSpikeEncoder(cfg)
        self.input_lif=AssociativeLIF(cfg.d_model,cfg,persistent=cfg.persistent_mem)
        self.sensory_blocks=nn.ModuleList([NordBlock(cfg,i,False,zone="sensory") for i in range(cfg.sensory_layers)])
        self.association_blocks=nn.ModuleList([NordBlock(cfg,cfg.sensory_layers+i,True,zone="association") for i in range(cfg.association_layers)])
        self.memory_cortex=MemoryCortex(cfg)
        self.executive_blocks=nn.ModuleList([NordBlock(cfg,cfg.sensory_layers+cfg.association_layers+i,False,zone="executive") for i in range(cfg.executive_layers)])
        self.readout_lif=AssociativeLIF(cfg.d_model,cfg,persistent=cfg.persistent_mem)
        self.readout_ema_raw=nn.Parameter(torch.tensor(1.4))
        self.readout_norm=nn.LayerNorm(cfg.d_model)
        self.lm_head=nn.Linear(cfg.d_model,cfg.vocab_size,bias=False)
        self.stdp=STDPEngine(cfg); self._last_loss=None
        self.spike_regulator=AuxiliarySpikeRegulator(cfg)

    @property
    def readout_ema_decay(self)->Tensor: return torch.sigmoid(self.readout_ema_raw)
    def reset_state(self):
        self.input_lif.reset_state(); self.readout_lif.reset_state()
        self.memory_cortex.reset_state()

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

        for i,bl in enumerate(self.executive_blocks):
            x,bs=bl(x); spike_ts.append(x)
            for k,v in bs.items(): stats[f"exec_{i}_{k}"]=v

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