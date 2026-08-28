from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import WarpStateConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x32.to(dtype) * self.weight.to(dtype))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_cos_sin(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(length, device=device, dtype=torch.float32)
    inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    freqs = torch.outer(pos, inv)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, N, H, C, Dh], cos/sin: [C, Dh]
    cos = cos.view(1, 1, 1, cos.shape[0], cos.shape[1])
    sin = sin.view(1, 1, 1, sin.shape[0], sin.shape[1])
    return x * cos + rotate_half(x) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up_gate = nn.Linear(dim, hidden * 2, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.up_gate(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


@dataclass
class LayerCache:
    fast_state: torch.Tensor
    slow_state: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    write_accum: torch.Tensor
    chunk_len: int = 0


class WarpStateCore(nn.Module):
    """
    GPU-first experimental block.

    The local path performs causal attention only inside a fixed-size tile.
    The long-range path compresses completed tiles into two constant-size
    tensor states (fast + slow). A single fused input projection produces
    Q/K/V/gate/U for both paths.
    """

    def __init__(self, cfg: WarpStateConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        h = cfg.n_heads
        self.h = h
        self.hd = cfg.head_dim
        self.chunk = cfg.chunk_size

        self.norm1 = RMSNorm(d, cfg.rms_eps)
        self.in_proj = nn.Linear(d, 5 * d, bias=False)  # Q K V gate memory-U
        self.out_proj = nn.Linear(d, d, bias=False)

        self.norm2 = RMSNorm(d, cfg.rms_eps)
        self.ffn = SwiGLU(d, cfg.ffn_hidden)

        # Per-head timescales. Start with a quick and a slow memory.
        self.fast_decay_logit = nn.Parameter(torch.full((h,), math.log(0.90 / 0.10)))
        self.slow_decay_logit = nn.Parameter(torch.full((h,), math.log(0.99 / 0.01)))
        self.memory_mix_logit = nn.Parameter(torch.zeros(h))

    def _split(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, t, _ = z.shape
        q, k, v, g, u = self.in_proj(z).chunk(5, dim=-1)
        def heads(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, t, self.h, self.hd)
        return heads(q), heads(k), heads(v), heads(g), heads(u)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if t % self.chunk != 0:
            raise ValueError(f"sequence length {t} must be divisible by chunk_size={self.chunk}")
        n_chunks = t // self.chunk

        z = self.norm1(x)
        q, k, v, g, u = self._split(z)

        # [B, N, H, C, Dh]
        def chunkify(y: torch.Tensor) -> torch.Tensor:
            return y.view(b, n_chunks, self.chunk, self.h, self.hd).permute(0, 1, 3, 2, 4).contiguous()

        q = chunkify(q)
        k = chunkify(k)
        v = chunkify(v)
        g = chunkify(g)
        u = chunkify(u)

        cos, sin = rope_cos_sin(self.chunk, self.hd, x.device, q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Local tile path: all tiles are batched together, so this is a single
        # large SDPA workload rather than token-by-token attention.
        q_local = q.reshape(b * n_chunks, self.h, self.chunk, self.hd)
        k_local = k.reshape(b * n_chunks, self.h, self.chunk, self.hd)
        v_local = v.reshape(b * n_chunks, self.h, self.chunk, self.hd)
        local = F.scaled_dot_product_attention(
            q_local,
            k_local,
            v_local,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=True,
        ).view(b, n_chunks, self.h, self.chunk, self.hd)

        # Long-range tensor memory. Memory read uses only previous completed
        # tiles; the current tile is written after its outputs are formed.
        fast = torch.zeros((b, self.h, self.hd, self.hd), device=x.device, dtype=q.dtype)
        slow = torch.zeros_like(fast)
        fast_decay = torch.sigmoid(self.fast_decay_logit).to(q.dtype).view(1, self.h, 1, 1)
        slow_decay = torch.sigmoid(self.slow_decay_logit).to(q.dtype).view(1, self.h, 1, 1)
        memory_mix = torch.sigmoid(self.memory_mix_logit).to(q.dtype).view(1, self.h, 1, 1)

        out_chunks: list[torch.Tensor] = []
        for i in range(n_chunks):
            qi = q[:, i]
            ki = k[:, i]
            ui = u[:, i]
            gi = torch.sigmoid(g[:, i])

            read_fast = torch.matmul(qi, fast) / math.sqrt(self.hd)
            read_slow = torch.matmul(qi, slow) / math.sqrt(self.hd)
            memory = memory_mix * read_fast + (1.0 - memory_mix) * read_slow

            yi = gi * local[:, i] + (1.0 - gi) * memory
            out_chunks.append(yi)

            # B,H,D,C @ B,H,C,D -> B,H,D,D. tanh bounds the write and makes
            # early training considerably more stable than an unbounded outer product.
            write = torch.matmul(
                torch.tanh(ki).transpose(-2, -1),
                torch.tanh(ui),
            ) / float(self.chunk)
            fast = fast_decay * fast + (1.0 - fast_decay) * write
            slow = slow_decay * slow + (1.0 - slow_decay) * write

        y = torch.stack(out_chunks, dim=1)  # [B,N,H,C,Dh]
        y = y.permute(0, 1, 3, 2, 4).contiguous().view(b, t, d)
        x = x + self.out_proj(y)
        x = x + self.ffn(self.norm2(x))
        return x

    @torch.no_grad()
    def init_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> LayerCache:
        state_shape = (batch_size, self.h, self.hd, self.hd)
        kv_shape = (batch_size, self.h, self.chunk, self.hd)
        return LayerCache(
            fast_state=torch.zeros(state_shape, device=device, dtype=dtype),
            slow_state=torch.zeros(state_shape, device=device, dtype=dtype),
            k_cache=torch.zeros(kv_shape, device=device, dtype=dtype),
            v_cache=torch.zeros(kv_shape, device=device, dtype=dtype),
            write_accum=torch.zeros(state_shape, device=device, dtype=dtype),
            chunk_len=0,
        )

    @torch.no_grad()
    def step(self, x: torch.Tensor, cache: LayerCache) -> torch.Tensor:
        # x: [B,1,D]
        assert x.shape[1] == 1
        b = x.shape[0]
        z = self.norm1(x)
        q, k, v, g, u = self._split(z)
        q = q.permute(0, 2, 1, 3)  # B,H,1,Dh
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        g = g.permute(0, 2, 1, 3)
        u = u.permute(0, 2, 1, 3)

        pos = cache.chunk_len
        cos, sin = rope_cos_sin(pos + 1, self.hd, x.device, q.dtype)
        c = cos[pos : pos + 1].view(1, 1, 1, self.hd)
        s = sin[pos : pos + 1].view(1, 1, 1, self.hd)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s

        cache.k_cache[:, :, pos : pos + 1].copy_(k)
        cache.v_cache[:, :, pos : pos + 1].copy_(v)
        keys = cache.k_cache[:, :, : pos + 1]
        vals = cache.v_cache[:, :, : pos + 1]
        local = F.scaled_dot_product_attention(q, keys, vals, is_causal=False, dropout_p=0.0)

        fast_decay = torch.sigmoid(self.fast_decay_logit).to(q.dtype).view(1, self.h, 1, 1)
        slow_decay = torch.sigmoid(self.slow_decay_logit).to(q.dtype).view(1, self.h, 1, 1)
        memory_mix = torch.sigmoid(self.memory_mix_logit).to(q.dtype).view(1, self.h, 1, 1)

        read_fast = torch.matmul(q, cache.fast_state) / math.sqrt(self.hd)
        read_slow = torch.matmul(q, cache.slow_state) / math.sqrt(self.hd)
        memory = memory_mix * read_fast + (1.0 - memory_mix) * read_slow
        gate = torch.sigmoid(g)
        y = gate * local + (1.0 - gate) * memory

        cache.write_accum.add_(torch.matmul(torch.tanh(k).transpose(-2, -1), torch.tanh(u)))

        y = y.permute(0, 2, 1, 3).contiguous().view(b, 1, self.cfg.d_model)
        x = x + self.out_proj(y)
        x = x + self.ffn(self.norm2(x))

        if pos + 1 == self.chunk:
            write = cache.write_accum / float(self.chunk)
            cache.fast_state.mul_(fast_decay).add_(write * (1.0 - fast_decay))
            cache.slow_state.mul_(slow_decay).add_(write * (1.0 - slow_decay))
            cache.k_cache.zero_()
            cache.v_cache.zero_()
            cache.write_accum.zero_()
            cache.chunk_len = 0
        else:
            cache.chunk_len = pos + 1
        return x


class WarpStateLM(nn.Module):
    def __init__(self, cfg: WarpStateConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.cores = nn.ModuleList([WarpStateCore(cfg) for _ in range(cfg.n_cores)])

        # Tiny depth-specific modulation lets shared cores behave differently on
        # each recurrent-depth pass without duplicating the large matrices.
        self.depth_scale = nn.Parameter(torch.zeros(cfg.logical_depth, cfg.d_model))
        self.depth_bias = nn.Parameter(torch.zeros(cfg.logical_depth, cfg.d_model))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

        self.apply(self._init_weights)
        # Smaller residual projections are useful because the same cores are reused.
        scale = 1.0 / math.sqrt(2.0 * cfg.logical_depth)
        for core in self.cores:
            nn.init.normal_(core.out_proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(core.ffn.down.weight, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(input_ids)
        for depth in range(self.cfg.logical_depth):
            x = x * (1.0 + self.depth_scale[depth]) + self.depth_bias[depth]
            core = self.cores[depth % self.cfg.n_cores]
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(core, x, use_reentrant=False)
            else:
                x = core(x)
        x = self.final_norm(x)
        logits = F.linear(x, self.embed.weight)  # tied LM head
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def init_generation_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> List[LayerCache]:
        caches: list[LayerCache] = []
        for depth in range(self.cfg.logical_depth):
            core = self.cores[depth % self.cfg.n_cores]
            caches.append(core.init_cache(batch_size, device, dtype))
        return caches

    @torch.no_grad()
    def step(self, input_ids: torch.Tensor, caches: List[LayerCache]) -> torch.Tensor:
        # input_ids [B,1], returns logits [B,1,V]
        x = self.embed(input_ids)
        for depth in range(self.cfg.logical_depth):
            x = x * (1.0 + self.depth_scale[depth]) + self.depth_bias[depth]
            core = self.cores[depth % self.cfg.n_cores]
            x = core.step(x, caches[depth])
        x = self.final_norm(x)
        return F.linear(x, self.embed.weight)
