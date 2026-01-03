from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

def _get(tree: Any, *path: str) -> Any:
    x = tree
    for p in path:
        if isinstance(x, dict):
            x = x[p]
        elif isinstance(x, (list, tuple)):
            x = x[int(p)]
        else:
            raise TypeError(f"Cannot descend into type {type(x)} at key {p}")
    return x


def rmsnorm(x: jnp.ndarray, weight: jnp.ndarray, eps: float) -> jnp.ndarray:
    x2 = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x_norm = x * jax.lax.rsqrt(x2 + eps)
    return x_norm * weight


def silu(x: jnp.ndarray) -> jnp.ndarray:
    return x * jax.nn.sigmoid(x)


def linear(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray | None = None) -> jnp.ndarray:
    y = x @ w
    if b is not None:
        y = y + b
    return y

def rope_frequencies(cfg, positions: jnp.ndarray) -> jnp.ndarray:
    half = cfg.head_dim // 2
    inv_freq = cfg.rope_theta ** (-jnp.arange(0, half, dtype=jnp.float32) / half)
    
    freqs = jnp.einsum("s,d->sd", positions.astype(jnp.float32), inv_freq)
    
    return freqs


def apply_rope(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    x_dtype = x.dtype
    freqs = freqs.astype(jnp.float32)  # keep stable trig in fp32

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    cos = jnp.cos(freqs)[None, :, None, :].astype(x_dtype)
    sin = jnp.sin(freqs)[None, :, None, :].astype(x_dtype)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos

    y = jnp.empty_like(x)
    y = y.at[..., 0::2].set(y1)
    y = y.at[..., 1::2].set(y2)
    
    return y

def causal_mask(seq_len: int) -> jnp.ndarray:
    m = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    return m[None, None, :, :]


def attention_gqa(cfg, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    b, s, h, d = q.shape
    kv = k.shape[2]
    g = cfg.q_per_kv

    qg = q.reshape(b, s, kv, g, d)

    scores = jnp.einsum("bskgd,btkd->bkgst", qg, k) / jnp.sqrt(d).astype(q.dtype)

    neg_inf = jnp.array(-1e30, dtype=scores.dtype)
    scores = jnp.where(mask[:, :, None, :, :], scores, neg_inf)

    probs = jax.nn.softmax(scores, axis=-1)

    ctx = jnp.einsum("bkgst,btkd->bskgd", probs, v)
    ctx = ctx.reshape(b, s, h, d)
    return ctx

def transformer_block(cfg, params: Any, x: jnp.ndarray, freqs: jnp.ndarray, mask: jnp.ndarray, layer_idx: int) -> jnp.ndarray:
    attn_norm_w = _get(params, "model", "layers", str(layer_idx), "input_layernorm", "weight")
    x_norm = rmsnorm(x, attn_norm_w, cfg.rms_norm_eps)

    wq = _get(params, "model", "layers", str(layer_idx), "self_attn", "q_proj", "kernel")
    wk = _get(params, "model", "layers", str(layer_idx), "self_attn", "k_proj", "kernel")
    wv = _get(params, "model", "layers", str(layer_idx), "self_attn", "v_proj", "kernel")
    wo = _get(params, "model", "layers", str(layer_idx), "self_attn", "o_proj", "kernel")

    q = linear(x_norm, wq)
    k = linear(x_norm, wk)
    v = linear(x_norm, wv)

    b, s, _ = q.shape
    q = q.reshape(b, s, cfg.num_heads, cfg.head_dim)
    k = k.reshape(b, s, cfg.num_kv_heads, cfg.head_dim)
    v = v.reshape(b, s, cfg.num_kv_heads, cfg.head_dim)

    q = apply_rope(q, freqs)
    k = apply_rope(k, freqs)

    ctx = attention_gqa(cfg, q, k, v, mask)
    ctx = ctx.reshape(b, s, cfg.hidden_size)
    attn_out = linear(ctx, wo)

    x = x + attn_out

    ffn_norm_w = _get(params, "model", "layers", str(layer_idx), "post_attention_layernorm", "weight")
    x_norm2 = rmsnorm(x, ffn_norm_w, cfg.rms_norm_eps)

    w_gate = _get(params, "model", "layers", str(layer_idx), "mlp", "gate_proj", "kernel")
    w_up   = _get(params, "model", "layers", str(layer_idx), "mlp", "up_proj", "kernel")
    w_down = _get(params, "model", "layers", str(layer_idx), "mlp", "down_proj", "kernel")

    gate = linear(x_norm2, w_gate)
    up   = linear(x_norm2, w_up)
    ff   = linear(silu(gate) * up, w_down)

    x = x + ff
    return x

def forward(cfg, params: Any, input_ids: jnp.ndarray) -> jnp.ndarray:
    b, s = input_ids.shape

    embed = _get(params, "model", "embed_tokens", "embedding")
    x = embed[input_ids]

    positions = jnp.arange(s, dtype=jnp.int32)
    freqs = rope_frequencies(cfg, positions)
    mask = causal_mask(s)

    for i in range(cfg.num_layers):
        x = transformer_block(cfg, params, x, freqs, mask, i)

    final_norm_w = _get(params, "model", "norm", "weight")
    x = rmsnorm(x, final_norm_w, cfg.rms_norm_eps)

    logits = jnp.einsum("bsh,vh->bsv", x, embed)

    return logits
