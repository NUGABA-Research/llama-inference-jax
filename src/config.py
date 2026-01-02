from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json

def _as_int_list(x: Any) -> tuple[int, ...]:
    if x is None:
        return tuple()
    
    if isinstance(x, int):
        return (x,)
    
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    
    raise TypeError(f"Expected int or list/tuple of ints, got: {type(x)}")

@dataclass(frozen=True, slots=True)
class Config:
    model_dir: Path

    # Core architecture
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int

    # Sequence / positions
    max_position_embeddings: int

    # Normalization
    rms_norm_eps: float

    # RoPE
    rope_theta: float
    rope_scaling: Mapping[str, Any] | None
    rope_type: str | None

    # Special tokens
    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]

    # Derived values
    head_dim: int
    q_per_kv: int

    @staticmethod
    def from_model_dir(model_dir: str | Path) -> "Config":
        model_dir = Path(model_dir)
        cfg_path = model_dir / "config.json"

        with cfg_path.open("r", encoding="utf-8") as f: 
            raw = json.load(f)

        vocab_size = int(raw["vocab_size"])
        hidden_size = int(raw["hidden_size"])
        num_layers = int(raw["num_hidden_layers"])
        num_heads = int(raw["num_attention_heads"])
        num_kv_heads = int(raw.get("num_key_value_heads", num_heads))
        intermediate_size = int(raw["intermediate_size"])

        max_position_embeddings = int(raw["max_position_embeddings"])
        rms_norm_eps = float(raw["rms_norm_eps"])

        rope_theta = float(raw["rope_theta"])
        rope_scaling = raw["rope_scaling"]
        rope_type = raw["rope_type"]

        bos_token_id = raw["bos_token_id"]
        eos_token_ids = _as_int_list(raw["eos_token_id"])

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})")
        head_dim = hidden_size // num_heads
        
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        q_per_kv = num_heads // num_kv_heads
        
        return Config(
            model_dir=model_dir,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            rope_type=rope_type,
            bos_token_id=int(bos_token_id) if bos_token_id is not None else None,
            eos_token_ids=eos_token_ids,
            head_dim=head_dim,
            q_per_kv=q_per_kv,
        )
    
    @staticmethod
    def summarize(cfg: "Config") -> str:
        return (
            "Config("
            f"layers={cfg.num_layers}, hidden={cfg.hidden_size}, heads={cfg.num_heads}, "
            f"kv_heads={cfg.num_kv_heads}, head_dim={cfg.head_dim}, vocab={cfg.vocab_size}, "
            f"max_pos={cfg.max_position_embeddings}, rope_theta={cfg.rope_theta}, "
            f"eos={list(cfg.eos_token_ids)}"
            ")"
        )
            