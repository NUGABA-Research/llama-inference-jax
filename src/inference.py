from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import jax.numpy as jnp

from config import Config
from tokenizer import Message, TokenizerBundle
from weights import WeightsBundle
from model import forward


@dataclass(frozen=True, slots=True)
class GenConfig:
    max_new_tokens: int = 64
    temperature: float = 0.1


def _to_jnp_int32(x: Sequence[int]) -> jnp.ndarray:
    return jnp.asarray(x, dtype=jnp.int32)[None, :]


def greedy_generate(
    cfg: Config,
    params,
    tok: TokenizerBundle,
    input_ids: list[int],
    gen_cfg: GenConfig,
) -> list[int]:
    ids = list(input_ids)

    for _ in range(gen_cfg.max_new_tokens):
        x = _to_jnp_int32(ids)
        logits = forward(cfg, params, x)
        next_logits = logits[0, -1]
        next_id = int(jnp.argmax(next_logits))

        ids.append(next_id)

        if next_id in tok.eos_token_ids:
            break

    return ids


def main() -> None:
    model_dir = Path("models/llama3_2_1b_instruct_flax")

    cfg = Config.from_model_dir(model_dir)
    tb = TokenizerBundle.from_model_dir(model_dir)
    wb = WeightsBundle.from_model_dir(model_dir)

    params = wb.params

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain what JAX is in one paragraph."),
    ]

    input_ids = tb.encode_chat(messages, add_generation_prompt=True)

    gen_cfg = GenConfig(max_new_tokens=80, temperature=0.0)
    out_ids = greedy_generate(cfg, params, tb, input_ids, gen_cfg)

    text = tb.decode(out_ids)
    print(text)


if __name__ == "__main__":
    main()
