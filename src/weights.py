from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.serialization
import jax
import jax.numpy as jnp

@dataclass(frozen=True, slots=True)
class WeightsBundle:
    model_dir: Path
    params: Any

    @staticmethod
    def from_model_dir(model_dir: str | Path) -> "WeightsBundle":
        model_dir = Path(model_dir)
        path = model_dir / "flax_model.msgpack"

        data = path.read_bytes()
        params = flax.serialization.from_bytes(None, data)

        return WeightsBundle(model_dir=model_dir, params=params)

    def cast(self, dtype: jnp.dtype) -> "WeightsBundle":
        params2 = jax.tree_util.tree_map(
            lambda x: x.astype(dtype) if hasattr(x, "astype") else x,
            self.params,
        )
        return WeightsBundle(model_dir=self.model_dir, params=params2)