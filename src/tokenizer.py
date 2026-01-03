from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from transformers import AutoTokenizer

from utils import _as_int_list, _flatten_ids

Role = Literal["system", "user", "assistant"]

@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str

@dataclass(frozen=True, slots=True)
class TokenizerBundle:
    model_dir: Path
    tok: Any

    bos_token_id: int | None
    eos_token_ids: tuple[int, ...]
    pad_token_id: int | None

    @staticmethod
    def from_model_dir(model_dir: str | Path) -> "TokenizerBundle":
        model_dir = Path(model_dir)

        tok = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            local_files_only=True,
        )

        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        eos_ids = _as_int_list(getattr(tok, "eos_token_id", None))

        return TokenizerBundle(
            model_dir=model_dir,
            tok=tok,
            bos_token_id=getattr(tok, "bos_token_id", None),
            eos_token_ids=eos_ids,
            pad_token_id=getattr(tok, "pad_token_id", None),
        )

    def encode_text(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def encode_chat(
        self,
        messages: Sequence[Message],
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:

        ids = self.tok.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in messages],
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors=None,
        )

        return _flatten_ids(ids)