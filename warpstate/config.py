from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class WarpStateConfig:
    vocab_size: int = 32768
    d_model: int = 1280
    n_heads: int = 20
    n_cores: int = 4
    logical_depth: int = 16
    ffn_hidden: int = 4480
    chunk_size: int = 128
    dropout: float = 0.0
    rms_eps: float = 1e-5
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        assert self.d_model % self.n_heads == 0
        assert self.logical_depth >= self.n_cores
        assert self.chunk_size > 0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "WarpStateConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
