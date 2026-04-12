"""Frozen dataclass configs for cuntz_bootstrap runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FockConfig:
    D: int
    L_trunc: int
    include_mixed: bool = False

    @property
    def n_labels(self) -> int:
        return 2 * self.D


@dataclass(frozen=True)
class OptConfig:
    n_steps: int = 3000
    lr: float = 1e-2
    warmup: int = 200
    grad_clip: float = 1.0
    tol: float = 1e-10
    log_every: int = 100
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    phase: str
    lam: float
    L_max_loops: int
    fock: FockConfig
    opt: OptConfig
    w_unit: float = 1.0
    w_mm: float = 1.0
    w_sup: float = 0.0
    output_dir: Path = Path("results")
    validate: bool = True
    verbose: bool = True
