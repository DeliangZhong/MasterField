"""config.py — Hyperparameter dataclasses for TEK training runs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TEKConfig:
    """TEK model configuration.

    N must equal L² with L prime (the TEK reduction requires this). The twist
    flux k defaults to 1 (symmetric twist); for D=4 a different k may be
    needed to avoid center-symmetry breaking.
    """

    D: int = 2  # spacetime dimension: 2, 3, or 4
    N: int = 49  # matrix size (must equal L²)
    L: int = 7  # N = L²; L should be prime for the standard TEK construction
    k: int = 1  # twist flux integer (n_μν = k·L on twisted planes)
    twist: bool = True  # if False, untwisted EK for Phase B


@dataclass(frozen=True)
class OptConfig:
    """Optimizer hyperparameters."""

    n_steps: int = 3000
    lr: float = 1e-2
    warmup: int = 200
    grad_clip: float = 1.0
    tol: float = 1e-8
    log_every: int = 100
    seed: int = 42


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration."""

    model: str = "tek"  # gw, ek, tek
    lam: float = 1.0  # single-λ run
    schedule: list[float] = field(
        default_factory=lambda: [20.0, 10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.3]
    )  # continuation schedule (strong → weak)
    tek: TEKConfig = field(default_factory=TEKConfig)
    opt: OptConfig = field(default_factory=OptConfig)
    output_dir: str = "results"
    validate: bool = False
    verbose: bool = True
