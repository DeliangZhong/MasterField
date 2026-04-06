"""
config.py — Hyperparameters and model definitions for master field ML.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Which matrix model to solve."""
    name: str = "gaussian"           # gaussian, quartic, sextic, two_matrix_coupled, yang_mills_qm, gross_witten
    n_matrices: int = 1              # number of matrix variables
    coupling: float = 0.0            # interaction coupling g
    mass: float = 1.0                # mass parameter (coefficient of Tr M^2)
    max_word_length: int = 10        # truncation level L for loop space
    potential_type: str = "polynomial"  # polynomial or yang_mills


@dataclass
class OptimConfig:
    """Optimisation hyperparameters."""
    learning_rate: float = 1e-3
    n_epochs: int = 5000
    batch_size: int = 64             # number of SD equations sampled per step
    optimizer: str = "adam"          # adam, lbfgs, natural_gradient
    scheduler: str = "cosine"       # cosine, exponential, constant
    warmup_steps: int = 200
    sd_loss_weight: float = 1.0
    eom_loss_weight: float = 0.1
    psd_barrier_weight: float = 0.01  # log-barrier for PSD (if not using Cholesky)
    grad_clip: float = 1.0
    seed: int = 42


@dataclass
class NetworkConfig:
    """Neural network architecture."""
    hidden_dim: int = 256
    n_layers: int = 4
    activation: str = "gelu"
    use_cholesky: bool = True        # enforce PSD via Cholesky parametrisation
    use_residual: bool = True        # residual connections
    dropout: float = 0.0
    # For the R-transform flow (Direction 2)
    n_flow_layers: int = 6
    flow_hidden_dim: int = 128


@dataclass 
class FullConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    output_dir: str = "results"
    device: str = "cpu"              # cpu or gpu
    validate: bool = False           # run validation against exact solutions
    verbose: bool = True
