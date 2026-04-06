# Master Field — Cluster Package

Standalone package for running master field ML computations on a PBS cluster.

## Quick Start

```bash
# Submit two-matrix coupled training (Stage 3)
qsub submit.pbs

# Submit scaling study L=4,6,8 (Stage 4)
qsub submit_scaling.pbs
```

## Setup

Edit the `module load` lines in the PBS scripts to match your cluster.
The scripts auto-create a virtual environment on first run.

For GPU clusters, uncomment `module load cuda` and remove `JAX_PLATFORM_NAME=cpu`.

## Files

| File | Purpose |
|------|---------|
| `train.py` | CLI entry point |
| `neural_master_field.py` | JAX training loops (MasterFieldTrainer, MultiMatrixTrainer) |
| `cuntz_fock.py` | Cuntz algebra, Fock space operators |
| `one_matrix.py` | Exact one-matrix solutions (validation) |
| `schwinger_dyson.py` | Schwinger-Dyson equations |
| `bootstrap_sdp.py` | SDP bootstrap bounds (requires cvxpy) |
| `visualize.py` | Plotting (requires matplotlib) |
| `config.py` | Hyperparameter dataclasses |
| `test_master_field.py` | Test suite (`python3 test_master_field.py`) |

## Key Commands

```bash
# Two coupled matrices (Stage 3)
python3 train.py --model two_matrix_coupled --coupling 1.0 \
    --max_word_length 6 --n_epochs 20000 --lr 5e-4 \
    --interaction commutator_squared

# One-matrix quartic validation (Stage 1b)
python3 train.py --model quartic --coupling 0.5 --validate \
    --max_word_length 12

# Run tests
python3 test_master_field.py
```

## PBS Parameters

| Parameter | `submit.pbs` | `submit_scaling.pbs` |
|-----------|-------------|---------------------|
| CPUs | 8 | 8 |
| Memory | 32 GB | 64 GB |
| Walltime | 24h | 48h |

For L=8+ two-matrix, memory grows as ~2^L. Consider 64-128 GB for L≥8.
