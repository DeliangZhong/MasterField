# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Large-N matrix model master field computation. Numerically construct the master field (N=∞ saddle point) using ML optimisation. Physics from Gopakumar–Gross (hep-th/9411021).

## Setup

```bash
pip install jax jaxlib numpy scipy matplotlib optax flax cvxpy torch --break-system-packages
```

## Commands

```bash
# Sanity checks (run first)
python master_field/cuntz_fock.py
python master_field/one_matrix.py
python master_field/schwinger_dyson.py

# Tests (10 tests, all must pass)
python master_field/test_master_field.py

# One-matrix training
python master_field/train.py --model gaussian --validate
python master_field/train.py --model quartic --coupling 0.5 --validate --max_word_length 12

# Two-matrix coupled training
python master_field/train.py --model two_matrix_coupled --coupling 1.0 --max_word_length 6 --n_epochs 20000 --lr 5e-4
```

## Architecture

- **One-matrix models**: scipy SLSQP with Hankel PSD constraint (SD equations are underdetermined without PSD)
- **Multi-matrix models**: JAX Cholesky parametrization + JIT-compiled gradient descent with pre-computed word→index arrays
- **Symmetry constraints** (multi-matrix): cyclicity, M₁↔M₂ exchange, Z₂ (M→-M) enforced in loss
- **Unitary master field (QCD₂ Phase 0)**: `cuntz_fock.build_unitary_gaussian` + `lattice.py` + `qcd2.py`. Single-α Gaussian ansatz Û_μ = exp(iα(â+â†)) matches plaquette by construction; larger Wilson loops deviate (motivates Phase 1 ML parametrization). See `reference/qcd2_master_field.md`.

## Cluster

Standalone package in `cluster/` with PBS scripts. Uses EasyBuild modules:
```bash
module load jax/0.4.25-gfbf-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
```

## Critical Physics Gotchas

- **SD indexing**: LHS is `Σ_k v_k m_{n+k}`, NOT `m_{n+k-1}`. From tr[M^k · M^n] = m_{k+n}.
- **PSD enforcement**: Cholesky diagonal must use `exp(ℓ_i)`, not raw parameters.
- **Float64 required**: `jax.config.update("jax_enable_x64", True)` at top of every JAX file.
- **Normalisation**: m₀ = 1 is a hard constraint, not optimisable.
- **Cyclic equivalence**: Reduce loop moments to lexicographically smallest rotation.
- **Odd moments**: For symmetric potentials (V(M)=V(-M)), enforce odd-moment vanishing by construction.
- **Large-L divergence**: Use relative residuals; moments grow factorially.
- **Eigenvalue density**: For quartic V'=M+gM³, correct density is ρ(x)=(gx²+1+ga²/2)√(a²-x²)/(2π).
- **Free cumulants**: Use recursive formula via z·G=1+Σκ_k G^k, NOT hardcoded classical formulas.

## Full Context

@discussion_AI.md
