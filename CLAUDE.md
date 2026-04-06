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

# Tests
python master_field/test_master_field.py

# Training
python master_field/train.py --model gaussian --validate
python master_field/train.py --model quartic --coupling 0.5 --validate
python master_field/train.py --model two_matrix_coupled --coupling 1.0 --max_word_length 6
```

## Critical Physics Gotchas

- **SD indexing**: LHS is `Σ_k v_k m_{n+k}`, NOT `m_{n+k-1}`. From tr[M^k · M^n] = m_{k+n}.
- **PSD enforcement**: Cholesky diagonal must use `exp(ℓ_i)`, not raw parameters.
- **Float64 required**: Add `jax.config.update("jax_enable_x64", True)` in every file using JAX.
- **Normalisation**: m₀ = 1 is a hard constraint, not optimisable.
- **Cyclic equivalence**: Reduce loop moments to lexicographically smallest rotation.
- **Odd moments**: For symmetric potentials (V(M)=V(-M)), enforce odd-moment vanishing by construction.
- **Large-L divergence**: Use relative residuals; moments grow factorially.

## Full Context

@discussion_AI.md
