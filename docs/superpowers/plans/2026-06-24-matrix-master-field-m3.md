# Matrix Master Field — Milestone 3 Implementation Plan (two-matrix)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. New physics code (multi-matrix SD residual, 2-matrix SDP) is built strictly test-first; code blocks are starting points.

**Goal:** Construct the master field of the "unsolvable" two-matrix commutator+mass model as ML-optimized Cuntz–Fock operators, validate it lands inside our own SDP island and converges in truncation, and *specifically stress-test the spurious-solution failure mode* that does not appear at one matrix.

**Architecture:** Reuse the M2 engine (JAX moment evaluator, Adam+L-BFGS, positivity automatic). Add: multi-matrix monomial ansatz, the commutator-model loop-equation residual, cyclicity/exchange/Z₂ losses, and a 2-matrix cvxpy SDP island as ground truth (there is NO exact g>0 solution — the island + the g=0 free limit are the validators).

**Tech Stack:** JAX/optax, NumPy, SciPy, cvxpy. Runner: `uv run --no-project --with numpy --with scipy --with jax --with optax --with cvxpy --with pytest python -m pytest …`.

## Global Constraints

- **Model (commutator+mass), pinned to match the validated `schwinger_dyson.TwoMatrixSD`:**
  S = N·tr[ ½(M₁²+M₂²) − (g²/4)[M₁,M₂]² ], coupling g (so λ≡g²). Derivative
  V′₁ = M₁ + (g²/2)(M₁M₂² + M₂²M₁ − 2 M₂M₁M₂), and 1↔2. (Sign verified by hand:
  ∂_{M₁} tr[M₁,M₂]² = −2(M₁M₂²+M₂²M₁−2M₂M₁M₂).)
- **Loop equation:** ⟨tr(V′_a·w)⟩ = Σ_{m: w[m]=a} ⟨tr w_left⟩⟨tr w_right⟩ (factorized, N=∞).
- **Validation anchors:** (a) g=0 → two *free* semicirculars, all mixed moments known (free Wick); (b) g>0 → our 2-matrix SDP island + convergence in Fock truncation. There is no closed-form g>0 answer.
- **Positivity automatic** (Hermitian operators, vacuum state). **Cyclicity is NOT automatic** on the Cuntz vacuum → imposed; plus **M₁↔M₂ exchange** and **Z₂ (M→−M)** symmetries.
- **Spurious-solution discipline (the M3 point):** a low residual is NOT success. Success = moments inside the SDP island AND stable under increasing truncation AND monomial/dense ansätze agree. If residual→0 but moments leave the island, that is the failure mode — record it, do not hide it.
- Validate INTERIOR moments (truncation-edge moments unconstrained); float64.

## Tasks

### Task 1: Multi-matrix monomial ansatz
**Files:** `matrix_master_field/ansatz.py` (add `MultiMonomialAnsatz`); test `tests/test_ansatz_multi.py`.
- M̂_i = Σ over word-monomials â†_u â_v (u,v words in the 2-letter alphabet, |u|+|v|≤degree) of real coeffs, Hermitized as monomial + adjoint. One independent coefficient set per matrix i.
- **Test contract:** (a) each M̂_i Hermitian; (b) degree-1 free config (M̂_i=â_i+â†_i) reproduces the free-Wick mixed moments via `word_moment` — τ(x1x2x1x2)=0, τ(x1²x2²)=1; (c) parameter count reported.

### Task 2: Commutator-model loop-equation residual (JAX)
**Files:** `matrix_master_field/loss.py` (add `two_matrix_sd_residual`); test `tests/test_loss_two_matrix.py`.
- Port the `TwoMatrixSD` equation structure: for each test word w and a∈{0,1}, LHS = moment of V′_a·w (kinetic + the 3 commutator terms with weight g²/2), RHS = factorized splits; relative-scaled MSE over equations. Mixed moments via `word_moment(ops, word)`.
- **Test contract:** at g=0 the residual evaluated on the free-field operators is ≤1e-9 (free Gaussians satisfy the decoupled SD); perturbing a moment makes it >1e-3.

### Task 3: Cyclicity + exchange + Z₂ losses
**Files:** `matrix_master_field/loss.py` (add `symmetry_losses`); test `tests/test_symmetry.py`.
- Cyclicity: ⟨Ω|AB|Ω⟩=⟨Ω|BA|Ω⟩ over a word basis (reuse `schwinger_dyson.cyclic_reduce` to pair words). Exchange: tr w(M₁,M₂)=tr w(M₂,M₁). Z₂: odd-total-degree word moments = 0.
- **Test contract:** the free-field operators satisfy all three to ≤1e-10; a hand-broken case is detected.

### Task 4: Two-matrix SDP island (ground truth)
**Files:** `matrix_master_field/bootstrap_sdp.py` (add `bootstrap_two_matrix`); test `tests/test_bootstrap_two_matrix.py`.
- cvxpy: moment matrix Ω over a 2-letter word basis ⪰ 0, linearized commutator-model SD equations, cyclicity/exchange/Z₂, m_∅=1; min/max a target moment (e.g. tr M₁²) at fixed g.
- **Test contract:** at g=0, bounds bracket the free value tr M₁²=1; at g>0 the island has positive but finite width (record it as the target the operator solution must fall inside).

### Task 5: Solve the two-matrix operator master field + spurious-solution stress test
**Files:** `matrix_master_field/train.py` (generalize `solve` to multi-matrix loss); test `tests/test_train_two_matrix.py`.
- Loss = two_matrix_sd_residual + w·(cyclicity+exchange+Z₂). Adam+L-BFGS, λ/g-homotopy from g=0 (free field) upward.
- **Test contract (the milestone):** at a test g (e.g. g=1), the monomial-ansatz solution's interior moments fall **inside the Task-4 SDP island**, are **stable** between two Fock truncations, and **agree** with the dense-ansatz solution within tolerance. Explicitly assert the in-island check — that is the spurious-solution guard.

### Task 6: Observables
**Files:** `matrix_master_field/observables.py`; test `tests/test_observables.py`.
- ρ(x) of M̂₁ (eigendecompose; weights |⟨Ω|v_i⟩|² — the spectral measure in the vacuum state). ⟨tr[M₁,M₂]²⟩ vs g (non-commutativity). Brown measure of M̂₁+iM̂₂ (stretch; flagged delicate).
- **Test contract:** at g=0, ρ(x) of M̂₁ matches the semicircle (moments ≤1e-3); ⟨tr[M₁,M₂]²⟩→0 as g→0.

## Self-Review
- Coverage: ansatz (T1) + physics loss (T2) + symmetries (T3) + ground-truth island (T4) + solve & the spurious-solution guard (T5) + observables (T6).
- The crux differs from M1/M2: there is no exact g>0 target, so validation = SDP island + g=0 free limit + truncation-convergence + cross-ansatz agreement. T5 asserts in-island explicitly.
- Placeholders: physics code is test-first with concrete contracts; the g=0 free limit is the always-available exact anchor.
