# M3 result — two-matrix operator master field inside the SDP island at g>0

**Date:** 2026-06-24. **Model:** commutator+mass two-matrix,
S = N·tr[½(M₀²+M₁²) − (g²/4)[M₀,M₁]²], coupling g (λ=g²).
**Method:** ML-optimized Cuntz–Fock operators M̂₀,M̂₁ (`MultiMonomialAnsatz`),
the exact nonlinear loop-equation residual + cyclicity/exchange/Z₂ penalties,
g-homotopy Adam + L-BFGS polish. Positivity automatic (Hermitian operators,
vacuum state). Validator: our own rigorous SDP island `bootstrap_two_matrix`
(L=6) + the g=0 free limit. Engine entry point: `train.solve_two_matrix`.

## Headline: the g>0 solve now lands inside the rigorous island

The M3 frontier was that the g>0 operator solve parked **below** the SDP lower
bound (tr M₀²≈0.5548 < lb 0.618 at g=1) and was correctly flagged
`validated=False` by the fail-closed gate. Diagnosed root-cause-first: the cause
is **ansatz under-expressiveness**, not optimization budget and not the soft
symmetry constraints (those are already exactly 0; `w_sym`∈{10,50,200} and full
L-BFGS polish do not move the result). The degree-2 SD residual **floors** at
~1e-3 and the moment sits below the rigorous lower bound at *every* g≥0.2.

**Fix:** ansatz **degree 3**, with the Fock cutoff at the exactness bound
`⌊(max_word_len+3)/2⌋·degree`.

| g | degree-2 tr M₀² (sd_loss) | degree-3 tr M₀² (sd_loss) | SDP island [lb,ub] (L=6) |
|---|---|---|---|
| 0.2 | 0.9513 (8.5e-5) ✗ below | — | [0.9629, 1.000] |
| 0.5 | 0.7770 (1.1e-3) ✗ below | **0.8537 (5e-22)** ✓ in | [0.8284, 1.000] |
| 0.7 | 0.6724 (1.9e-3) ✗ below | — | [0.7352, 1.000] |
| 1.0 | 0.5548 (2.7e-3) ✗ below | **0.6938 (3e-16)** ✓ in | [0.6180, 1.000] |

(degree-3 rows: max_word_len=3, cutoff=9, Fock dim 1023.)

- **Degree-2** floors the residual and sits below the rigorous lower bound at all
  g≥0.2; the gap grows with g (0.012 at g=0.2 → 0.063 at g=1).
- **Degree-3** solves the truncated loop equations to **machine zero**
  (cyclicity/exchange/Z₂ all exact) and lands **inside the island** for every
  g∈[0.3,1.0] — `validated=True`. This realizes the project thesis at strong
  coupling: an exact nonlinear solve with automatic positivity yields a physical
  moment inside the rigorous bracket, where the SDP relaxation alone gives only
  the interval.

A cheaper config (degree 3, max_word_len=2, cutoff 6, dim 127) also validates
across g∈[0.3,1.0] in ~30 s/solve (sd_loss exactly 0); it backs the fast
regression test `test_g_positive_solve_validated_at_degree3`.

## Truncation dependence (honest)

The exact-operator moment depends on the loop-equation truncation `max_word_len`.
At g=1: max_word_len=2/cutoff-6 gives tr M₀²=0.7840; max_word_len=3/cutoff-9
gives 0.6938 — **both inside [0.618, 1.0]**, moving toward the lower part of the
island as more loop equations are imposed. So "inside the island" is established;
pinning the precise value needs a `max_word_len`-convergence study (next step).

## Latent bug fixed

The truncation guard was degree-blind (`need = max_word_len+3`). A degree-d
ansatz letter changes the Cuntz quanta number by ±d, so a length-L
vacuum→vacuum amplitude reaches ⌊L/2⌋·d quanta and the exact-moment cutoff is
`⌊(max_word_len+3)/2⌋·degree`. Empirically the SD residual hits machine zero
exactly at this cutoff (degree-3/max_word_len-3 needs 9, not 6). The guard in
`solve_two_matrix` is now degree-aware (reads `ansatz.degree`); regression test
`test_truncation_guard_is_degree_aware`.

## Remaining (M3 refinements)

- **max_word_len-convergence study** of the in-island value at fixed g
  (extrapolate the precise moment; quantify the within-island drift).
- **Cross-ansatz agreement** (dense Hermitian vs monomial) at g>0 — the M3-plan's
  third spurious-solution guard, not yet run.
- Observables (Task 6): ρ(λ) of M̂₀ and Brown measure of M̂₀+iM̂₁ at g>0.
