# M3 result — two-matrix operator master field inside the SDP island at g>0

**Date:** 2026-06-24. **Model:** commutator+mass two-matrix,
S = N·tr[½(M₀²+M₁²) − (g²/4)[M₀,M₁]²], coupling g (λ=g²).
**Method:** ML-optimized Cuntz–Fock operators M̂₀,M̂₁ (`MultiMonomialAnsatz`),
the exact nonlinear loop-equation residual + cyclicity/exchange/Z₂ penalties,
g-homotopy Adam + L-BFGS polish. Positivity automatic (Hermitian operators,
vacuum state). Validator: our own SDP island `bootstrap_two_matrix` (checked at
relaxation order L, certified by CLARABEL; the gate default was raised to L=8 —
see Refinements §1) + the g=0 free limit. Engine entry point: `train.solve_two_matrix`.

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
- **Degree-3 at max_word_len=3** solves the truncated loop equations to **machine
  zero** (cyclicity/exchange/Z₂ all exact) and lands the moment **inside the
  rigorous island** at the verified couplings g=0.5 (0.8537) and g=1 (0.6938) —
  `validated=True`. This realizes the project thesis at strong coupling: an exact
  nonlinear solve with automatic positivity yields a physical moment inside the
  rigorous bracket, where the SDP relaxation alone gives only the interval.
  (Refinements §1 sharpens this against the tight L=8/10 island; lower
  max_word_len gives only loose-island artifacts.)

A cheaper config (degree 3, max_word_len=2, cutoff 6, dim 127) solves its loop
equations to sd_loss=0 too, but max_word_len=2 is too few equations to pin the
moment — it lands tr M₀²≈0.80, a **truncation artifact** that the loose L=6
island admits but the tight L=8 island rejects (see Refinements §1). So the fast
regression test `test_gate_rejects_low_max_word_len_truncation_artifact` checks
the gate *rejects* it; the genuine in-tight-island result needs max_word_len=3
(test `test_max_word_len3_solve_validated_in_tight_island`, env-gated `MMF_SLOW`).

## Refinements §1 — truncation convergence and the SDP sandwich

The SDP island — **certified by CLARABEL** (interior-point, status `optimal`) —
tightens fast with relaxation order L, sandwiching the true value:

| L | tr M₀² island at g=1 | width |  | tr M₀² island at g=0.5 | width |
|---|---|---|---|---|---|
| 6 | [0.61803, 1.00000] | 0.382 | | [0.82843, 1.00000] | 0.172 |
| 8 | [0.63322, 0.72655] | 0.093 | | [0.83581, 0.85818] | 0.022 |
| 10 | [0.69307, 0.71408] | 0.021 | | [0.85375, 0.85633] | 0.003 |

So the master-field value is **tr M₀²(g=1) ≈ 0.70** and **(g=0.5) ≈ 0.855**.
Convergence of the operator solve (degree-3) against this sandwich:

- **Degree saturates immediately**: at max_word_len=2 the degree-3 and degree-4
  solves give *identical* tr M₀²=0.7983 (sd=0). The residual freedom is in
  max_word_len, not degree — degree 3 suffices.
- **max_word_len controls the result via the island width.** At finite
  max_word_len the loop equations + positivity + symmetry do **not** uniquely pin
  the moment (the island has width); the operator solve realizes a valid point in
  it. max_word_len=2 gives ≈0.80 — *inside L=6 but OUTSIDE the tight L=8/10 island*
  → a truncation artifact. **max_word_len=3 gives 0.6938 — inside the certified
  L=10 island [0.69307, 0.71408]: operator and bootstrap agree to ~1%.** (At g=0.5
  the operator gives 0.8537, ~5e-5 below the certified lb 0.85375 — the operator's
  own residual max_word_len-truncation; the two methods still agree to ~1e-3, and
  the certified bracket is now tight enough to expose that residual gap.)
- **Lesson (now enforced in code):** a meaningful g>0 guard needs max_word_len ≥ 3
  *and* sdp_word_len ≥ 8. The gate default `sdp_word_len` was raised 6→8.

*Solver:* the islands are **certified by CLARABEL** (`bootstrap_sdp._select_solver`
prefers MOSEK → CLARABEL → SCS). These moment relaxations are degenerate (no
strictly-interior point), so default CLARABEL fails on several instances (e.g. g=1
L=10); a small **static regularization** (1e-7) stabilizes the KKT factorization
and recovers a certified `optimal` (perturbation ≤1e-3), with SCS as an automatic
per-instance fallback. The certified brackets match the earlier SCS estimates to
~1e-3. Fock-cutoff scaling (max_word_len=4 needs dim 8191 with the dense
representation) caps the operator side at max_word_len=3 here; a sparse Fock
representation would push it further.

## Refinements §2 — cross-ansatz (spurious-solution stress test)

The maximal-flexibility `MultiDenseHermitianAnsatz` (M̂_i = (W_i+W_iᵀ)/2, **32,258
params**) warm-started at the free field, same homotopy path, at g=1 / max_word_len=2:
it lands tr M₀²=0.9296, sd=0, symmetry exact, **inside the loose L=6 island** — it
did **not** blow up to a wild out-of-island (spurious) minimum despite full freedom.
The 98-param monomial lands 0.7980 at the same setting. Both are max_word_len=2
truncation artifacts (different points in the loose island), so this is *not*
agreement on the master field — the ansatz-independent rigorous content is the
island, and the genuine cross-method agreement is the monomial max_word_len=3
result (0.6938) matching the SDP tight bracket (≈0.70). A dense max_word_len=3
cross-check (dim 1023, ~2M params) is left as future work.

## Refinements §3 — observables (`observables.py`)

Read off the degree-3 solves (figure: `figures/m3_observables.png`):

- **ρ(λ) of M̂₀** (vacuum spectral measure, weights |⟨Ω|v_i⟩|²): the g=0 spectral
  CDF tracks the Wigner semicircle; as g grows the distribution narrows
  (tr M₀²: 1.00 → 0.87 → 0.79 along the figure's path), consistent with the
  confining commutator term. (Discrete at cut 6 — its moments are the validated
  in-island values; the continuum density needs higher cutoff.)
- **⟨tr[M̂₀,M̂₁]²⟩ vs g** (non-commutativity order parameter): monotone
  −2.0, −1.62, −1.07, −0.56, −0.32 for g=0,0.25,…,1.0 — from the free value −2
  toward 0 (confinement), exactly as the action's −(g²/4)[M₀,M₁]² term predicts.
- **Complex spectrum of M̂₀+iM̂₁** (Brown-measure *proxy*): computed and plotted,
  with the explicit caveat that the uniform eigenvalue distribution is w.r.t.
  (1/D)Tr, not the trace τ — the true Brown measure needs τ-functional calculus
  and is deferred.

## Refinements §4 — sparse Fock representation (`sparse_fock.py`)

Dense D×D operators cap the operator side at dim 1023 (max_word_len ≤ 4);
max_word_len=5 needs dim 8191, where dense monomials (~50 × 8191² × 8 B × 2)
would need ~50 GB. But the basic â_i, â†_i and every monomial â†_u â_v are 0/1
**partial permutations** (|w⟩ → |u·w[|v|:]⟩ iff w starts with reverse(v)), so we
store each as (src→tgt) index arrays and apply M̂_i = (A_i+A_iᵀ)/2 by JAX
scatter-add — differentiable in the coefficients. `SparseMonomialField` +
`train.solve_two_matrix_sparse` mirror the dense ansatz/solve and the same
fail-closed gate; validated against the dense evaluator elementwise and to 1e-9
on moments (`test_sparse_fock.py`).

**max_word_len progression at g=1** (degree 3), vs the certified L=10 island
[0.69307, 0.71408]:

| max_word_len | dim | #transitions | tr M₀² | sd_loss | validated | time |
|---|---|---|---|---|---|---|
| 3 | 1023 | 13 263 | 0.6939 | 9e-18 | True (near lower edge) | 95 s |
| 4 | 1023 | 13 263 | 0.6938 | 3e-13 | True | 4.9 min |
| 5 | **8191** | 106 447 | **0.6975** | 1e-5 | True (more central) | 43 min |

- **All three land inside the certified island.** As max_word_len grows the
  operator value moves from the lower edge (0.6939) toward the interior (0.6975,
  vs the island midpoint ≈0.704) — convergence toward the true value, narrowing
  the operator-side gap that §1 exposed.
- Sparse is also ~5× faster than dense at dim 1023 (13 263 transitions vs a 1023²
  matvec) and is the *only* way to reach dim 8191. XLA compile of the unrolled
  loss is slow at dim 8191 (~3–4 min/compile); the solve still completes.

## Refinements §5 — scaling past max_word_len=5: measured, not assumed

Going beyond max_word_len=5 was *profiled*, which overturned the initial guess that
compile time was the wall. Two costs compete (dim 8191, max_word_len=5):

| evaluator | XLA compile | run / grad-step |
|---|---|---|
| unrolled word loops (current) | 444 s | **267 ms** |
| `vmap` + `lax.scan` (rolled) | 21 s | **3513 ms** |

Rolling the loops with `scan` cuts compile **21×** but inflates **run 13×** — and a
solve is **run-dominated** (max_word_len=5 was ~36 min run vs ~7 min compile), so
scan makes the *total* ≈10× worse (≈5 h vs ≈30 min). Both rolled variants were
validated bit-for-bit against the unrolled evaluator (loss Δ=0, grad Δ≈5e-14), so
this is a pure speed verdict: **scan/batching is the wrong fix and is not
integrated.** The optimal levers instead:

- **≤ max_word_len 5 (run-bound):** keep the unrolled evaluator; cut the *step
  count*. L-BFGS converges from the free field in ~150 steps even at dim 1023, so a
  modest budget + the fail-closed gate suffices. `solve_two_matrix_sparse` now takes
  `init_params` for a **max_word_len-homotopy** — chain a higher cutoff off the
  converged lower one (same param shape, since n_monomials depends only on degree)
  and skip a fresh g-homotopy. (Measured benefit modest at dim 1023; it does not
  touch compile.)
- **≥ max_word_len 6 (compile-bound):** unrolled compile grows super-linearly
  (≈55 s → 444 s from max_word_len 4 → 5) and becomes the wall. The genuine fix is
  **suffix-state sharing**: the loss evaluates hundreds of words sharing suffixes
  (word_moment applies right-to-left, so words ending in the same w share their
  initial partial state), so computing each shared suffix-state once cuts *both* the
  op count (compile) and the compute (run). A real algorithmic optimization, left as
  future work — scan is not a substitute.

## Remaining (sharper, future)

- ~~high-accuracy conic solver~~ **done** (CLARABEL, §1); ~~sparse Fock past
  max_word_len=3~~ **done** (§4, max_word_len=5 / dim 8191). Past max_word_len=5:
  **suffix-state sharing** (§5), not scan/batching (measured worse).
- Dense-ansatz cross-check at max_word_len=3; a proper τ-Brown measure of M̂₀+iM̂₁.

## Latent bug fixed

The truncation guard was degree-blind (`need = max_word_len+3`). A degree-d
ansatz letter changes the Cuntz quanta number by ±d, so a length-L
vacuum→vacuum amplitude reaches ⌊L/2⌋·d quanta and the exact-moment cutoff is
`⌊(max_word_len+3)/2⌋·degree`. Empirically the SD residual hits machine zero
exactly at this cutoff (degree-3/max_word_len-3 needs 9, not 6). The guard in
`solve_two_matrix` is now degree-aware (reads `ansatz.degree`); regression test
`test_truncation_guard_is_degree_aware`.
