# AI Development Discussion Log

<!-- INSTRUCTIONS FOR AI:
  - Reverse chronological order: newest entries on top.
  - Only read the TOP entry — everything below has already been discussed.
  - Naming convention:
      "Discussion-" = AI/human open-ended discussion and brainstorming
      "Implementation-" = AI's implementation results and findings
      "Feedback-"       = Human/AI feedback on the preceding implementation
  - When adding a new entry, prepend it above the previous top entry.
-->

## Implementation-21: Phase B-MM — MM loss works at strong coupling; underdetermined at weak (Apr 12, 2026)

### What was built

`tek_master_field/mm_loss.py`: computes Wilson loops W[C] = Re[ z_12^{signed_area_2d(C)} · Tr(Π U_{μ_i}) ] / N from TEK matrices over a precomputed LoopSystem (imported from `master_field/lattice.py`), evaluates Makeenko-Migdal candidate-D residuals, returns their sum-of-squares. `optimize_tek_mm` plugs into Adam + cosine schedule with the same conj+hermitianize gradient projection used in the classical optimizer. Works for both orientation and full ansätze; D=2 only for now (twist factor uses signed_area_2d).

`tek_master_field/phase_b_mm.py`: runs the MM optimizer across λ ∈ {10, 5, 2, 1.5, 1.2, 1, 0.8, 0.5} at D=2, N=9, L_max=6 with warm-starting. Records W[plaquette], W[2×1], W[2×2] vs GW strong-coupling targets (W[C] = w_+^|Area|, w_+ = 1/(2λ) or 1 − λ/2).

### Result

**Orientation ansatz, N=9:**

| λ | W[plaq] | GW | err% | W[2×1] | GW² | mm_loss |
|---|---|---|---|---|---|---|
| 10 | +0.049 | 0.050 | 1.1% | +0.049 | 0.0025 | 1.1e-3 |
| 5 | +0.099 | 0.100 | 1.1% | +0.099 | 0.010 | 1.3e-2 |
| 2 | +0.291 | 0.250 | 16.5% | +0.291 | 0.063 | 2.3e-1 |
| 1.5 | +0.500 | 0.333 | 50.0% | +0.500 | 0.111 | 1.5e-7 |
| 1 | +1.000 | 0.500 | 100% | +1.000 | 0.250 | 2.8e-8 |

Plaquette matches GW to 1.1% at strong coupling (λ ≥ 5). Deviates at intermediate. Orientation mode's constraint (eigenvalues = L-th roots L-fold-degenerate) produces the degenerate "all W[C] = w_+^MM" family that satisfies MM with constant Wilson loops — the candidate-D self-consistent solution w_+^MM = λ − √(λ²−1), which coincides with w_+^GW only to leading order in 1/λ.

**Full ansatz, N=9:**

| λ | W[plaq] | GW | err% | W[2×1] | GW² | mm_loss |
|---|---|---|---|---|---|---|
| 10 | +0.0502 | 0.0500 | 0.4% | +0.0035 | 0.0025 | 7.5e-12 |
| 5 | +0.1014 | 0.1000 | 1.4% | +0.0136 | 0.0100 | 1.0e-31 |
| 2 | +0.2728 | 0.2500 | 9.1% | +0.0912 | 0.0625 | 1.3e-31 |
| 1.5 | +0.3966 | 0.3333 | 19.0% | +0.1897 | 0.1111 | 2.8e-31 |

Full-ansatz MM loss reaches MACHINE ZERO (10⁻³¹). The plaquette still matches GW to 1% at strong coupling. W[2×1] is NO LONGER identical to W[plaq] — the extra parameters let the optimizer find a non-constant Wilson-loop configuration. But W[2×1] is ~40% larger than GW's product law W[plaq]². The MM equations admit a family of solutions; our optimizer picks one that has the right plaquette but wrong larger-loop structure.

### Relative to Phase B (classical action)

Phase B with classical action alone gave plaquette = 1 at every λ — totally wrong for MC comparison. Phase B-MM gives plaquette tracking GW to ~1% at strong coupling — a significant physics improvement. So MM loss is definitively better than classical action, as predicted.

### Relative to Phase 1b

This reproduces Phase 1b's finding exactly. From Implementation-14:
> MM equations + unitarity alone do NOT uniquely determine Wilson loop values in D=2. [...] R4 (supervised warm-start 1500 epochs, then MM-only 3000 epochs): best result. W[plaq] at λ=5 = 0.10012 vs GW 0.1 (near-exact). W[2×1] at λ=1 drifts from supervised 0.25 to 0.16 after MM refinement.

Same issue, different parametrization: MM candidate D is underdetermined; uniquely pinning W[C] requires additional constraints (positivity) or a bias toward the physical solution (warm-start).

### Phase 3 risk status

R6 is **partially resolved**. With MM loss:
- ✓ Classical-saddle pathology fixed: plaquette is coupling-dependent.
- ✓ Strong-coupling plaquette matches GW to ~1%.
- ✗ Area law W[R×T] = W[plaq]^{RT} not captured.
- ✗ Weak coupling still unreliable.

These are exactly the failure modes Phase 1b identified. Next step is to add positivity constraints (R3 from the original roadmap) or supervised warm-start at the GW strong-coupling value.

### Proposal for next step

Add a **Toeplitz-PSD constraint** on the Wilson loop algebra (or equivalently a supervised anchor at the plaquette to the GW strong-coupling value) as an additional loss term:

    L_total = L_MM + λ_anchor · (W[plaq] − 1/(2λ))²         (supervised mode)
    L_total = L_MM + λ_psd · barrier(−min_eig(Toeplitz))    (positivity mode)

Supervised anchor is cheaper and proven in Phase 1b. Positivity is more rigorous but harder to compute. Recommend: anchor first to confirm the pipeline, then positivity to lift the anchor.

### Phase 3 risk status table (updated)

| Risk | Status |
|------|--------|
| R1 D=4 k=1 center-symmetry break | deferred |
| R2 rectangular Wilson loop phase | resolved (Impl-16) |
| R3 sign conventions | resolved (scaffolding) |
| R4 orientation-only sufficiency | resolved (Impl-18) |
| R5 Γ spectrum mismatch | resolved (Impl-17) |
| R6 classical action misses master field | **partially resolved (this entry)** — MM correct at strong coupling, underdetermined elsewhere |
| R7 MM alone underdetermined at D=2 | **NEW** — add positivity constraints or supervised anchor next |

---

## Discussion-20: Choosing the R6 fix — Haar entropy vs MM loss vs SDP bootstrap (Apr 12, 2026)

### Context

Implementation-19 (Phase B) established empirically that minimizing the classical TEK action S_classical over matrices U_μ gives the zero-entropy classical vacuum at every coupling, not the master field. This is R6. Three routes to recover coupling-dependent master-field observables were listed. This entry evaluates them before picking one.

### Physics recap — what the master field actually is

The master field is not a canonical N×N matrix. It is:
- in Witten's sense, the N=∞ saddle of the path integral, defined by large-N factorization;
- in Gopakumar-Gross's sense, a pair of operators Û_μ in the Cuntz-Fock space, encoding all single-trace expectation values;
- in MC's sense, the limiting configuration around which finite-N samples cluster with O(1/N²) fluctuations.

All three of these encode the SAME single-trace observables (Wilson loops). The master field has well-defined Wilson loops at N=∞, but the specific matrix realization at finite N is gauge-dependent.

The path integral at finite N is Z = ∫ dU e^{−S[U]}. In eigenvalue coordinates, the Haar measure contributes a Vandermonde repulsion ∏|e^{iθ_i} − e^{iθ_j}|². At large N this log-Vandermonde term is O(N²), same order as the action. At the saddle the two balance — yielding the MC-observed plaquette vs coupling. Minimizing S alone drops the Vandermonde and collapses to the classical vacuum; this is what Phase B saw.

### Option 1 — Add Haar entropy to the loss

Derive log|J| for the parametrization and add it.

For U = exp(iH) ∈ U(N) with H Hermitian: the Jacobian of the exponential map gives

    dU_Haar / dH = |det[(1 − e^{−i ad H}) / (i ad H)]|   (Duflo/BCH formula)

In eigenvalue coordinates of H (eigenvalues θ_i), this specializes to

    |det J| = ∏_{i<j} |sinc((θ_i − θ_j)/2)|²

Adding −log|J| to the loss restores the Vandermonde repulsion. The optimizer would see an eigenvalue-repulsion pressure that, balanced against the plaquette-attraction, produces a coupling-dependent saddle.

**Pros.** Direct. Preserves the "find matrix master field" framing. Cleanly matches MC at N=∞ by construction.

**Cons.** (i) The Duflo Jacobian is expensive to evaluate — requires spectral decomposition or a determinant of an (N² × N²) adjoint matrix per gradient step. (ii) Gradients of −log|J| through autodiff on a near-degenerate spectrum can be numerically fragile. (iii) The orientation ansatz's coadjoint-orbit measure is constant (no log|J| contribution), so this option only rescues the FULL ansatz, not orientation. (iv) Still unclear whether the saddle of (S + log|J|) is unique — may have center-broken and center-symmetric minima depending on initialization.

**Cost estimate.** A week of physics derivation + implementation + debugging. Substantial.

### Option 2 — Makeenko-Migdal loop equations as loss

The MM equations are loop-space statements of the master-field condition:

    λ · W[C]  =  Σ_{P ∋ e} W[P_e ∘ C]  −  Σ_{splits} W[C_1] · W[C_2]

These equations are satisfied BY the master field at N=∞, independent of how we parametrize. We compute Wilson loops from our TEK matrices U_μ, sum MM residuals squared, minimize.

**Pros.** (i) Already proven on D=2 lattice YM in Phase 1b (with partial success — MM loss converges, but not uniquely without positivity). (ii) No Haar-entropy derivation needed — MM IS the correct master-field condition. (iii) Works for both orientation and full ansätze. (iv) Natural coupling dependence: λ enters explicitly in the equation.

**Cons.** (i) Phase 1b on D=2 revealed that MM alone is underdetermined (K unknowns, K − few equations). Needs positivity constraints for uniqueness. (ii) At each step, we must evaluate W[C] for many lattice loops via Tr(product of U_μ)/N — O(loop_length × N²) per loop. For L_max = 6 on TEK, ~35 loops times 32 equations; manageable. (iii) TEK requires the twist-adjusted MM equations; need to verify our existing MM implementation (staple convention from `master_field/mm_equations.py`) applies correctly after volume reduction.

**Cost estimate.** Moderate. Most of the infrastructure exists in `master_field/lattice.py` and `mm_equations.py`. Porting to TEK means: (a) evaluating W[C] from TEK U_μ instead of neural functionals; (b) adding the TEK twist phase to the MM equation (analogous to rectangular Wilson loop twist from Impl-16); (c) coupling-continuation schedule over λ. Perhaps 2–3 days.

### Option 3 — SDP bootstrap with continuation

Formulate as a semidefinite program: MM equations as linear constraints, positivity of the Toeplitz moment matrix as PSD constraint, minimize a scalar observable (or find feasible point). With optimum over W[C] values directly; recover U_μ from W[C] post-hoc (harder).

**Pros.** (i) Rigorous — Kazakov-Zheng 2021 proved this converges to the unique master field at N=∞ with certifiable bounds. (ii) Handles non-uniqueness of MM alone via positivity. (iii) No autodiff/gradient issues.

**Cons.** (i) SDP solvers (cvxpy + SCS/MOSEK) don't easily give us U_μ* matrices — they give W[C]* values. Matrix reconstruction is a separate step (hard, probably using the Cuntz-Fock framework). (ii) SDP scaling is harder at large N_loops (N_loops grows exponentially with loop length). (iii) Doesn't exploit our Phase 3 matrix parametrization; largely replaces it.

**Cost estimate.** High, and pivots away from the "master field as matrices" narrative. Perhaps a week + significant physics to reconstruct matrices from bootstrap bounds.

### Recommendation

**Pursue Option 2 (MM loop-equation loss) with Option 3's positivity as a later addition.**

Rationale:
- MM is the correct master-field condition — no ambiguity about what we are solving.
- It reuses Phase 1's infrastructure (lattice loop enumeration, MM equation machinery).
- It works for both ansätze, so we can test which ansatz is better after fixing R6.
- Failure mode (MM underdetermined) is known from Phase 1b and has a clean fix (R3 positivity + warm-start from supervised or SDP) if needed.
- Keeps the "matrix master field" story intact: we optimize matrices U_μ, just with a different (correct) loss.
- Cheapest of the three in wall time.

Option 1 (Haar entropy) is the most physically direct, but computationally fragile and expensive. Defer unless Option 2 fails on D=2 TEK.

Option 3 (SDP) is the gold standard for W[C] values but shifts us away from matrix construction. Defer for now; if we need rigorous bounds for validation, use existing Kazakov-Zheng results.

### Proposed next action

Create `tek_master_field/mm_loss.py` that:
1. Takes a LoopSystem (enumerated lattice loops + MM equation indices) for D=2.
2. Builds a function `mm_loss_tek(params, Gamma, z, D, ansatz, loop_sys)`:
   a. Build U_μ from params (existing).
   b. For each canonical loop C in loop_sys, compute W[C] = Re[Z(C) · Tr(product)] / N with the TEK twist phase Z(C) along the path.
   c. Compute MM residuals: λ · W[C] − Σ W[P_e ∘ C] + Σ W[C_1] W[C_2].
   d. Return sum of squared residuals.
3. Plug into `optimize_tek` as an alternative loss (`loss="classical" | "mm"`).
4. Validate on untwisted D=2 EK against Phase 1b's D=2 results (should match area law W = (1/(2λ))^A at strong coupling).

Phase B-MM would then be: run both ansätze with MM loss on untwisted and twisted D=2. If plaquette matches MC strong coupling (1/(2λ)) and weak coupling behavior, R6 is resolved.

---

## Implementation-19: Phase B — classical saddle found; master-field requires Haar entropy (Apr 12, 2026)

### Result

Phase B ran the direct-optimization TEK program on the untwisted Eguchi-Kawai action (z = 1) at D=2, N=49 over a coupling schedule λ ∈ {20, 10, 5, 2, 1, 0.5, 0.3} with BOTH orientation and full U(N) ansätze. After a critical gradient-sign fix (see below), both ansätze converged cleanly to the **classical TEK vacuum** at every λ:

| ansatz | plaq | |P_1| | |P_2| | cs_order | |grad|/N |
|---|---|---|---|---|---|
| orientation | +1.0000 | 0 | 0 | 10⁻³² | 10⁻¹¹ |
| full | +1.0000 | 1.0 | 1.0 | 2.00 | 10⁻¹¹ |

Orientation converges to U_1 = U_2 = Γ (center-symmetric traceless). Full converges to U_1 = U_2 = I (maximally center-symmetry-broken). Both give plaquette = 1 regardless of coupling.

### Interpretation (R6, NEW)

The Monte-Carlo answer for untwisted EK D=2 at finite λ has plaquette that depends on λ: strong coupling → 0, weak coupling → 1 with Z_N breaking. Our result (plaq = 1 at every λ) does not reproduce this.

The physics: at N=∞ the path integral saddle is determined by δS_eff/δU = 0 where S_eff = S_classical − log|Jacobian of dU|. The Haar entropy (−log|J|) adds an eigenvalue-repulsion pressure that spreads eigenvalues away from the classical vacuum and makes the saddle coupling-dependent. Direct minimization of S_classical alone finds only the classical vacuum (the "zero-entropy" configuration).

The orientation ansatz DOES use a Haar-measure argument (KKS-invariant measure on the coadjoint orbit is constant), which is why its S_eff restricted to the orbit equals S_classical. But the classical minimum of S_classical ON the orbit is still the trivial U_μ = Γ (the orbit's most symmetric configuration). The coadjoint orbit contains the classical TEK saddle U_2 = Q_L ⊗ P_L (for the TWISTED case — different from Phase B's untwisted test), but both U_μ = Γ and the clock-shift pair give plaq = 1 at untwisted z=1.

The full ansatz has no measure restriction at all, so it finds the global classical minimum (all U_μ = I, plaq = 1).

Neither sees the quantum fluctuations encoded in the Haar measure. This is R6: the loss function needs a Haar-entropy term (or switch to MM loop-equation residuals, à la Phase 1 Direction A) to recover coupling-dependent master-field observables.

### Gradient-sign bug discovered and fixed

Unrelated to the physics, Phase B surfaced a JAX convention gotcha. For a real loss f of complex-matrix parameter H, `jax.grad(f)(H)` returns ∂f/∂x − i·∂f/∂y (note minus sign on imaginary derivative), which is the **conjugate** of the physical descent direction ∂f/∂x + i·∂f/∂y. Feeding the raw JAX gradient to optax via `params − lr · grad` updates in the wrong direction.

Fixed in `optimize.py::_step`: conjugate the gradient before hermitianizing and passing to the optimizer. Verified by line search against direct loss evaluation: `-lr·conj(grad)` decreases loss, `-lr·grad` increases it. Confirmed also on a trivial |z|² test (JAX returns 2−2j at z=1+1j where the correct descent direction is 2+2j = 2z).

After the fix:
- At H=0 (untwisted): |grad|/N = 0, plaquette = 1, optimizer correctly stays at the classical vacuum.
- At random init: |grad|/N monotonically decreases to 10⁻¹¹ over the schedule, plaquette converges to 1.
- All 89 pytest tests continue to pass.

### Deliverables

- `tek_master_field/phase_b.py`: standalone Phase B experiment script.
- `results/phase_b_summary.json`: per-ansatz, per-λ observables (plaquette, P_μ, center-symmetry order, eigenvalue histogram, loss, grad norm, elapsed time).
- `optimize.py::_step`: conj + hermitianize gradient projection.

### What Phase B tells us about the Phase 3 program

The scaffolding is **technically correct**: both ansätze represent the intended parameter spaces, the gradient converges, the tests pass. But the LOSS FUNCTION is physically incomplete — minimizing classical S gives the classical vacuum, not the master field.

Three options for next steps:

1. **Add Haar entropy to the loss.** Compute log|J| of the exp(iH) parametrization (related to sinh(ad·H/2) / (ad·H/2) determinant) or the Jacobian of the coadjoint orbit inclusion. This is a nontrivial physics derivation but most direct.
2. **Switch to Makeenko-Migdal loop-equation loss** (Phase 1 Direction A) applied to TEK. Works for lattice loops; proven successful at D=2 in Phase 1b.
3. **SDP bootstrap with continuation** (Phase 1 Direction C). Hard constraints from Kazakov-Zheng bounds + neural parametrization.

Phase B has delivered a CLEAN, REPRODUCIBLE FAILURE per the original plan gate. The next concrete step is to pick one of the above three directions and formulate the new loss.

### Phase 3 risk status (updated)

| Risk | Status |
|------|--------|
| R1 D=4 k=1 center-symmetry break | deferred |
| R2 rectangular Wilson loop phase | resolved (Impl-16) |
| R3 sign conventions | resolved (scaffolding) |
| R4 orientation-only sufficiency | resolved (Impl-18) — full ansatz available |
| R5 Γ spectrum mismatch | resolved (Impl-17) |
| R6 classical action alone misses master field | **NEW, unresolved** — requires Haar entropy or switch to MM/SDP |

---

## Implementation-18: R4 resolved — full U(N) ansatz as alternative to orientation-only (Apr 12, 2026)

### What was added

Alongside the orientation-only ansatz U_μ = Ω_μ Γ Ω_μ† (eigenvalues locked to L-th roots of unity, L-fold degenerate), we now also support the **full U(N) ansatz**:

    U_μ = expm(i M_μ),   M_μ Hermitian,   μ = 1, …, D

Parameters: D · N² real (vs. (D−1) · N² for orientation). Eigenvalues of each U_μ are free; this ansatz can represent center-symmetry-breaking configurations that the orientation-only ansatz cannot (e.g., fluxons relevant to D=4 TEK at k=1 per hep-th/0612097).

### Implementation

- `tek.py`: new `build_link_matrices_full`, `tek_loss_full`, `plaquette_average_full`, `init_M_list_zero`, `init_M_list_random`.
- `optimize.py`: `optimize_tek` and `coupling_continuation` accept `ansatz ∈ {"orientation", "full"}` (orientation default). `OptResult.params` holds H_list or M_list depending on ansatz, `OptResult.ansatz` stores the choice. Backward-compat `res.H_list` / `res.M_list` properties alias `res.params`.
- `train.py`: new `--ansatz {orientation, full}` flag. Output file naming now includes the ansatz tag.
- `test_tek.py`: 11 new tests (full ansatz unitarity, M=0 plaquette matches H=0 orientation value, optimizer loss-decrease and unitarity preservation, input validation). 89/89 total pass.

### At-M=0 vs at-H=0 parity

Verified: at H=0 (orientation, all U_μ = Γ) and M=0 (full, all U_μ = I) the loss has the same value = −(mean Re(z_μν)). Both plaquette products equal I. The two ansätze start from the same loss but on different parameter manifolds; Adam explores different directions from there.

### Parameter counts (N = 49, D = 2 example)

| ansatz | params | orbit dim | gauge dim |
|--------|--------|-----------|-----------|
| orientation | 1 × 49² = 2401 | ~2058 | 343 (U(L)^L stabilizer) |
| full | 2 × 49² = 4802 | 4802 | N² − 1 = 2400 global gauge |

Full has more parameters AND more gauge redundancy. Per gradient step the full mode is about 2× slower (two expm's vs one), but the step count to saddle may differ either way.

### Purpose

Phase B will run both ansätze on untwisted EK (D=2) and compare. Expected:

- If orientation reaches a coupling-dependent saddle with the correct GW strong-coupling answer, great — use orientation for Phase C/D (cheaper).
- If orientation stays stuck at the classical TEK vacuum (plaquette ≈ 1 regardless of λ), that's R4 failing, and full should produce the quantum master field (at the cost of D·N² params).
- If BOTH fail to produce coupling-dependent plaquette, the loss function itself needs a Haar-entropy term — a deeper physics issue noted in the reference doc §"Open Question".

### Phase 3 risk status summary

| Risk | Status | Note |
|------|--------|------|
| R1 D=4 center-symmetry breaking at k=1 | deferred | Addressed in Phase D via `--k` and modified flux per arXiv:1005.1981 |
| R2 rectangular Wilson loop twist phase | resolved (Impl-16) | W[R×T] = Re[z^{RT} · Tr(...)]/N, verified on classical saddle |
| R3 sign conventions | resolved (Phase 3 scaffolding) | Standardized loss = −(mean plaquette)/N_pairs |
| R4 orientation-only sufficiency | resolved (this entry) | Full U(N) available via `--ansatz full` |
| R5 Γ spectrum mismatch with TEK saddle | resolved (Impl-17) | Γ = kron(P_L, I_L), classical saddle reachable |

---

## Implementation-17: R5 resolved — Γ replaced with kron(clock_L, I_L) (Apr 12, 2026)

### Fix

`build_clock_matrix(N)` was returning `diag(1, ω_N, …, ω_N^{N-1})` (eigenvalues = N-th roots of unity, all distinct). This is the wrong spectrum for TEK: the classical saddle uses twist eaters `P_L ⊗ I_L` whose eigenvalues are L-th roots with L-fold degeneracy. Since spectra are invariants under unitary conjugation, the old ansatz Ω Γ Ω† could not reach the TEK classical saddle at finite N.

Replaced with:

    Γ = kron(P_L, I_L)    where P_L = L-dim clock matrix, L² = N

Properties after the swap:
- Γ^L = I (stronger than the old Γ^N = I)
- Eigenvalues are L-th roots of unity, each with multiplicity L
- Traceless for L > 1
- Matches TEK twist-eater U_1 in arXiv:1708.00841 §2.2 eq. 2.16
- Coadjoint orbit {Ω Γ Ω†} now contains the classical-saddle partner Q_L ⊗ P_L (same spectrum), so the orientation-only parametrization can reach the saddle

### Verification

78/78 pytest tests pass. New tests added:
- `test_clock_matrix_eigenvalues_are_L_roots_L_degenerate`: direct eigenvalue check for N ∈ {9, 25, 49}
- `test_clock_matrix_L_periodic`: Γ^L = I to 1e-10 for N ∈ {9, 25, 49, 121}
- `test_clock_matrix_rejects_non_perfect_square`: input validation
- `test_clock_matrix_traceless`: Tr(Γ) = 0 to 1e-12
- `test_clock_matrix_matches_tek_classical_saddle_U1`: Γ = kron(P_L, I_L) to 1e-14

All pre-existing tests continue to pass (plaquette at H=0, rectangular Wilson loops, TEK classical-saddle W[R×T] = 1, etc.). The change is invisible to the plaquette observable at H=0 because all U_μ = Γ still commute (Γ is block-diagonal).

### Measure of progress

Phase 3 scaffolding now has both R2 (rectangular Wilson loop twist phase) and R5 (ansatz Γ spectrum) resolved. Remaining risks:
- **R1** (D=4 center-symmetry breaking at k=1, cure via modified flux k ≈ L/2): deferred to Phase D.
- **R3** (sign conventions): standardized in Phase 3 scaffolding, no action needed.
- **R4** (orientation-only vs full U(N)): Phase B will test. With R5 fixed the orientation-only ansatz is now MORE likely to succeed since it can reach the classical saddle.

### Parameter count

- Old Γ (N-th roots, distinct): stabilizer = U(1)^N, orbit dimension N² − N.
- New Γ (L-th roots, L-fold): stabilizer = U(L)^L, orbit dimension N² − L·L² = N² − N^{3/2}.

For N = 49: 2401 − 49 = 2352 (old) vs 2401 − 343 = 2058 (new). Fewer effective parameters, but they encode the correct TEK structure. The remaining L·L² = 343 parameter-directions are gauge (stabilizer of Γ); Adam tolerates this overparametrization but a stabilizer-projected parametrization may be needed for large-N fine-tuning.

---

## Implementation-16: R2 resolved — rectangular Wilson loop twist phase (Apr 12, 2026)

### Result

`observables.wilson_loop_rectangular` is now implemented and verified. The twist phase for an R×T rectangular Wilson loop in the (μ,ν) plane on single-site TEK is **z_μν^{R·T}**:

    W[R×T]_{μν} = Re[ z_μν^{R·T} · Tr(U_μ^R U_ν^T U_μ^{-R} U_ν^{-T}) ] / N

### Source

García Pérez, González-Arroyo, Okawa, arXiv:1708.00841 eq. (2.4): "W_{R,T}(b, N, L, n_μν) = (1/N) Z(R,T) ⟨Tr(U(R,T))⟩, where Z(R,T) is the product of the Z_μν(n) factors for all plaquettes which fill up the rectangle." For single-site TEK (their L=1), every elementary plaquette has the same twist factor z_μν = Ẑ_μν = exp(2πi n_μν / N), giving Z(R,T) = z_μν^{R·T}. Consistent with arXiv:1212.3835 eq. (1.1) plaquette convention.

### Verification

Three independent tests:
1. **R = T = 1 reduces to the plaquette.** Direct equality to `wilson_loop_plaquette` for random H, to 10⁻¹².
2. **At H = 0** (all U_μ = Γ, diagonal, commuting): W[R×T] = Re(z_μν^{R·T}) exactly, to 10⁻¹⁰, across (R,T) ∈ {(1,1), (1,2), (2,1), (2,2), (3,2), (2,3), (3,3)} and N ∈ {9, 25, 49}, k ∈ {1, 3}.
3. **TEK classical saddle** (the strongest check): construct U_1 = P_L ⊗ I_L, U_2 = Q_L ⊗ P_L from L×L clock-shift matrices. Verify U_1 U_2 = ω_L^{-1} U_2 U_1 (Heisenberg relation) to 10⁻¹² for L ∈ {3, 5, 7}. At this saddle W[R×T] = 1 exactly for every (R, T) tested (L ∈ {3,5,7}, R,T ≤ 4) to 10⁻¹⁰. This independently reproduces the arXiv:1708.00841 eq. (2.4) claim.

65/65 pytest tests pass after R2 resolution (+26 new tests on top of Phase 3 scaffolding's 39).

### New finding (R5)

Building the TEK classical saddle explicitly revealed a subtle ansatz mismatch. The TEK twist eaters P_L ⊗ I_L have eigenvalues equal to the **L-th roots of unity, L-fold degenerate**. Our current Γ = diag(1, ω_N, …, ω_N^{N-1}) ansatz has eigenvalues equal to the **N-th roots of unity, all distinct**. These spectra differ (N = L² > L), and since spectra are conjugation invariants, no Ω Γ Ω† can reach the TEK classical saddle starting from our current Γ.

At N → ∞ both spectra become uniform on the unit circle and the distinction vanishes. At finite N, our ansatz Ω Γ Ω† reaches a different saddle (the maximizer over a different submanifold), not the TEK classical saddle.

**Fix is straightforward and deferred to Phase B** if Phase B shows pathological behavior: replace `build_clock_matrix(N)` with `kron(clock_L, I_L)`. The Haar-measure argument still holds (coadjoint orbit has constant measure) and the orientation parametrization matches the TEK structure explicitly. The orbit dimension drops from N² to N² − L³, encoding the TEK center-symmetric constraint.

Documented in reference/tek_master_field.md §"Ansatz Caveat (R5)".

### Phase C/D unblocked

With R2 resolved, Phase C Wilson-loop comparisons against the lattice GW strong-coupling product law `W[R×T] ≈ (1/(2λ))^{R·T}` are now computable. Creutz ratios χ(R,R) for string tension become well-defined. Only R5 (ansatz choice) remains between the current scaffolding and full Phase C/D execution.

---

## Discussion-15: Phase 3 — Direct Optimization of the TEK Master Field (Apr 12, 2026)

### Motivation

At N = ∞ the path integral is dominated by a single saddle — the master field. Monte Carlo samples *around* the saddle with O(1/N²) and O(1/√T) noise. Direct gradient-descent optimization finds the saddle itself: no thermalization, no autocorrelations, no statistical noise. For computing the master field this is strictly better than MC if it converges.

Nobody has solved the Twisted Eguchi-Kawai (TEK) model by direct optimization. Phase 3 attempts this. If Phase D (TEK D = 4 at N = 289) succeeds with plaquette matching published MC < 1 %, the result is the first explicit construction of the SU(∞) master field for 4D lattice Yang-Mills — a problem open since Witten (1979) / Gopakumar-Gross (1994). Direction B of the Discussion-11 roadmap.

### Model

TEK reduces D-dimensional SU(N) lattice YM to D unitary N×N matrices at a single site with action

    S = −(N/λ) Σ_{μ<ν} Re (1/N) Tr(z_{μν} U_μ U_ν U_μ† U_ν†)

and path-integral weight exp(−S). The twist z_{μν} = exp(2πi n_{μν}/N) with symmetric flux n_{μν} = k·L (on twisted planes, N = L² with L prime) pins Z_N^D center symmetry and restores volume independence, so N = ∞ TEK observables equal the infinite-volume theory.

### Master Field Ansatz

Center-symmetric eigenvalues are locked to the N-th roots of unity:

    U_μ = Ω_μ · Γ · Ω_μ†,   Γ = diag(1, ω, ω², …, ω^{N-1}),   ω = e^{2πi/N}

Gauge-fix Ω_1 = I; parametrize Ω_μ = exp(i H_μ) with H_μ Hermitian for μ ≥ 2. The Haar measure, restricted to the coadjoint orbit {g Γ g†}, is proportional to the Vandermonde of the eigenvalues — constant since eigenvalues are fixed. Total parameters: (D − 1) · N².

### Physics Risks (R1–R4, tracked in the plan)

- **R1 — Center-symmetry breaking at D = 4, k = 1.** Observed for N ≥ 100 (hep-th/0612097). Cure: modified flux k ≈ L/2 per González-Arroyo-Okawa 2010 (arXiv:1005.1981). Safe for D = 2, D = 3 with k = 1.
- **R2 — Rectangular Wilson loop twist phase f(R,T).** Nontrivial from Heisenberg non-commutativity. Must be transcribed from PRD 27 (1983) eq. (3.5) or arXiv:1708.00841 before computing W[R×T]. Currently gated as `NotImplementedError`.
- **R3 — Sign conventions.** Standardized: loss = −Σ_{μ<ν} Re[z_{μν} Tr(U_μ U_ν U_μ† U_ν†)/N] / N_pairs. No λ inside the loss.
- **R4 — Orientation-only vs full U(N).** If the ansatz cannot produce coupling-dependent plaquette (classical saddle ≠ master field), enlarge to U_μ = exp(i M_μ) with M_μ Hermitian, D · N² parameters, no fixed eigenvalues.

### Implementation

New subfolder `tek_master_field/` with seven Python modules: `tek.py` (core: clock matrix, twist, link builder, action), `optimize.py` (Adam + warmup_cosine schedule + coupling continuation), `observables.py` (plaquette, Polyakov, eigenvalue density; rectangular gated), `gross_witten.py` (Phase A sanity check), `train.py` (CLI), `config.py`, `test_tek.py` (39 pytest tests). JAX + optax with float64, H_μ re-Hermitianized after each gradient step.

Infrastructure mirrors `master_field/`: float64 config, optax chain pattern from `neural_master_field.py:436`, training-step JIT pattern from `:468`, coupling continuation from `:597`, pytest marks from `test_qcd2.py`.

### Phases

- **Phase A — Gross-Witten** (1-matrix unitary; not TEK). Path C: parametrize support endpoint a, find by normalization, verify exact w_1, w_2. **Gate: err < 10⁻⁶.**
- **Phase B — untwisted EK, D = 2.** First test of the orientation-only ansatz. If it fails to develop a coupling-dependent saddle, adopt R4 fallback.
- **Phase C — TEK, D = 2.** First real TEK computation. Compare to exact GW strong coupling at N = 49, 121, 289. **Gate: plaquette within 0.5 % of reference.**
- **Phase D — TEK, D = 4** (the target). Compare to González-Arroyo–Okawa MC at N = 289, β = 0.356. **Gate: plaquette within 1 % of MC.**
- **Phase E — N → ∞ extrapolation.** Fit O(N) = O(∞) + c₁/N² + c₂/N⁴ from N ∈ {49, 121, 289, 529}.

### Current status (end of this session)

- Scaffolding: `tek_master_field/` created, `__init__.py`, `conftest.py`, `results/` (gitignored).
- **Phase A gate PASSES.** At t ∈ {0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0}: worst |w_1 − exact| = 5.5 × 10⁻¹², worst |w_2 − exact| = 1.6 × 10⁻¹³. Well below the 10⁻⁶ gate.
- **Core infrastructure works end-to-end.** 39/39 pytest tests pass. Smoke test: D = 2, N = 9, k = 1 continuation over λ = {10, 5, 2, 1} completes in 3.8 s, JIT + optax update + hermitianize stable, link matrices remain unitary to 10⁻¹⁰.
- Reference doc `reference/tek_master_field.md` written (symmetric twist, center-symmetry caveat, open question about classical vs quantum saddle, benchmark MC table, bibliography).
- Cluster script `cluster/submit_tek.pbs` for N = 289 runs (mirrors `cluster/submit.pbs`).
- Phases B–E deferred to subsequent implementations. Phase B is the first test of R4.

### What this delivers

The infrastructure to run direct-optimization TEK computations at any D ∈ {2, 3, 4} and any N = L² with L prime, up to the center-symmetry ansatz assumption. The code is the direct executable of the plan in `/Users/dz1614/.claude/plans/piped-twirling-narwhal.md`. Next concrete step is Phase B: the orientation-only ansatz will either produce a coupling-dependent saddle (confirming the Gopakumar-Gross "axial-gauge Gaussian" picture in a TEK context), OR fail to do so (triggering R4 fallback to the full U(N) parametrization). Phase B is a two-matrix numerical experiment; its outcome decides the architecture for Phases C and D.

### References

Eguchi-Kawai 1982 (original). González-Arroyo-Okawa 1983 PRD 27 (original TEK + eq. 3.5 for rectangular loops). González-Arroyo-Okawa 2010 arXiv:1005.1981 (modified flux). García Pérez-González-Arroyo-Okawa 2017 arXiv:1708.00841 (perturbative Wilson loops with twist). Teper et al. 2006 hep-th/0612097 (center-symmetry breaking). Gopakumar-Gross hep-th/9411021 §1 (spacetime-independent master field). Local: `reference/tek_master_field.md` (full bibliography).

---

## Implementation-14: Phase 1 Steps 2–5 — LoopSystem, Neural Model, D=2 Training (Apr 12, 2026)

### What was built

- `lattice.LoopSystem` dataclass + `build_loop_system(D, L_max)`: enumerates canonical lattice loops up to L_max + 2 (to cover MM staple insertions) and precomputes every MM equation as index arrays (`MMEquation`). D=2 L_max=6 gives K=35 loops, 32 equations. D=3 L_max=6 gives 627 loops, 288 equations.
- `master_field/neural_loop.py`: `NeuralLoopFunctional` — MLP mapping λ to K-vector of Wilson loop values. `mm_loss`, `supervised_loss_2d`, `unitarity_penalty` as JIT functions. Training helpers: `train_supervised_2d`, `train_mm_2d`, `train_mm_2d_curriculum`, `train_mm_2d_warmstart`.

### Phase 1a (supervised D=2): works

Trained against GW lattice answer W[C] = w_+^Area with w_+ = 1/(2λ). After 3000 epochs at L_max=6:
- λ=1: W[plaq]=0.502 (exact 0.500), W[2×1]=0.243 (exact 0.250)
- λ=2: W[plaq]=0.251 (exact 0.250), W[2×1]=0.064 (exact 0.0625)
- λ=5: W[plaq]=0.093 (exact 0.100), W[2×1]=0.006 (exact 0.010)

Max error ~1–2%. Architecture validated; more epochs / bigger net would tighten further.

### Phase 1b (MM-only D=2): partially works, exposes real structure

**Initial attempt (tanh + Xavier init) FAILED**: converges to spurious local minimum w[C] = −1 for all non-empty loops. Final MM loss 4.3. Tanh saturation blocks the gradient; Xavier init starts well outside the GW basin.

**Remediations implemented** (per approved plan):
- **R1** (no tanh, small W_out init=0.01, soft unitarity penalty): dramatic improvement. MM loss 4.3 → 6.8e-4. W[plaq] closely tracks the self-consistent candidate-D solution w_+^MM = λ−√(λ²−1). But W[2×1] is driven to ≈ 0 (wrong).
- **R2** (curriculum λ: 10→1 over 8 stages): MM loss → 1e-26 (essentially zero) BUT converges to a solution that violates unitarity (W[plaq]=1.3 at λ=5). MM + soft unitarity penalty at weight 10 isn't strong enough.
- **R4** (supervised warm-start 1500 epochs, then MM-only 3000 epochs): best result. W[plaq] at λ=5 = 0.10012 vs GW 0.1 (near-exact). W[2×1] at λ=1 drifts from supervised 0.25 to 0.16 after MM refinement.

### The real finding

**MM equations + unitarity alone do NOT uniquely determine Wilson loop values in D=2.** Multiple solutions exist:
- The GW answer w_+ = 1/(2λ), W[C] = w_+^Area
- The candidate-D self-consistent solution w_+^MM = λ − √(λ²−1), close to GW only at strong coupling
- Solutions violating |W| ≤ 1 if unitarity penalty is weak
- Solutions where high-area loops → 0 regardless of what the area law predicts

At K=35 unknowns vs 32 MM equations, with one hard constraint (W[empty]=1), the system is 2 equations short of uniqueness. Curriculum training (R2) shows that the MM loss has a continuous family of near-zero minima, not a single point.

### Interpretation and next step

This is consistent with the role of the MM equation in the full QCD bootstrap: MM equations are NECESSARY but not SUFFICIENT. The Kazakov-Zheng program uses SDP positivity (R3) on top of MM to pin down the solution. Our Phase 1 on D=2 validates this claim empirically.

**Figure-8 subtlety discovered**: our `abs_area_2d` returns 0 for loops with signed area 0 (e.g., figure-8 = CCW plaquette + CW plaquette joined at a vertex). These loops factorize at the self-intersection and have W = w_+·w_−, not W = w_+^0 = 1. The exact-target formula `W[C] = w_+^abs_area_2d(C)` is only valid for simple (non-self-intersecting) loops. Documented for future D=2 validation.

### Decision

Move to Phase 1c (D=3) where:
1. The architecture + R4 (warm-start with a strong-coupling initialization) can be validated against bootstrap bounds (Kazakov-Zheng 2203.11360) rather than an exact area law.
2. MM + positivity (R3) becomes the natural next implementation step. The bootstrap bounds tell us approximately where W must lie; MM refines.
3. The exploration problem (unique solution selection) is attacked directly in the regime where it matters.

R3 (positivity / Toeplitz-PSD) is deferred to Phase 1c where we'll need it anyway. Phase 1b on D=2 provides the calibration baseline.

### Tests

All 16 pre-existing + 11 MM + 18 QCD₂ pytest tests still pass.

---

## Implementation-13: Phase 1 Step 1 — MM Equation for Lattice 2D YM (Apr 12, 2026)

### Goal

Per the Phase 1 plan, Step 1 is: fix and validate the Makeenko-Migdal equation machinery on D=2 before any neural-net code. The plan specified `test_mm_all_loops_2d` pass at 10⁻⁶ residual against exact area law.

### What was built

- Fixed `lattice.plaquette_insertions`: now uses STAPLE-replacement convention. Edge +μ at position e_j is replaced by the 3-step staple (ν, +μ, −ν) for each ν ≠ ±|μ|. Net displacement preserved, loop stays closed.
- New `mm_equations.py`: candidate catalog A–G of MM equation forms; `gw_w_plus(λ)` returning the Gross-Witten single-plaquette strong/weak coupling value; `mm_residual_staple(loop, edge_idx, D, lam, W, include_self_closure)`; `scan_candidates_2d(target="lattice" or "continuum")` empirically compares candidates.
- New `test_mm.py`: 6 tests documenting the candidate-D behavior.

### Key findings — the MM equation for 2D lattice YM is NOT uniquely "exact"

**Finding 1**: The continuum area law `W[C] = exp(−λ Area/2)` does NOT satisfy our MM candidates at machine precision. Best residual with candidate D (LHS = Σ_P W[staple(C,e,P)]/λ, RHS = 2W[C] + splits) is 8.65e-02 at λ=5 — clearly not lattice-exact.

**Finding 2**: The LATTICE answer on an infinite 2D lattice at N=∞ is `W[C] = w_+^Area` where w_+ is the Gross-Witten single-plaquette value:
- Strong coupling (λ ≥ 1): w_+ = 1/(2λ) — exact, no corrections.
- Weak coupling (λ < 1): w_+ = 1 − λ/2.

Against this lattice answer, candidate D residuals shrink: λ=2 → 2e-3; λ=5 → 2.5e-4. **Leading order in 1/λ is correct; there are subleading corrections at O(1/λ³).**

**Finding 3**: Candidate D self-consistent solution w_+^MM = λ − √(λ²−1) matches GW's w_+^GW = 1/(2λ) to leading order:
  w_+^MM = 1/(2λ) + 1/(8λ³) + ...

The O(1/λ³) corrections are where candidate D differs from the physical answer. Which means our candidate D is the LEADING-ORDER MM, not the full MM.

### Why the MM equation isn't unique (physics interpretation)

The Wilson-action lattice MM has multiple equivalent formulations with different coefficient structures depending on which plaquettes are counted, how the self-closure is handled, and whether the "staple" or "symmetric difference" convention is used. Various references (Migdal 1975, Wadia 1981, Anderson-Kruczenski 2017, Kazakov-Zheng 2021) write the equation differently; comparing across them requires explicit convention dictionaries.

For the Phase 1 program, the KEY question isn't "which form is exactly right" but "does the MM loss train the neural network to the correct Wilson loop values?" If the leading-order form is reasonable, the ML can compensate for subleading corrections.

### Decision

- **Adopt candidate D as the working MM equation** for Phase 1b.
- Train neural loop functional by minimizing its residuals.
- Validate: Phase 1b neural-network training should recover W[C] consistent with the GW strong-coupling answer (w_+ = 1/(2λ)) at least to 1-few%.
- If Phase 1b fails (residuals converge to zero but Wilson loops wrong), iterate on MM form.

### Empirical test results

`test_candidate_D_leading_order_at_strong_coupling`: for λ ∈ {2, 5, 10}, max residual < 1/λ². ✓
`test_self_consistent_w_plus_candidate_D`: rel error vs GW < 1/λ² for λ ∈ {2, 5, 10, 100}. ✓
`test_plaquette_insertions_preserves_closure`: all 2D plaquette insertions closed. ✓
`test_gw_wplus_values`: matches standard GW formulas. ✓
All 6 MM tests pass. All 16 pre-existing tests still pass.

### Next step

Phase 1 Step 2: implement `LoopSystem` (precomputed loop enumeration + MM-equation index tables) in `lattice.py`. Then the neural loop functional and training can proceed on top of these precomputed structures.

---

## Implementation-12: Phase 0 — QCD₂ Master Field Infrastructure (Apr 12, 2026)

### What was built

Following the Discussion-11 roadmap, Phase 0 delivers the infrastructure for constructing unitary master field operators in the Cuntz-Fock space and computing lattice Wilson loops.

**New modules:**
- `master_field/lattice.py`: lattice loop encoder with signed step indices `μ ∈ {±1,...,±D}`. Provides backtrack reduction, cyclic canonicalization, hyperoctahedral symmetry orbits (`B_D`), non-self-intersecting loop enumeration, and (preliminary) plaquette-insertion and self-intersection split operators for MM-equation residuals.
- `master_field/qcd2.py`: `solve_alpha_for_plaquette` (brentq root-finding), `validate_wilson_loops` (comparison against exact area law), `validate_mm_equation_exact` (MM residual check against exact area-law Wilson loops), `qcd2_main` acceptance output.
- `master_field/test_qcd2.py`: 18 pytest tests. All pass.

**Extended modules:**
- `cuntz_fock.CuntzFockSpace`: `build_unitary_gaussian(α, μ) = exp(iα(â_μ+â_μ†))` via `scipy.linalg.expm`. `wilson_loop_vev(operators, word)` handles signed step indices (negative → conjugate transpose). `check_unitarity` monitors `‖UU† − I‖_F`.

**Documentation:**
- `reference/qcd2_master_field.md`: physics reference with the exact formulas (area law, axial-gauge Gaussian master field per Gopakumar-Gross §5, lattice MM equation) and the observed Phase 0 behavior.

### Empirical findings

**Unitarity** (machine-level). `‖Û_μ Û_μ† − I‖_F = O(10^{−15})` at every truncation tested. `scipy.linalg.expm` of a Hermitian matrix is unitary to machine precision, even in a truncated Fock space.

**α convergence** (rapid). The fitted α(λ, L) that matches the plaquette converges quickly:

| λ | α(L=4) | α(L=6) | α(L=8) | α(L=10) |
|---|---|---|---|---|
| 1.0 | 0.95181358 | 0.95176911 | 0.95176915 | 0.95176915 |

Stable to 8 decimals by L=8.

**Plaquette** (exact by construction). `|W[□]_ML − e^{−λ/2}| < 10^{−10}` for all tested λ ∈ {0.5, 1, 2, 5}, L ∈ {4,6,8,10}.

**The key physical finding**: the single-α Gaussian ansatz Û_μ = exp(iα(â_μ+â_μ†)) **does not reproduce the area law** for Wilson loops larger than the plaquette. The error saturates at a fixed, L-independent value:

| Loop | λ=1, L=6 | λ=1, L=8 | λ=1, L=10 |
|---|---|---|---|
| 1×1 (plaquette) | 0 (fit) | 0 (fit) | 0 (fit) |
| 2×1 rectangle | 5.1×10⁻³ | 4.9×10⁻³ | 4.9×10⁻³ |
| 2×2 square | 1.35×10⁻¹ | 1.35×10⁻¹ | 1.35×10⁻¹ |

Since α has converged but the Wilson loop errors have not, the deficiency is in the **ansatz form**, not in the truncation. With only one real parameter, the ansatz has 1 degree of freedom; the area law requires W[m×n] = W[□]^{mn} for all m,n, giving infinitely many independent constraints.

### Interpretation

Gopakumar-Gross §5 states that 2D YM has a Gaussian master field in axial gauge. Two possible readings:

1. **Continuum/scaling statement**. "Gaussian" refers to the continuum master field. On a finite Cuntz-Fock truncation with spacetime-independent link operators, the "Gaussian" form `exp(iα(â+â†))` is too restrictive and yields only approximate area law.

2. **Generalized Gaussian**. The correct ansatz might be a Voiculescu-style expansion Û_μ = Σ_n c_{μ,n} (â_μ†)^n + (h.c.) with infinitely many coefficients tuned to reproduce the full Wilson-loop algebra. At the truncation level L, this gives O(L) coefficients per direction — enough to match O(L) independent Wilson loops.

Either way, the implication for our program is the same: **Phase 1 must optimize over a richer ansatz.** Two natural candidates:

- **Polynomial coefficients** `{c_{μ,n}}_{n=0}^{L-1}` trained against MM-equation residuals (Direction B in Discussion-11).
- **Neural loop functional** `f_θ(C, λ)` with architectural equivariances, trained against MM-equation residuals directly (Direction A).

### MM-equation machinery status

The `plaquette_insertions` and `self_intersection_splits` functions in `lattice.py` are implemented but untested for correctness against specific conventions. `validate_mm_equation_exact` currently reports large residuals when evaluated on the exact area law — this is most likely because our insertion convention (replacing link μ with a plaquette closed path) does not match the standard Migdal/Chatterjee convention, which uses edge-set symmetric difference. Deferred to Phase 1 planning when we actually need MM loss functions.

### Phase 0 acceptance

✓ Infrastructure built and tested (18/18 pytest, 16/16 pre-existing tests).  
✓ Plaquette matches exactly to machine precision.  
✓ α(λ, L) converges with L.  
✓ Unitarity preserved.  
✓ Backtrack invariance (`W[(1,-1,2,3,-3,2)] = W[(2,2)]`) holds.  
✓ Exchange symmetry `W[2×1] = W[1×2]` holds.  
✗ Full area-law reproduction for larger loops — deferred to Phase 1 (as expected; single-parameter ansatz is insufficient).

### Next step — Phase 1

Train a multi-parameter ansatz against the full set of Wilson loops up to L_max. Candidate: polynomial Û_μ = exp(i Ĥ_μ) with Ĥ_μ = Σ_n h_n (â_μ + â_μ†)^n (truncated Taylor series with trainable coefficients). This has ~L real parameters per direction. Loss = Σ over loops of |W_ML[C] − e^{−λ·Area(C)/2}|² + unitarity penalty.

Alternatively (Direction A): parametrize W[C] as a neural network with built-in equivariances, train against MM-equation residuals. No Fock space needed for the forward pass; Fock-space checks only for validation.

---

## Discussion-11: Revised Plan — Original Directions Only (Apr 12, 2026)

### What's been done (by others — don't repeat)

- **Bootstrap/SDP** (Kazakov-Zheng, Li-Zhou, Guo-Qiao-Zheng): Rigorous *bounds* on Wilson loop averages from MM equations + positivity, via SDP. Works in D=2,3,4 for SU(∞), SU(2), SU(3). Up to L_max=24. Already implemented, published, well-understood. *We are not re-doing this.*
- **Collective field optimization** (Rodrigues et al.): Master field for Hermitian multi-matrix QM via constrained minimization of V_eff in loop space. Up to ~10⁴ variables. Works for matrix QM but *not* applied to lattice gauge theory.
- **Gopakumar-Gross**: Explicit master field construction in Cuntz-Fock space. Done for QCD₂ (Gaussian in axial gauge) and for independent/coupled Hermitian matrix models. *Never done for lattice YM in D≥3.*

### What has NOT been done — the gap

**Nobody has explicitly constructed the master field (as an operator in a well-defined Hilbert space) for lattice Yang-Mills theory in D≥3.**

The bootstrap bounds the observables. The Rodrigues program finds the master field for matrix QM. But lattice YM in D=3,4 — actual QCD — has no explicit master field construction.

This is the target. The ML is not an end in itself — it's the computational engine for a problem that is well-posed but computationally intractable by brute force.

### The physics setup

**Spacetime-independent master field** (Gopakumar-Gross §1, eq. 1.6-1.7): the master gauge field can be made spacetime-independent by a gauge transformation. On the lattice: master link variables Ū_μ depend only on direction μ, not on site. The entire N=∞ lattice YM theory is encoded in **D unitary operators** Û_1, ..., Û_D in a Cuntz-Fock space.

Wilson loop for closed path C = (μ_1,...,μ_k) with μ_i ∈ {±1,...,±D} and Û_{-μ} = Û_μ†:

W[C] = ⟨Ω| Û_{μ_1} Û_{μ_2} ⋯ Û_{μ_k} |Ω⟩

This is the Gopakumar-Gross framework applied to unitary operators. Fock space has 2D creation operators â_μ†.

### Why this hasn't been done

1. **Unitarity constraint**: Û_μ Û_μ† = 1 as operator equation is nontrivial in truncated Fock space; couples all orders.
2. **Nonlinear loop equations** at N=∞ via factorization: MM equation involves products W[C_1]·W[C_2].
3. **Exponential growth** of distinct loops with length (~618 at L_max=12 in 4D, per Qiao-Zheng).
4. **No small parameter**: D≥3 has genuine interactions; no globally convergent expansion.

### Three genuinely new directions

**Direction A — Neural loop functional**. Parametrize W[C] as a neural network f_θ: loops → ℝ. Transformer/GRU on step sequences with built-in equivariance (cyclic, reversal, hyperoctahedral symmetry B_D). Loss = MM residuals. Coupling λ as additional input — single trained network gives full phase diagram. Deconfinement transition appears as a sharp feature in f_θ(C, λ). **Novelty**: first neural parametrization of the observable W[C] itself. Generalizes beyond truncation (training at length L constrains predictions at L+2 via MM).

**Direction B — Cuntz-Fock master field operators**. Directly construct Û_μ in truncated Cuntz-Fock space. Parametrize Û_μ = exp(i Ĥ_μ) with Ĥ_μ Hermitian. Constraints: MM residuals, unitarity Û Û† ≈ I, PSD correlation matrix (automatic from Fock structure). Loss = Σ|MM|² + μ Σ ||Û_μ Û_μ† - I||². For large truncation, parametrize Ĥ_μ via NN. **Novelty**: the Rodrigues/Jevicki-Sakita program applied to lattice YM (unitary operators), not matrix QM.

**Direction C — Hybrid bootstrap-ML**. Use Kazakov-Zheng bootstrap bounds W⁻[C] ≤ W[C] ≤ W⁺[C] as *hard box constraints* on a neural loop functional, then train against MM residuals. Bootstrap eliminates spurious local minima; ML extrapolates beyond the bootstrap truncation and finds the unique interior point. **Novelty**: uses rigorous bounds as constraints rather than stopping at bounds.

### Implementation phases

- **Phase 0** (QCD₂): validate the Cuntz-Fock unitary master field against the exactly-solvable 2D theory. Gaussian in axial gauge (Gopakumar-Gross §5), W[C] = exp(-λ Area/2).
- **Phase 1** (D=2 neural): train neural loop functional against MM, compare to exact area law. Test architecture, equivariances, generalization to long loops.
- **Phase 2** (D=3): the first unsolved case. All three directions applied in parallel. Validate against published SDP bounds and MC.
- **Phase 3** (D=4 QCD): the target. Extract string tension, glueballs, deconfinement from f_θ(C, λ).

### What makes this original

| Existing work | What it does | What it doesn't |
|---|---|---|
| Kazakov-Zheng SDP | Bounds on W[C] | Not the master field. Only up to L_max. |
| Rodrigues collective field | Master field for matrix QM | Not lattice gauge theory. Hermitian. |
| Gopakumar-Gross | Cuntz-Fock framework + QCD₂ | Not D≥3. Framework, not solution. |
| Normalizing flows for LGT | Sample gauge configurations | Not N=∞. Not master field. |

Our contribution: (A) first neural parametrization of W[C] itself; (B) first explicit master field construction for lattice YM in D≥3; (C) bootstrap bounds as hard ML constraints.

If Direction A or B succeeds in D=4: first explicit construction of the large-N master field for QCD. Problem posed by Witten (1979), formalized by Gopakumar-Gross (1994). 47 years open.

### References

1. Gopakumar-Gross, hep-th/9411021, §1 (spacetime-independent master field), §5 (QCD₂ master field)
2. Gopakumar PhD thesis (1997)
3. Rodrigues et al. JHEP 2022, 2024 — master variable / collective field
4. Kazakov-Zheng 2203.11360 — MM equations on lattice, SDP bootstrap
5. Qiao-Zheng 2601.04316 — systematic loop equation construction
6. Raissi et al., J. Comp. Phys. 378 (2019) — physics-informed NN
7. Han-Hartnoll, Phys. Rev. X 10 (2020) — NN ansatz for matrix model ground states

---

## Discussion-10: Gross-Witten Model — Unitary Loop Equations (Apr 12, 2026)

### The problem

The Gross-Witten-Wadia (GWW) model is the simplest lattice gauge theory — a single-plaquette unitary matrix model:

Z = ∫ dU exp(N/(2t) Tr(U + U†))

with 3rd-order phase transition at t_c = 1. Exact solution known (Gross-Witten 1980, Wadia 1980). Stepping stone to QCD₂ → QCD₃ → QCD₄.

### Why U = exp(iM) does NOT simplify to Hermitian

Initial idea: write U = exp(iM), M Hermitian, giving V(M) = -(1/t)cos(M). Then V'(M) = (1/t)sin(M) and we reuse the Hermitian MasterFieldTrainer.

**This fails.** The path integral measures differ:
- Hermitian: Π(θ_j - θ_k)² (Vandermonde polynomial)
- Unitary (Haar): Π|2 sin((θ_j - θ_k)/2)|² (Vandermonde on circle)

The different Vandermonde gives **different saddle-point equations** and **different SD/loop equations**. Empirically verified: Hermitian SD equations give large residuals (O(1)) when evaluated on exact GWW moments. Tested 4 candidate unitary SD equations — all fail for the weak-coupling (gapped) phase at n ≥ 2.

### Exact solution (verified numerically)

**Strong coupling (t > 1, ungapped):**
ρ(θ) = (1/2π)(1 + (1/t)cos θ) on [-π, π]
w₁ = 1/(2t), wₙ = 0 for n ≥ 2.

**Weak coupling (t < 1, gapped):**
ρ(θ) = (1/(πt)) cos(θ/2) √(t - sin²(θ/2)) on [-θ_c, θ_c], sin²(θ_c/2) = t.
w₁ = 1 - t/2, w₂ = (1-t)².

### Saddle-point equation on the circle

The correct constraint is:
P.V. ∫ ρ(φ) cot((θ-φ)/2) dφ = (1/t) sin θ

Via Hilbert transform on the circle (ungapped phase):
2 Σ_{n≥1} wₙ sin(nθ) = (1/t) sin θ → w₁ = 1/(2t), wₙ = 0 for n ≥ 2.

For the gapped phase, the Hilbert transform doesn't directly give a simple Fourier relation because the density has compact support.

### Open question: the correct moment recursion

The standard Hermitian SD equation Σ_k v_k m_{n+k} = splitting does **not** apply. The unitary loop equation involves:
1. Resolvent on the unit circle (outer R⁺ for |z| > 1, inner R⁻ for |z| < 1)
2. The spectral curve relating R⁺ and R⁻
3. Moment recursion derived from the spectral curve

### Implementation paths

**Path A — Saddle-point as loss:** Parameterize ρ(θ) directly (Fourier coefficients or support endpoint + polynomial). Loss = saddle-point equation residual. Constraint: ∫ρ = 1, ρ ≥ 0. The PV integral is numerically tricky.

**Path B — Toeplitz moment matrix + resolvent:** Parameterize Wilson loops wₙ. Enforce Toeplitz PSD (T_{ij} = w_{|i-j|} ≽ 0) and the resolvent equation. Requires deriving the correct moment recursion.

**Path C — Direct density optimization:** Parameterize the support endpoint a = sin(θ_c/2) and use the known density form ρ ∝ cos(θ/2)√(a² - sin²(θ/2)). The only free parameter is a (determined by the coupling). This reduces to 1D optimization.

### Toward QCD₂

Once GWW is solved (1 unitary matrix), extending to QCD₂ on a small lattice = multiple coupled unitaries with plaquette interactions. The same Toeplitz moment matrix framework applies, just with more matrices and lattice Makeenko-Migdal equations as the loop constraints.

### Reference

See `reference/gross_witten_model.md` for detailed formulas and bibliography.

---

## Implementation-9: Two-Matrix Scaling Study with Symmetry (Apr 6, 2026)

### Symmetry constraints added

Three constraint types pre-computed at init and enforced in the loss:
- **Cyclicity**: tr[w] = tr[cyclic rotation of w] (5 pairs at L=6)
- **Exchange M₁↔M₂**: tr[w(M₁,M₂)] = tr[w(M₂,M₁)] (7 pairs at L=6)
- **Z₂ (M→-M)**: tr[odd-length words] = 0 (10 terms at L=6)

Verification at L=6: |tr[M₁²]-tr[M₂²]| = 0, |tr[M₁M₂]-tr[M₂M₁]| = 0, |tr[M₁]| = 3e-21.

### Scaling study results (g=1.0, commutator-squared, with symmetry)

| L | dim | loss | min eig | tr[M₁²] | tr[M₁M₂] | time |
|---|-----|------|---------|---------|----------|------|
| 4 | 7 | 6.4e-253 | 0.961 | -0.003 | -0.004 | 17s |
| 6 | 15 | 2.2e-34 | 0.572 | 0.179 | 0.335 | 21s |
| 8 | 31 | 6.5e-38 | 0.426 | 0.348 | 0.000 | 40s |

**Not yet converged**: tr[M₁²] changes 0.18→0.35 from L=6 to L=8. Scaling study extended to L=4..16 to find convergence. L=14 (dim=255) est. ~1-2 hr, L=16 (dim=511) est. ~6-12 hr.

### Cluster setup

Standalone `cluster/` package with PBS scripts. Uses EasyBuild modules (`jax/0.4.25-gfbf-2023a`, `SciPy-bundle/2023.07-gfbf-2023a`). Auto-installs `optax` to user site-packages.

---

## Implementation-8: One-Matrix Quartic via scipy SLSQP (Apr 6, 2026)

### Problem

Direct moment parametrization fails for quartic: SD equations are underdetermined (always 2 more unknowns than equations). Without PSD, optimizer finds spurious solutions (SD loss=0 but wrong moments). Continuation in g and relative residuals help stability but don't fix uniqueness.

### Solution

scipy.optimize.minimize with SLSQP + Hankel PSD constraint:
- Objective: SD residuals (relative)
- Constraint: min eigenvalue of even Hankel submatrix G_{ij}=m_{2(i+j)} ≥ 0
- Key fix: Hankel dim must be n_even//2+1 (not n_even) to avoid truncation artifacts making the matrix falsely non-PSD

### Results

All couplings g=0.1..5.0 converge (SD loss < 1e-15). Low moments within 3-5% of exact. Quartic g=0.5 at L=12: m₂=0.614 (exact 0.631), m₄=0.773 (exact 0.738). Remaining error from underdetermined high moments.

---

## Implementation-7: Code Fixes & Training Pipeline (Apr 6, 2026)

### Fixes applied (CLAUDE_CODE_INSTRUCTIONS.md)

All 7 listed fixes implemented. Additionally discovered and fixed:
- **schwinger_dyson.py SD indexing**: used `n+k-1` (wrong), fixed to `n+k`
- **Eigenvalue density formula**: code doubled g-dependent terms. Correct: P(x) = gx² + 1 + ga²/2
- **Free cumulant κ_6**: hardcoded formula was classical, not free. Replaced with general recursion via z·G = 1 + Σ κ_k G^k — works to arbitrary order

### Training results

**Gaussian (Stage 1a)**: Perfect. Loss = 0, all moments match Catalan numbers to machine precision.
**Quartic (Stage 1b)**: Solved via scipy SLSQP (see Implementation-8).
**Two-matrix (Stage 3)**: JIT-compiled trainer converges. Symmetry constraints added (see Implementation-9).

---

## Discussion-5: Scaling Strategies & Future Directions (Apr 6, 2026)

### Scaling to larger truncation

The bottleneck is moment matrix dimension N_Ω ~ n^(L/2) for n matrices at truncation L.

**Strategies:**
- **Symmetry reduction**: Z₂ symmetry (M → -M) halves basis. Exchange symmetry (M₁ ↔ M₂) reduces further.
- **Sparse Cholesky**: L₁ regularisation on near-zero entries.
- **Neural parametrisation**: Neural network θ → L(θ) maps low-dim latent code to Cholesky factor.
- **Stochastic SD**: Sample random SD equation subset per epoch (SGD over constraint space).

### Adding new matrix models

1. Define V'(M) coefficients in `config.py`
2. Add case to `train.py`
3. If multi-matrix: define interaction in `schwinger_dyson.py`
4. If exactly solvable: add exact solution to `one_matrix.py` for validation

### Gross-Witten phase transition

One-plaquette model V(U) = -(1/2g²)(U + U†) for unitary U:
- Weak coupling (g² < 2): single-cut eigenvalue density
- Strong coupling (g² > 2): two-cut (gapped) density
- 3rd-order phase transition at g² = 2

Requires changing from Hermitian to UNITARY matrices. Moments become tr[Uⁿ], SD equations change, master field in Cuntz-Fock space is Û = f(â, â†) with ÛÛ† = 1.

### Toward QCD₄

Ultimate target. On lattice with L_lat sites: 4·L_lat unitary matrices (link variables) satisfying lattice Makeenko-Migdal equations. Bootstrap approach of Kazakov-Zheng (2021) shown computationally feasible for small lattices.

---

## Discussion-4: Known Bugs, Pitfalls & Code Fixes Needed (Apr 6, 2026)

### Known pitfalls

1. **SD equation indexing**: LHS is Σ_k v_k m_{n+k}, NOT m_{n+k-1}. From tr[M^k · M^n] = m_{k+n}.
2. **Cyclic equivalence**: Canonical representative = lexicographically smallest rotation. Without this, redundant variables cause ill-conditioning.
3. **PSD enforcement**: Cholesky diagonal must use L_ii = exp(ℓ_i). Forgetting exponential → diagonal can go negative → Ω not PSD.
4. **Normalisation**: m₀ = tr[I] = 1 is hard constraint. Either fix by construction or add large penalty (λ ~ 100).
5. **Symmetry**: For V symmetric under M → -M, enforce odd-moment vanishing by construction. Otherwise spurious flat directions.
6. **Large-L divergence**: Moments m_{2k} grow factorially (~(2k)!/(k!(k+1)!) for Gaussian). Loss must use RELATIVE residuals.
7. **JAX**: Use `jax_platform_name='cpu'` if no GPU. JIT-compile training step. Use float64 (`jax_enable_x64=True`). Use parametric Cholesky, not jnp.linalg.cholesky.
8. **Moment matrix vs. vector**: Moment MATRIX Ω_{ij} = tr[w_i† w_j] is N_Ω × N_Ω. PSD constraint applies to MATRIX, not the moment vector.

### Code fixes needed

**Fix 1: `neural_master_field.py` — SD loss function**
`sd_loss_one_matrix` has incorrect indexing. Correct: LHS = Σ_k v_k * m_{n+k}, RHS = Σ_{j=0}^{n-1} m_j * m_{n-j-1}. Loop should be `for n in range(0, K - max_v_degree)`.

**Fix 2: `neural_master_field.py` — Gaussian initialisation**
Initial parameters should give exact Gaussian solution: m_{2k} = C_k (Catalan) = binom(2k,k)/(k+1). For quartic, initialise at Gaussian (g=0) and let optimizer deform.

**Fix 3: `neural_master_field.py` — Multi-matrix Cholesky**
`MultiMatrixTrainer.moment_from_matrix` uses Python loop (not JIT-compatible). Fix: pre-compute word → (i,j) index mapping at init, store as static array for JIT.

**Fix 4: `schwinger_dyson.py` — Two-matrix SD**
`TwoMatrixSD.sd_residuals` splitting sum is correct but doesn't properly handle V' interaction terms for commutator-squared. Full V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a - 2 M_b M_a M_b).

**Fix 5: `bootstrap_sdp.py` — Linearisation**
Current bootstrap has bilinear terms (not SDP). Proper approach: define Hankel moment matrix H_{ij} = m_{i+j}, SD equations become linear in H, impose H ≽ 0 as SDP constraint.

**Fix 6: Add float64 throughout**
Add `jax.config.update("jax_enable_x64", True)` at top of every JAX file.

**Fix 7: Implement Voiculescu coefficient extraction**
After training: moments → free cumulants κ_n (moment-cumulant formula) → Voiculescu coefficients M_n = κ_{n+1} → build operator in Cuntz-Fock space → verify ⟨Ω|M̂^k|Ω⟩ = m_k.

---

## Discussion-3: Computational Pipeline — Stages 0–4 (Apr 6, 2026)

### Stage 0: Sanity checks (run FIRST, must ALL pass)

```bash
python master_field/cuntz_fock.py
```
Expected: `✓ Cuntz algebra verified (n=1, L=6, dim=7)`, Gaussian moments = Catalan numbers (m₀=1, m₂=1, m₄=2, m₆=5, m₈=14, m₁₀=42), all to < 10⁻¹².

```bash
python master_field/one_matrix.py
```
Expected: Gaussian free cumulants κ₁=0, κ₂=1, κ_{k≥3}=0. Quartic (g=0.5) moments m₂≈0.5162, m₄≈0.4838.

```bash
python master_field/schwinger_dyson.py
```
Expected: Gaussian SD residuals max < 10⁻¹². Quartic SD residuals max < 10⁻⁶.

### Stage 1: One-matrix models (exactly solvable — MUST match)

**1a. Gaussian**: `python train.py --model gaussian --validate --max_word_length 12 --n_epochs 3000 --lr 1e-2`
Success: all even moments match Catalan numbers to < 10⁻⁶. Loss < 10⁻¹⁰. κ₂ = 1.0000, all others ≈ 0.

**1b. Quartic**: Run at g = 0.5, 1.0, 5.0 with --validate. Moments match exact values (eigenvalue density integration) to < 10⁻⁴.

**1c. Coupling scan**: g ∈ {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}. R-transform coefficients κ_n(g) should be smooth in g.

### Stage 2: Bootstrap validation

`python train.py --model quartic --coupling 0.5 --bootstrap --max_word_length 10`
SDP bootstrap produces rigorous bounds. ML solution must lie within them.

### Stage 3: Two coupled matrices (the real challenge)

`python train.py --model two_matrix_coupled --coupling 1.0 --max_word_length 6 --n_epochs 20000 --lr 5e-4 --interaction commutator_squared`
Word space grows as 2^L. No closed-form solution. At L=6: basis dim ~100, Cholesky has ~5000 parameters.
Success: Loss < 10⁻⁴, min eig(Ω) > -10⁻⁸, stable under increasing L.
Benchmark: Rodrigues et al. JHEP 2024.

### Stage 4: Scaling study

L = 4, 6, 8, 10 at fixed coupling. First few moments (m₂, m₄) should stabilise quickly; higher moments need larger L.

### Output specification

Results directory: `results/`
- `moments_{model}_g{g}.npy` — optimised moments
- `free_cumulants_{model}_g{g}.npy` — free cumulants
- `losses_{model}_g{g}.npy` — training loss history
- `moment_matrix_{model}_g{g}.npy` — full moment matrix (multi-matrix)
- `convergence_{model}_g{g}.png` — loss vs epoch
- `moments_{model}_g{g}.png` — ML vs exact moments
- `eigenvalue_density_{model}_g{g}.png` — reconstructed ρ(x) vs exact
- `sd_residuals_{model}_g{g}.txt` — final SD residuals

---

## Discussion-2: Physics Framework & Schwinger-Dyson Equations (Apr 6, 2026)

### The master field problem

For matrix model with action S[M₁,...,Mₙ] invariant under M_i → U M_i U†, large-N factorisation gives:
⟨O₁ O₂⟩ = ⟨O₁⟩⟨O₂⟩ + O(1/N²)

The path integral concentrates on a single configuration — the master field M̄_i:
⟨O⟩ = tr[M̄_{i₁} M̄_{i₂} ⋯ M̄_{iₖ}]

### What we optimise

Loop moments Ω(C) = tr[M̄_{i₁} ⋯ M̄_{iₖ}] for word C = (i₁,...,iₖ). These satisfy:
1. **Schwinger-Dyson equations** — exact recursion from the action
2. **Positive semidefiniteness** — moment matrix Ω_{ij} = tr[w_i† w_j] ≽ 0
3. **Cyclicity** — Ω(C) = Ω(cyclic permutations of C)
4. **Hermiticity** — Ω(C)* = Ω(C⁻¹)
5. **Normalisation** — Ω(∅) = 1

### SD equations

Single matrix with potential V(M): for test word M^n:
LHS = Σ_k v_k m_{n+k}  (where V'(M) = Σ_k v_k M^k)
RHS = Σ_{j=0}^{n-1} m_j m_{n-j-1}

**CRITICAL**: LHS index is n+k, NOT n+k-1. From tr[V'(M)·M^n] = Σ_k v_k tr[M^{k+n}].

Two matrices with S = Tr[M₁²/2 + M₂²/2 - (g²/4)[M₁,M₂]²]:
V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a - 2 M_b M_a M_b)

SD for derivative w.r.t. M_a on test word w = M_{i₁}⋯M_{iₖ}:
⟨tr[V'_a · w]⟩ = Σ (over positions m where i_m = a) ⟨tr[w_left]⟩⟨tr[w_right]⟩

### Loss function

L = L_SD + λ_EOM · L_EOM + λ_reg · L_reg
- L_SD = Σ_n |SD residual_n|²
- L_EOM = Σ_ψ |⟨Ω|[V'(M̂) - 2Π̂]|ψ⟩|²
- L_reg: spectral regularisation for eigenvalue non-negativity

### PSD constraint via Cholesky

Ω = LL† with L lower-triangular. Neural network outputs L; Ω automatically PSD.

### References

1. SD equations: Gopakumar-Gross §2.5 (eq. 2.30-2.35)
2. Master field operator: Gopakumar-Gross §2.3 (eq. 2.16-2.18)
3. Hermitian representation: Gopakumar-Gross §3.1 (eq. 3.46-3.47)
4. Master field EOM: Gopakumar-Gross §2.5 (eq. 2.33) and §4 (eq. 4.69-4.70)
5. Cholesky/master variables: Rodrigues et al. JHEP 2022, §2
6. SDP bootstrap: Anderson & Kruczenski, Nucl. Phys. B 921 (2017); Lin, JHEP 2020
7. Free cumulants/R-transform: Voiculescu, Dykema, Nica, "Free Random Variables" (1992)

---

## Implementation-1: Project Setup, Structure & Dependencies (Apr 6, 2026)

### Mission

Numerically construct the master field for large-N matrix models using ML optimisation. The master field is the N=∞ saddle point of the matrix path integral; all gauge-invariant observables are computable from it without functional integration. An explicit construction in QCD₄ would be one of the most important results in theoretical physics.

### Dependencies

```bash
pip install jax jaxlib numpy scipy matplotlib optax flax cvxpy torch --break-system-packages
```

JAX for autodiff + JIT (compiles inner loop to XLA), cvxpy for SDP validation (SCS/MOSEK solvers), PyTorch available as alternative.

### Project structure

```
master_field/
├── config.py                  # Hyperparameters, model registry
├── cuntz_fock.py              # Cuntz algebra, Boltzmann Fock space, operator reps
├── one_matrix.py              # Exact one-matrix solutions (resolvent, density, moments)
├── schwinger_dyson.py         # Loop equations for multi-matrix models
├── neural_master_field.py     # Neural ansätze + JAX training loops
├── bootstrap_sdp.py           # SDP bounds via cvxpy
├── train.py                   # CLI entry point
├── visualize.py               # Plots
├── test_master_field.py       # Automated test suite
└── results/                   # Output directory
```

### Test suite

Run: `python master_field/test_master_field.py`
Tests: Cuntz algebra relations, Gaussian moments (Catalan), Gaussian free cumulants, SD indexing verification, quartic SD consistency, free product relations, PSD constraint, Voiculescu roundtrip.
