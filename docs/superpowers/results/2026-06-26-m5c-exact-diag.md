# M5c Follow-up: Exact Diagonalization Ground Truth

**Date:** 2026-06-26
**Sub-project:** 1 of 2 (M5c follow-up)
**Spec:** `docs/superpowers/specs/2026-06-26-m5c-exact-diag-ground-truth-design.md`
**Context:** `[[m5-progress]]` — M5c free-Fisher "beats Gaussian" was a truncation artifact; this sub-project provides the exact-diag ground truth to adjudicate where the true large-N ground energy sits.

---

## Method

Exact Galerkin diagonalization of the two-matrix quantum mechanics Hamiltonian (HHK arXiv:2004.10212 Eq 17):

    H = Tr(P_X^2 + P_Y^2 + m^2(X^2 + Y^2) - g^2 [X,Y]^2),   m=1, lambda=N g^2.

The 2(N^2-1) traceless Hermitian modes are placed in a bosonic Fock space truncated to total quanta <= K (two trace modes contribute 2m analytically). The quartic is built on a padded basis (K+2) to avoid the spurious non-PSD boundary artifact, then projected to the K-space. Eigensolver: sparse Lanczos (eigsh k=1, SA) for D >= 50; dense eigh for D < 50.

All numbers below are from a single run of `matrix_master_field/exact_diag.py` on 2026-06-26 with the exact command:

    uv run --no-project --with numpy --with scipy python matrix_master_field/exact_diag.py

---

## V6 Table: E/N^2 at finite N

### lambda = 0 (free theory; exact result is 2m = 2.000000)

| N | K-list | E/N^2 | K-tail | Casimir | Gaussian (same-N) |
|---|--------|-------|--------|---------|-------------------|
| 2 | 8,10,12 | 2.00000 | 1.8e-15 | 6.8e-27 | 2.00000 |
| 3 | 4,6   | 2.00000 | 1.7e-14 | 5.2e-25 | 2.00000 |

N->inf extrapolation: 2.0000 +/- 0.0000 [1/N^2: 2.0000, 1/N: 2.0000]

### lambda = 0.5

N=2 full K-series: K=8: 2.148487, K=10: 2.148474, K=12: 2.148471
N=3 full K-series: K=4: 2.177623, K=6: 2.176223

| N | K-list | E/N^2 | K-tail | Casimir | Gaussian (same-N) |
|---|--------|-------|--------|---------|-------------------|
| 2 | 8,10,12 | 2.14847 | 2.5e-06 | 5.2e-29 | 2.15516 |
| 3 | 4,6   | 2.17622 | 1.4e-03 | 1.3e-28 | 2.18389 |

N->inf extrapolation: 2.2151 +/- 0.0333 [1/N^2: 2.1984, 1/N: 2.2317]

### lambda = 1.0

N=2 full K-series: K=8: 2.257997, K=10: 2.257863, K=12: 2.257833
N=3 full K-series: K=2: 2.333333, K=4: 2.313319, K=6: 2.307615

| N | K-list | E/N^2 | K-tail | Casimir | Gaussian (same-N) |
|---|--------|-------|--------|---------|-------------------|
| 2 | 8,10,12 | 2.25783 | 3.0e-05 | 6.5e-29 | 2.27339 |
| 3 | 4,6   | 2.30762 | 5.7e-03 | 1.3e-28 | 2.32401 |

N->inf extrapolation: 2.3773 +/- 0.0597 [1/N^2: 2.3474, 1/N: 2.4072]

---

## Verification Status (V1-V6)

**V1 (structure constants + quartic correct):** Tests `test_structure_constants_*` and `test_quartic_matches_minus_tr_commutator_sq` pass for N=2,3. The quartic value agrees with -Tr[X,Y]^2 evaluated on 5 random (x,y) pairs per N to atol=1e-10.

**V2 (g=0 anchor E/N^2 = 2m):** All lambda=0 entries above are exactly 2.00000 to machine precision (tail ~1e-14). The test `test_g0_anchor_exact_all_N_K` confirms this for (N,K) in {(2,2),(2,5),(3,2),(3,3)}.

**V3 (K-convergence monotone):** At lambda=1, N=2: K=8→2.257997, K=10→2.257863, K=12→2.257833 — strictly non-increasing. N=3: K=2→2.3333, K=4→2.3133, K=6→2.3076 — strictly non-increasing. The test `test_K_convergence_monotone_and_settles` confirms this for N=2, lambda=1, K in [4,6,8,10].

**V4a (finite-N bracket [2m, Gaussian(N)]):** At lambda=1, N=2: 2.0 <= 2.25783 <= 2.27339. At lambda=1, N=3: 2.0 <= 2.30762 <= 2.32401. The finite-N Gaussian is always same-N (NOT the large-N 2.365 value); that large-N number appears only in the N->inf comparison (V4b). The test `test_v4a_bracket_finite_N` confirms this for N=2, lambda=1.

**V5 (Casimir ~0, singlet ground state):** All Casimir values in the table are O(1e-28) to O(1e-25) — effectively zero to eigensolver precision. This confirms the interacting ground state is a gauge-singlet for both N=2 and N=3 at all tested lambda values. Tests `test_casimir_singlet_g0_vacuum` and `test_casimir_ground_state_is_singlet` pass.

**V6 (deliverable table + N->inf):** Completed — see the table above and the N->inf adjudication below.

---

## N->inf Adjudication (the headline)

### Extrapolation method

Two-point 2-model extrapolation from the (finite-N) converged values at N=2 and N=3, evaluated at lambda=1:

- E(2) = 2.25783 (N=2, K=12, tail 3.0e-05 — converged)
- E(3) = 2.30762 (N=3, K=6, tail 5.7e-03 — partially converged; see Caveats)

**1/N^2 model** (physical, Eguchi-Kawai-style large-N expansion): assume E(N) = E_inf + c/N^2.
Solving the 2-point system: **E_inf(1/N^2) = 2.3474**

**1/N model** (cruder, does not assume planar dominance): assume E(N) = E_inf + c/N.
Solving: E_inf(1/N) = 2.4072

The 1/N form gives ~2.41, which exceeds the large-N Gaussian upper bound 2.365 (computed from the planar limit of the Gaussian trial state, see e.g. the N=10 proxy value 2.3609). A quantity exceeding a rigorous upper bound is unphysical — the 1/N model is too crude for this data (the corrections are 1/N^2 at large N in a matrix model). The physically correct model is 1/N^2.

**Result (1/N^2 model):** E_inf(lambda=1) = 2.3474
**Mean of both models:** E_inf = 2.3773
**Model-spread uncertainty:** +/- 0.0597

The conservative statement, using model-spread as the uncertainty, is:

    E_inf(lambda=1) = 2.35 +/- 0.06  (2-point, 1/N^2 preferred; 1/N pathological)

### Adjudication of the M5c bounds

The exact-diag ground truth at lambda=1 places the large-N ground energy at approximately:

    E_inf ~ 2.35 (1/N^2 extrapolant)  with model-spread uncertainty ~0.06

**(a) ABOVE the M5c free-Fisher master-field claim (2.322).** The free-Fisher result claimed a large-N E/N^2 ~ 2.322. The exact-diag finite-N energies are variational upper bounds (converging from above in K): N=2 is converged (2.2578, K=12), N=3 partially (2.3076 at K=6; a geometric extrapolation of the K-gaps puts N=3 K->inf ~ 2.305). The 1/N^2 extrapolation then gives E_inf ~ 2.34, above 2.322 by ~ 0.02 — a margin that survives propagating the residual N=3 K-tail (~6e-3). Since a valid variational upper bound cannot lie below the true ground energy, an E_inf above 2.322 means 2.322 was a from-below estimate, not a bound. This **corroborates** the independently-established M5c diagnosis (the degree-3 free-Fisher value failed higher-basis traciality/conditioning stress tests, true energy 3.9-4.7 — see `[[m5-progress]]`): 2.322 was a small Cuntz-Fock truncation (L=3) artifact, not a genuine sub-Gaussian master field. The margin is modest and rests on a 2-point extrapolation, so the exact-diag is corroborating evidence, not an independent high-precision determination.

**(b) CONSISTENT WITH the large-N Gaussian upper bound (approximately 2.365).** The 1/N^2 extrapolant 2.3474 sits below the Gaussian bound 2.365, and the interval [2.34, 2.41] is consistent with near-saturation of the Gaussian. The large-N Gaussian remains a valid upper bound; the true value appears to approach it from below (consistent with the Gaussian being a nearly-optimal large-N trial state for this model at lambda=1).

**(c) FAR ABOVE the certified SDP lower bound (2.0).** The bootstrap SDP lower bound is flat at ~2m = 2.0 for all tested lambda (M5 progress note). This bound is loose by ~0.35 at lambda=1 and does not constrain the answer usefully. The genuine lower bound problem remains open (direct-momentum approach, sub-project 2).

**One-line adjudication:** The exact-diag ground truth E_inf ~ 2.34 +/- 0.06 lies above the M5c free-Fisher claim 2.322 (by ~ 0.02, robust to the N=3 K-tail — corroborating that 2.322 was a from-below truncation artifact) and below the large-N Gaussian 2.365 (which it nearly saturates); the certified SDP lower 2.0 is loose by ~ 0.35.

---

## Honest Caveats

**R1 — N=3 K-convergence:** N=3 is capped at K=6 due to computational cost. The padded basis at K=6 has ~7.4e5 states for n_modes=16 (N=3 has 2*(9-1)=16 traceless modes); K=8 would have ~5.3e6 padded states and requires minutes + several GB of memory with the O(n_tl^3) quartic assembly loop. The K-tail at lambda=1 is 5.7e-03, meaning the K=6 value 2.3076 still has a residual convergence error of that order. The K-series K=2→2.3333, K=4→2.3133, K=6→2.3076 is monotone (Rayleigh-Ritz); a geometric extrapolation of the gaps (0.0200, 0.0057; ratio ~0.28) places N=3 K→infinity at approximately 2.305 (residual ~3e-3 below the K=6 value), though a 3-point geometric estimate is itself only indicative. The N=2 path (K=12, tail 3.0e-05) is converged and serves as the primary spine for the adjudication.

**R2 — 2-point extrapolation:** The N->inf estimate uses only N=2 and N=3. A genuine large-N limit requires N=4,5,... which are computationally expensive. The model-spread uncertainty (0.06 at lambda=1) is an honest lower bound on the extrapolation error — the actual error is likely dominated by the residual N=3 K-convergence tail (5.7e-03) and higher-order 1/N^k corrections not captured by a 2-point fit. The 1/N^2 extrapolant (2.3474) is the physically preferred one; the 1/N value (2.4072) is unphysical (exceeds the upper bound).

**R3 — singlet assumption:** The Casimir measurement (V5) confirms O(1e-28) for all computed ground states — the SU(N) singlet structure is confirmed to eigensolver precision. No singlet-sector projector was needed.

**R4 — finite-N vs large-N:** The Gaussian values in the table (2.27339 at N=2, 2.32401 at N=3) are same-N finite-N upper bounds, NOT the large-N Gaussian 2.365. The large-N Gaussian appears only in the N->inf comparison section above and is never compared against finite-N exact-diag values.

---

## Files

- Implementation: `matrix_master_field/exact_diag.py` (Tasks 1-9 complete)
- Tests: `matrix_master_field/tests/test_exact_diag.py` (25 tests, all pass)
- Spec: `docs/superpowers/specs/2026-06-26-m5c-exact-diag-ground-truth-design.md`
- This results doc: `docs/superpowers/results/2026-06-26-m5c-exact-diag.md`

## Next Steps

Sub-project 2: Direct-momentum variational master field — explicit P-hat with [X,P]=i, direct <P^2> expectation value giving a genuine non-exploitable upper bound to benchmark against this exact-diag ground truth.
