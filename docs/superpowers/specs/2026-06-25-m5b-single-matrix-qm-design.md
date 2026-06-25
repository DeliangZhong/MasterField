# M5b — Single-matrix quantum mechanics as a bootstrap/collective-field sandwich

**Date:** 2026-06-25  **Status:** design (scope approved: full sandwich; spec under review)

**Goal:** Construct the large-N master field of single-matrix QM `H = Tr P² + Tr X² + (g/N)Tr X⁴`
by squeezing the ground-state energy density `E/N²` between a certified SDP lower bound and a
collective-field variational upper bound, with the exact free-fermion solution as referee —
extending the M5a QM machinery to a genuine large-N matrix model (with the SU(N) Gauss law).

**Where this sits.** Second rung of M5. M5a built the QM machinery (energy, stationarity,
`[x,p]=i`, oscillator Fock, the certified sandwich gate) on N=1. M5b lifts it to large N: the
new ingredients are the **SU(N) Gauss law**, **'t Hooft scaling**, and the **collective field**
(the singlet master field). The single matrix is "solvable" (free fermions) — like M1's
semicircle — so it is the validation rung before M5c (two-matrix QM, the unsolvable target,
where the operator master field is the open research question). **M5c is out of scope here.**

---

## Conventions (pinned — derived, not guessed; cross-checked against HHK arXiv:2004.10212)

- **Model (HHK Eq 8):** `H = Tr P² + Tr X² + (g/N) Tr X⁴`, X,P Hermitian N×N, `g ≥ 0` fixed as
  N→∞ ('t Hooft). `ℏ=1`.
- **Canonical (HHK):** `[P_ij, X_kl] = −i δ_il δ_jk` (so `[X_ij, P_kl] = i δ_il δ_jk`).
- **SU(N) Gauss law (HHK Eq 11–13):** `G = i[X,P] + N·I`; singlet states satisfy `⟨Tr(G·O)⟩=0`.
  **Derived consequence** (`Tr XP = Tr PX + Σ_ij[X_ij,P_ji]`, `Σ_ij[X_ij,P_ji]=iN²`, `⟨Tr XP⟩`
  imaginary): `⟨Tr XP⟩ = iN²/2`. The symplectic remnant of `[X,P]` that survives at large N.
- **'t Hooft scaling (derived for `E/N²` finite):** `X = √N X̃`, `P = √N P̃`, `λ = √N y`
  (eigenvalue). Then `[X̃,P̃] = (1/N)·(i δ…)` (commuting at leading order) but the *ordered*
  moment `m[X̃P̃] = (1/N²)⟨Tr XP⟩ = i/2` survives.
- **Normalized moments:** `m[w] = (1/N)⟨Tr(w)⟩` for a word `w` in `{X̃, P̃}`, all `O(1)`;
  `m[∅]=1` (i.e. `tr I = 1`, `tr ≡ (1/N)Tr`). Hermiticity: `m[w]* = m[reverse(w)]`. Parity
  (`X→−X`, `P→−P` each a symmetry): any word with odd total count of `X̃` or of `P̃` vanishes.
- **Energy density (derived):** `E/N² = m[P̃²] + m[X̃²] + g·m[X̃⁴]`. (At `g=0` the virial gives
  `m[P̃²]=m[X̃²]` so `E/N²=2m[X̃²]=2·½=1`.)

### Verified exact anchors (free fermions = collective field; see de-risk + `derivations/`)

| g | `E/N²` | `m[X̃²]=⟨X²⟩/N²` | `m[X̃⁴]=⟨X⁴⟩/N³` |
|---|---|---|---|
| 0.0 | **1.00000** | **0.50000** | 0.50000 |
| 0.5 | 1.18049 | 0.37943 | 0.28110 |
| 1.0 | 1.30190 | 0.33143 | 0.21301 |
| 2.0 | 1.48047 | 0.28161 | 0.15288 |

`g=0`: `E/N²=1` is exact for all N (`Σ_{n=0}^{N-1}(2n+1)=N²`, filling the `−∂²+λ²` levels).

---

## Architecture — the sandwich (mirrors M5a, lifted to large N)

```
   E_lo (SDP, certified)  ≤   E/N²(g)   ≤   E_var (collective field)
                              ‖  exact (free fermions, referee)
```

### A. Collective-field master field — variational upper bound (the operator field at large N)
The singlet sector of `H = Tr P² + Tr V(X)` is N free fermions; the large-N master field is the
**rescaled eigenvalue density** `σ(y)` (`∫σ=1`, `σ≥0`). Derived energy functional
(Jevicki–Sakita; fermion phase-space filling `p_F=πσ`, kinetic density `π²σ³/3`):
$$\frac{E}{N^2}[\sigma] = \int\!\Big[\tfrac{\pi^2}{3}\sigma(y)^3 + (y^2 + g\,y^4)\,\sigma(y)\Big]dy,\qquad \int\sigma=1,\ \sigma\ge0.$$
This convex functional's minimizer is the master field. **Variational upper bound:** minimize over
a parametrized/ML density ansatz `σ_θ` (positivity + normalization enforced) → `E_var ≥ E/N²`,
converging to exact as the ansatz improves. The analytic minimizer `σ(y)=(1/π)√(μ−y²−g y⁴)` (μ
from `∫σ=1`) is the exact large-N answer. Moments `m[X̃^{2k}] = ∫y^{2k}σ`.

### B. SDP bootstrap — certified lower bound (HHK Eq 9–15)
- **Variables:** normalized moments `m[w]`, `w` a word in `{X̃,P̃}` up to length `L` (HHK: L=3
  converges), modulo hermiticity + parity (odd-count words = 0) + cyclicity.
- **Constraints:**
  1. `m[∅]=1`; Gauss law `m[X̃P̃]=i/2` (and its descendants).
  2. **Stationarity** `⟨[H,O]⟩=0` for each single-trace `O` → linear relations among `m[w]`
     (the QM loop equations). Anchor example (HHK Eq 10, = virial `2⟨T⟩=⟨xV'⟩`):
     `m[P̃²] = m[X̃²] + 2g·m[X̃⁴]`. The full set (and the `[X̃,P̃]` reorderings + large-N
     factorization, HHK Eq 14) is derived and **verified against the free-fermion moments** in
     `derivations/m5b-single-matrix-qm.md`.
  3. **Large-N factorization:** double-trace terms `⟨tr A·tr B⟩ = ⟨tr A⟩⟨tr B⟩` (leading order),
     linearized by a product matrix as in `_bootstrap_two_matrix` (`G[0,k]=m_k`, `G⪰0`).
  4. **Gram positivity:** `M[u,v] = m[reverse(u)·v] ⪰ 0` over a word basis (GNS positivity).
- **Energy bounds:** minimize / maximize `E/N² = m[P̃²]+m[X̃²]+g·m[X̃⁴]` over the feasible set →
  certified `[E_lo, E_hi]` (reusing `bootstrap_sdp._solve`/`TRUSTED_SOLVERS`).

### C. Exact referee — free fermions
Independent of A and B: (i) the collective analytic `σ=(1/π)√(μ−V)` (table above); (ii) a
finite-N **level-filling** cross-check — diagonalize the single-particle `h=−∂²_λ+λ²+(g/N)λ⁴`,
fill N levels, `E/N²`, extrapolate N→∞ to the collective value. A,B,C must all agree.

---

## Module plan (`matrix_master_field/`)

- **`qm_collective.py` (new):** the collective-field master field.
  - `collective_energy_density(sigma, ys, g)` — the `E/N²[σ]` functional (trapezoid).
  - `collective_master_field(g)` — analytic minimizer `(μ, σ(y), E/N², m2, m4)`.
  - `collective_variational(g, ansatz)` — minimize `E/N²[σ_θ]` over a positive normalized ML
    density ansatz → variational upper bound + the master-field density.
  - `free_fermion_energy(g, N)` — finite-N level-filling cross-check (single-particle diag).
- **`bootstrap_sdp.py` (extend):** `bootstrap_single_matrix_qm(g, L, *, target, maximize, with_status)`
  — words in X̃,P̃; stationarity + Gauss law + factorization (product matrix) + Gram PSD; bounds
  on `E/N²` (and `m[X̃²]`). Helper `_sm_qm_stationarity(...)` building the loop equations.
- **`train.py` (extend):** `solve_single_matrix_qm(g, ...)` — collective variational upper bound
  + SDP lower bound + free-fermion referee; pure fail-closed gate `_sm_qm_gate` (validated iff a
  certified SDP island brackets the free-fermion `E/N²` AND the collective `E_var` is consistent).
- **`derivations/m5b-single-matrix-qm.md` (new):** the 't Hooft scaling, the Gauss-law derivation,
  the collective functional derivation, and the stationarity loop equations — each verified
  numerically against the free-fermion moments.
- **Tests:** `tests/test_qm_collective.py`, `tests/test_bootstrap_single_matrix_qm.py`,
  `tests/test_train_single_matrix_qm.py`.

---

## Validation / proof obligations

| # | Claim | Check |
|---|---|---|
| V1 | collective `E/N²=1`, `m2=½` at g=0 | analytic + functional evaluation |
| V2 | collective = free-fermion level-filling | finite-N diag → N→∞ matches the table (≥3 digits) |
| V3 | virial `m[P̃²]=m[X̃²]+2g m[X̃⁴]` | holds on the exact moments (residual ~0) |
| V4 | Gauss law `m[X̃P̃]=i/2` | derivation + use in the SDP |
| V5 | stationarity loop equations | each verified on free-fermion moments (residual ~0) |
| V6 | SDP brackets truth | `E_lo ≤ E/N²(g) ≤ E_hi`, certified; tightens with L; for g∈{0,0.5,1,2} |
| V7 | collective variational bound | `E_var ≥ E/N²`, → exact as the ansatz refines |
| V8 | the sandwich | `E_lo ≤ E/N² ≤ E_var`, all bracketing the free-fermion referee → `validated` |

Verification medium: Python + `pytest` (limits, residuals, convergence), as M1–M5a.

---

## Risks / open questions

- **R1 — the SDP setup is the main risk.** Mixed `X̃,P̃` word moments + the `[X̃,P̃]` reorderings
  + large-N factorization + the Gauss law make the single-matrix-QM bootstrap materially more
  intricate than M5a's powers-of-x. Mitigation: derive every stationarity relation explicitly and
  **verify each against the exact free-fermion moments** before trusting the SDP; start at L=3
  (HHK's convergence order) and grow.
- **R2 — bootstrap tightness / solver conditioning.** As in M5a, reuse the margin-SDP +
  trusted-solver hygiene; the bound is rigorous regardless of tightness.
- **R3 — collective variational ansatz.** Enforce `σ≥0`, `∫σ=1` (e.g. softmax-style
  parametrization); the bound is variational (upper) for any valid ansatz.

## Out of scope
- **M5c** (two-matrix QM, HHK Eq 17): the unsolvable target; its operator master field is the open
  research question — own brainstorm after M5b.
