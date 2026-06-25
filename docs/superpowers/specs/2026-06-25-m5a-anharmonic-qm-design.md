# M5a — Single-particle anharmonic oscillator as a bootstrap/operator-field sandwich

**Date:** 2026-06-25  **Status:** design (approved in brainstorming; spec under review)

**Goal:** Build the quantum-mechanics master-field machinery on the smallest exactly-checkable
model — `H = p² + x² + g x⁴` — by squeezing the ground-state energy between a certified SDP
lower bound and a variational (operator-field) upper bound, validated against exact
diagonalization.

**Where this sits.** M1–M4 solved matrix *integrals* (loop/SD equations, free Cuntz–Fock,
positivity automatic, validated in certified SDP islands). M5 is the step to matrix *quantum
mechanics* (a Hamiltonian, time, canonical `[X,P]=i`, an energy `E`). M5a is the warm-up rung of
the three-rung M5 ladder; it introduces every new QM ingredient on an N=1 model with a hard exact
answer, so the machinery is unit-tested before the large-N rungs.

- **M5a (this spec):** single particle, `H = p² + x² + g x⁴` (HHK Eq 1).
- **M5b (next spec):** single-matrix QM, `H = Tr P² + Tr X² + (g/N)Tr X⁴` (HHK Eq 8); large-N,
  SU(N) Gauss law; anchored on the free-fermion/collective-field exact solution.
- **M5c (later):** two-matrix QM, `H = Tr(P_X²+P_Y²+m²(X²+Y²)−g²[X,Y]²)` (HHK Eq 17); the
  genuinely-unsolvable target. Its operator-field representation is an open research question, to
  be brainstormed after M5a/M5b validate the machinery. **Out of scope here.**

---

## Conventions (pinned — do not deviate without updating this block and CONVENTIONS.md)

Source: Han–Hartnoll–Kruthoff, *Bootstrapping Matrix Quantum Mechanics*, arXiv:2004.10212.

- **Model (HHK Eq 1):** `H = p² + x² + g x⁴`, coupling `g ≥ 0`. Coefficients are exactly
  `1·p² + 1·x² + g·x⁴` (no ½'s).
- **Units:** `ℏ = 1`.
- **Canonical commutator:** HHK write `[p,x] = −i`, i.e. **`[x,p] = +i`**.
- **State:** an energy eigenstate `|E⟩`, `H|E⟩ = E|E⟩`; the deliverable is the ground state
  `E₀(g)`. Expectations `⟨O⟩ = ⟨E|O|E⟩`. Moments `m_k ≡ ⟨x^k⟩`, `m_0 = 1` (normalization, hard
  constraint, never optimized).
- **Parity:** `V(x)=x²+gx⁴` is even ⟹ eigenstates have definite parity ⟹ **all odd moments
  vanish** (`m_{2k+1}=0`) in any eigenstate. Enforced by construction.
- **Oscillator representation (operator-field side):** `x̂ = (â+â†)/√2`, `p̂ = −i(â−â†)/√2`,
  with `[â,â†] = 1`. Then `[x̂,p̂] = i` holds *identically* (verified below). NB this is the
  **bosonic** Fock space `[â,â†]=1`, distinct from the M1–M4 **free** Cuntz–Fock `ââ†=1`.

### g=0 anchors (derived, exact)
With `x̂,p̂` as above, `H₀ = p̂²+x̂² = 2â†â + 1`, so the spectrum is `E_n = 2n+1` and
`E₀(0) = 1`. Ground-state `m₂ = ⟨0|x̂²|0⟩ = ½`. These are the first validation targets.

*(Derivation of `H₀=2â†â+1`: `x̂²+p̂² = ½[(â+â†)² − (â−â†)²] = ½·2(ââ†+â†â) = 2â†â+1`, using
`[â,â†]=1`. Two-line identity, declared trivial.)*

---

## Derivations (the correctness core)

All three relations below are **re-derived from scratch** here (not transcribed), and will be
restated in `matrix_master_field/derivations/m5a-anharmonic-qm.md` with a `pytest` verification
that plugs exact-diagonalization moments into each relation and checks the residual is ~0.

### D0 — canonical commutator (trivial, declared)
`[x̂,p̂] = [(â+â†)/√2, −i(â−â†)/√2] = (−i/2)[â+â†, â−â†] = (−i/2)(−2[â,â†]) = i`. ✓

### D1 — energy relation (sandwiching p²)
`p² = H − x² − g x⁴`. In an eigenstate, `⟨x^{t-1}H⟩ = E⟨x^{t-1}⟩`, so
$$\langle x^{t-1} p^2\rangle = E\,m_{t-1} - m_{t+1} - g\,m_{t+3}. \tag{D1}$$
(HHK Eq 5.) This carries `E` into the system.

### D2 — Heisenberg commutators (building blocks)
From `[p,x]=−i` and Leibniz: `[p,x^t] = −i\,t\,x^{t-1}`, hence
`[H,x^t] = [p²,x^t] = −i\,t\,(p\,x^{t-1} + x^{t-1} p)`. Also
`[H,p] = [x²+gx⁴, p] = i(2x + 4g x³)` (= `i V'(x)`; Heisenberg's `ṗ=−V'`).

### D3 — stationarity recursion (closes the system)
Stationarity `⟨[H,O]⟩ = 0` with `O = x^t p`:
`[H,x^t p] = [H,x^t]p + x^t[H,p] = −i\,t\,(p x^{t-1} + x^{t-1} p)p + i\,x^t(2x+4gx³)`.
Taking `⟨·⟩=0`:
$$t\big(\langle p x^{t-1} p\rangle + \langle x^{t-1} p^2\rangle\big) = 2 m_{t+1} + 4g\,m_{t+3}. \tag{R2}$$
Reduce `⟨p x^{t-1} p⟩` with `[x,p]=i`:
`p x^{t-1} p = x^{t-1}p² − i(t-1)x^{t-2}p`, and (from `⟨x^{t-2}p+px^{t-2}⟩=0`, the `O=x^{t-1}`
stationarity, plus `⟨[x^{t-2},p]⟩ = i(t-2)m_{t-3}`) one gets
`⟨x^{t-2}p⟩ = (i/2)(t-2)m_{t-3}`, hence
`⟨p x^{t-1} p⟩ = ⟨x^{t-1}p²⟩ + ½(t-1)(t-2)m_{t-3}`. Substituting this and D1 into (R2):
$$\boxed{\,4tE\,m_{t-1} + t(t-1)(t-2)\,m_{t-3} - 4(t+1)\,m_{t+1} - 4g(t+2)\,m_{t+3} = 0\,} \tag{D3}$$
This reproduces **HHK Eq 6** exactly — an independent re-derivation that also cross-validates the
pinned conventions. Consequences used downstream:
- `t=1`: `4E − 8m₂ − 12g\,m₄ = 0 ⟹ m₄ = (E − 2m₂)/(3g)` (g>0).
- The recursion expresses every even moment `m_{2k}` as an affine function of `m₂` with
  `E`-dependent coefficients, given `m₀=1`. **Free parameters: `(E, m₂)`.** The g=0 limit gives
  `E = 2m₂`, consistent with `E₀=1, m₂=½`.

---

## Architecture — the sandwich

```
   E_SDP_lower(K)   ≤   E₀(g)   ≤   E_var(K)
   └─── bootstrap ──┘            └─ operator field ─┘
        (certified)                  (variational)
              both → E₀(g) as K → ∞ ; cross-checked by exact diagonalization
```

### A. SDP bootstrap (certified lower bound + (E,⟨x²⟩) island)
- **Variables:** `E` and even moments `m₂, m₄, …, m_{2K}` (`m₀=1`, odd = 0).
- **Equality constraints:** the recursion (D3) for `t = 1,3,5,…` up to the truncation.
- **Positivity:** Hankel moment matrix `M_{ij} = m_{i+j}` (`i,j = 0..K`) `⪰ 0`. (Localizing
  matrices `x·M`, `(x²-shift)` may be added later for tightness; not required for the first cut.)
- **Bilinearity handling:** `E·m_{t-1}` is bilinear. Resolve exactly as M1 does — *fix `E`*, which
  makes (D3) linear and the moments affine in `m₂`, so `M(E,m₂) ⪰ 0` is an LMI; **bisect on `E`**
  for the min/max feasible energy, and at fixed `E` solve min/max `m₂` for the island width. The
  feasible `E`-range is `[E_SDP_lower, E_SDP_upper]`; bisection edges certified via the existing
  `_solve` + `_select_solver` (MOSEK/CLARABEL) + status check.
- **Reuse:** `bootstrap_sdp.py` (`_solve`, `_select_solver`, `_mosek_usable`, `TRUSTED_SOLVERS`,
  certification status).

### B. Operator / variational master field (rigorous upper bound + the state)
- Build `â,â†` as truncated `(K+1)×(K+1)` ladder matrices (`â†|n⟩=√(n+1)|n+1⟩`, truncated),
  then `x̂,p̂` and `Ĥ = p̂² + x̂² + g x̂⁴` as dense float64 matrices.
- **Master field = ground eigenvector** `|Ω⟩` of `Ĥ_trunc`. `E_var(K) = λ_min(Ĥ_trunc)` is the
  Rayleigh–Ritz variational minimum — a rigorous **upper** bound converging down to `E₀(g)` as
  `K→∞`. Positivity (`|Ω⟩` is a genuine unit vector) and `[x̂,p̂]=i` are automatic.
- **Observables:** `m_{2k} = ⟨Ω|x̂^{2k}|Ω⟩`.
- **ML framing (light here, honest):** for the single particle the variational optimum *equals*
  `λ_min`, so the primary solve is eigendecomposition. A JAX gradient-descent solve over `|Ω⟩`
  (to mirror the operator-field methodology) is an optional consistency demo, not the deliverable.
  The genuine ML operator-field novelty is deferred to the large-N rungs (M5b/M5c).

### Sandwich validity
`E_SDP_lower(K) ≤ E₀(g) ≤ E_var(K)`, with `E_SDP_lower ↑ E₀` and `E_var ↓ E₀` as `K→∞`. Exact
diagonalization supplies `E₀(g)` to machine precision (the ground state converges fast in `K`),
giving an independent referee for both bounds.

---

## Module plan (`matrix_master_field/`)

- **`qm_fock.py` (new):** truncated bosonic oscillator Fock space.
  - `ladder(K)` → `(a, adag)` `(K+1)×(K+1)` float64/complex matrices.
  - `xp_operators(K)` → `(X, P)` with `X=(a+adag)/√2`, `P=-i(a-adag)/√2`.
  - `hamiltonian_anharmonic(K, g)` → `Ĥ = P@P + X@X + g·X⁴` (Hermitian).
  - `ground_state(K, g)` → `(E_var, Ω)` via `eigh`; `moment(Ω, k)` → `⟨Ω|X^k|Ω⟩`.
  - JAX float64 (`jax.config.update("jax_enable_x64", True)` at top), mirroring the package.
- **`bootstrap_sdp.py` (extend):** `bootstrap_qm_anharmonic(g, K, *, target, maximize, with_status)`
  — builds the recursion-constrained Hankel SDP; returns the `E`-bound (or `⟨x²⟩`-bound at fixed
  `E`) with solver/status. Helper `qm_recursion_coeffs(E, g, K)` returns the affine moment map.
- **`train.py` (extend):** `solve_qm_anharmonic(g, K, *, validate=True)` — runs the variational
  solve (operator field), assembles the sandwich, and applies a **fail-closed gate**:
  `validated = True` only if `E_SDP_lower ≤ E_var`, the exact-diag `E₀` ∈ `[E_SDP_lower, E_var]`,
  and both island edges are certified by a trusted solver. Returns `{E_var, E_lo, E_hi, m2_*,
  E_exact, validated, …}`.
- **`derivations/m5a-anharmonic-qm.md` (new):** full step-by-step of D0–D3 + the verification
  recipe (plug exact-diag moments → residuals ~0).
- **Tests:** `tests/test_qm_fock.py`, `tests/test_bootstrap_qm.py`, `tests/test_train_qm.py`.

---

## Validation / proof obligations (every claim → a check)

| # | Claim | Check |
|---|---|---|
| V1 | `[x̂,p̂]=i` on the interior | `[X,P]` equals `iI` exactly on the upper-left `K×K` block (levels 0..K−1); only the `(K,K)` entry deviates (`=−iK`, a pure truncation artifact). Assert the interior block `= iI`. |
| V2 | g=0 ground state exact | `E₀=1` and `m₂=½` are **exact at any K** (`\|0⟩` sits at the untruncated bottom: `H₀\|0⟩=\|0⟩`, `⟨0\|x̂²\|0⟩=½`); the low excited levels `→ 1,3,5,…` as `K` grows |
| V3 | recursion D3 is correct | plug exact-diag moments `m_{2k}(g)` + `E₀(g)` into D3 → residual < 1e-10 for `t=1,3,5,…` |
| V4 | energy relation D1 | same, residual < 1e-10 |
| V5 | exact-diag converges | `E₀(g)` stable to ≥6 digits as `K` grows (e.g. K=30→60), for `g∈{0.5,1,2}` |
| V6 | SDP brackets truth | `E_SDP_lower(K) ≤ E₀(g) ≤ E_SDP_upper(K)`; island shrinks with `K`; edges certified |
| V7 | variational bound | `E_var(K) ≥ E₀(g)`, monotone ↓ in `K`, `→ E₀` |
| V8 | the sandwich | `E_SDP_lower ≤ E₀ ≤ E_var`; report the gap closing with `K` |
| V9 | `⟨x²⟩` island | SDP `(E,⟨x²⟩)` island contains the exact-diag `(E₀, m₂)` point |

Optional external cross-check: compare `E₀(1)` against the literature value reported by HHK for
`g=1` (to be read off precisely, not guessed).

Verification medium follows the established project pattern (Python + `pytest` numerical checks:
limiting cases, residuals, convergence), as used throughout M1–M4 — not a Mathematica notebook.

---

## Out of scope / deferred

- **M5b (single-matrix QM):** large-N planar single-trace moments of words in `X,P`; SU(N) Gauss
  law `G=i[X,P]+N𝟙`, `⟨Tr XP⟩=iN²/2` (HHK Eq 11–13); free-fermion/collective-field anchor.
  Reuses `qm_fock.py` + the QM SDP extension. Own spec.
- **M5c (two-matrix QM, the target):** SDP extends directly (HHK Eq 17); the **operator-field
  representation of a coupled-matrix ground state is unresolved** and will be brainstormed
  separately. Not committed here.

## Risks / open questions

- **R1 — SDP tightness at small `K`:** the bare Hankel constraint may give a loose `E_lower`.
  Mitigation: add localizing matrices; increase `K`. The bound is rigorous regardless of tightness.
- **R2 — bisection robustness:** the fixed-`E` LMI feasibility can be near-degenerate at the island
  edges (cf. the two-matrix CLARABEL `static_regularization`/`max_iter` tuning from M3). Reuse the
  same solver hygiene.
- **R3 — honest novelty:** M5a's operator field coincides with textbook variational
  diagonalization; it is explicitly the machinery-validation rung, not a novelty claim. Stated
  plainly in the result writeup.
