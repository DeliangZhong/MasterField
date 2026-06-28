# M5c follow-up sub-project 2 — limits of the direct-momentum master field (findings)

**Date:** 2026-06-28 · **Status:** sub-project 2 PAUSED (no-go cluster + one positive theorem; the
milestone result is sub-project 1's exact-diag ground truth). · **Branch:** `matrix-master-field`

**One-paragraph summary.** Sub-project 2 set out to build a genuine, reusable operator master field
for the two-matrix QM (HHK arXiv:2004.10212 Eq 17) with an *explicit conjugate momentum* `P̂`
(`[X,P]=i`), giving a variational **upper** bound that — unlike M5c's from-below free-Fisher `Φ*` —
cannot be exploited downward, and that could legitimately beat the Gaussian (large-N `2.36452` at
λ=1). Three independent rigor checks (a Codex adversarial audit + two derivation/numerics gates)
established that **a certified, better-than-Gaussian direct-momentum master field is not reachable by
the natural constructions**, and *why*. The investigation produced one genuine positive theorem (the
large-N moment-flow closes) and a cluster of sharp, mechanism-level no-go results. Combined with
sub-project 1 (the exact-diag ground truth), the milestone has a complete, honest characterization of
the large-N energy: `E/N² ≈ 2.34` (exact diag, N→∞ extrapolation), inside the certified bracket
`[2.0 (SDP lower), 2.365 (Gaussian upper)]`.

---

## Goal (what SP2 attempted)

Replace M5c's kinetic estimate `¼Φ*` — a **from-below** functional (`Φ*=sup_h{…}`, so a truncated
basis under-counts the kinetic energy and the optimizer exploits it; the "degree-3 beats the Gaussian
2.322<2.365" claim was a truncation artifact, `docs/superpowers/results/2026-06-25-m5c-two-matrix-qm.md`)
— with an explicit momentum so `⟨P̂²⟩` is a **direct** moment of a genuine state and `⟨H⟩` is a true
variational upper bound. Two realizations were designed and tested: a **pure large-N moment-flow**
(spec v1) and, after that failed, a **finite-N variational** `e^{iÂ}|G⟩` with the flow as surrogate
(spec v2, `docs/superpowers/specs/2026-06-27-m5c-direct-momentum-master-field-design.md`).

---

## The one positive result — S1: the large-N moment-flow closes

*(Derivation + numerics: `matrix_master_field/derivations/sp2-moment-flow-closure.md`,
`sp2_flow_test.py`.)*

For a single-trace Hermitian generator `Â` (correct 't Hooft normalization `Â=N²τ(â)`, derived and
verified N-independent to `1e-14`), the single-trace moments evolve under the Heisenberg flow by a
**closed-at-large-N** nonlinear hierarchy:
- one CCR contraction yields one trace; degree-≥2 generators produce genuine **double-trace**
  operators at O(1), which **factorize** via large-N factorization `⟨τ(u)τ(v)⟩=⟨τ(u)⟩⟨τ(v)⟩+O(1/N²)`;
- the `1/N²` factorization rate was measured directly: `C_N·N²` flat at `0.500` for N=2,3,4
  (Makeenko–Migdal / Jevicki–Sakita; gate doc §2.3–2.4).

So the `N→∞` single-trace flow exists and is closed. **This is real and reusable:** the flow is a
sound, cheap, Fock-free **surrogate/initializer** for any finite-N variational search and a
consistency check on `N→∞` extrapolation. It is *not*, however, a bound (next section).

---

## No-go results (with mechanisms)

### S2 — the truncated large-N flow is not a bound

*(Same gate doc, Part 3.)* The `L`-truncated planar-flow energy does **not** equal `⟨ψ|H|ψ⟩` for any
state (it drops the longer moments the flow feeds into and uses factorization, exact only at `N=∞`).
Two structural obstructions to rescuing it:
- **One-sided positivity.** Bootstrap moment-matrix PSD certifies a **lower** bound (a relaxation —
  the SDP we already have, flat/loose at `2.0`); it can *never* upper-bound `E₀` from the trial. An
  upper bound requires an actual state and its exact energy.
- **Unsafe under optimization.** The per-`Â` truncation error is uncertain-sign, but minimizing the
  computed (non-bound) energy over `Â` **selects** the under-estimating direction → the result drifts
  *below* `E₀` — the same selection mechanism that produced the M5c artifact (without the `Φ*` sup).

### R1 — no scalable exact non-Gaussian upper bound (so finite-N staging is Fock-limited)

*(Derivation + numerics: `matrix_master_field/derivations/sp2-r1-feasibility.md`,
`sp2_r1_degree_growth.py`.)* The finite-N variational bound `E_hi=⟨G|e^{-iÂ}He^{iÂ}|G⟩` is exact and a
true upper bound, but evaluating it for **non-Gaussian** `Â` has no scalable, Fock-free route:
- **Degree-counting lemma:** `deg((ad_Â)^n H) = 2 + n(deg Â − 2)`. **Quadratic** `Â` → degree constant
  → the Heisenberg series **resums** to the linear (Bogoliubov) map — verified resumming to `e^{∓s}`
  (K-residual `5.8e-6→5.3e-12`) — but that family **is the Gaussian** (squeezed states): no
  improvement. **Cubic+** `Â` → degree grows `+(deg Â − 2)`/order → **no finite closure** (operator
  norms blow up super-exponentially).
- **No remainder bound:** `Â` is unbounded; the generic cubic drives a Riccati flow
  `X̃(s)=X̃₀(1+sX̃₀)⁻¹` with a **finite-`s` pole** on a finite-measure set of Gaussian configurations
  (`P(λ_min(X̃₀)≤−1)` = 24% at N=3 → 71% at N=8), so the BCH series **diverges at `s=1`** — an
  asymptotic series gives no one-sided remainder bound, losing the upper-bound property.
- **Structured-state / other routes** collapse to Gaussian or revert to the exponential Fock space;
  the only Fock-free exact object is the moment-flow, already not a bound (S2).

**Consequence:** `E_hi` is computable exactly only in the Fock space → only at **N≤3**, where exact
diagonalization (SP1) already returns the *true* ground energy. A single trial `e^{iÂ}|G⟩` in the same
space is `≥` that exact value — **strictly looser**. So finite-N staging yields no better number than
SP1; its only deliverable would be a reusable variational object that is numerically dominated.

---

## Synthesis

A certified, better-than-Gaussian operator master field for this two-matrix QM, built from an explicit
momentum via the natural constructions (pure large-N flow, or finite-N `e^{iÂ}|G⟩`), **does not
exist** within reach:
- the only **exactly computable** family is the Gaussian (quadratic `Â`);
- everything richer is either **not a bound** (truncated large-N flow — S2) or **Fock-limited to
  N≤3**, where it is dominated by the exact diagonalization (R1).

This is consistent with the SP1 finding that the Gaussian is **nearly saturated** (true `E/N² ≈ 2.34`
vs Gaussian `2.365`, a ~1% gap): there is very little room for a non-Gaussian improvement, and the
constructions that could capture it are precisely the ones that lose rigor or scalability.

---

## What stands (the milestone result)

- **Sub-project 1 — exact-diag ground truth** (`docs/superpowers/results/2026-06-26-m5c-exact-diag.md`):
  `E/N²` = 2.2578 (N=2, converged), 2.3076 (N=3, K=6), N→∞ extrapolation ≈ 2.34 ± 0.06 at λ=1; V1–V6
  verified; corroborates that the M5c free-Fisher 2.322 was a from-below artifact.
- **Certified bracket** `[2.0 (bootstrap SDP, loose), 2.36452 (Gaussian, rigorous)]` at λ=1.
- **S1 closure theorem** — the large-N single-trace moment-flow closes; a reusable surrogate.
- **Three M5c-style artifacts prevented** before shipping (the audit + two gates).

---

## Honest open directions (hard; not pursued here)

- A **fundamentally different** certified-upper-bound construction (not `e^{iÂ}|G⟩`, not the moment-flow)
  — no clear route is known; the obstructions above are structural, not incidental.
- The **lower-bound** problem: tighten the bootstrap SDP past the flat `2m=2.0` (higher `L` does not
  help — a commuting configuration `m[[X̃,Ỹ]²]=0` stays feasible; `MEMORY.md`). Separate, hard.
- The direct-`P̂` idea remains the conceptually right target, but every realization tried reduces to
  one of the above; a genuinely new handle would be required.

## Pointers
- Closure / S2: `matrix_master_field/derivations/sp2-moment-flow-closure.md` (+ `sp2_flow_test.py`).
- R1: `matrix_master_field/derivations/sp2-r1-feasibility.md` (+ `sp2_r1_degree_growth.py`).
- Spec (v2, finite-N staging): `docs/superpowers/specs/2026-06-27-m5c-direct-momentum-master-field-design.md`.
- SP1 ground truth: `docs/superpowers/results/2026-06-26-m5c-exact-diag.md`; M5c origin:
  `docs/superpowers/results/2026-06-25-m5c-two-matrix-qm.md`.
