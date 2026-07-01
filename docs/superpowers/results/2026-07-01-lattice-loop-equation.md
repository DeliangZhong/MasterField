# Exact large-N lattice loop equation (dimension-general) — the QCD₃ unblock

**Date:** 2026-07-01 · **Branch:** `matrix-master-field` · **Status:** derived + verified (machine-zero, D=2)

**One line.** The QCD₃ "Q2 block" was mis-scoped as *transcribe the D=3 loop equations*. The real
problem: the lattice Makeenko–Migdal equation hard-coded as "candidate D" (`cluster/lattice.py::
build_loop_system`) is only the **leading strong-coupling term** — it omits the quadratic *contact*
term. I derived and verified the exact equation; it is **dimension-general**, so D=3 needs no new
physics, only the larger perpendicular direction set.

## The bug in candidate D

On the exact D=2 area-law master field (`W[C]=w₊^{Area}`, `w₊=1/(2λ)`, strong coupling `λ≥1`),
candidate D's per-edge residual is **exactly**

```
res_candidateD(C) = (1/λ) · w₊^{Area(C)+1}
```

— verified by instrumenting `plaquette_insertions`. It vanishes only as λ→∞. Candidate D sums the two
link-detour staples with weight 1/λ and a `2·W[C]` self-term, but **drops the contact term**.

## The exact equation

For a Wilson loop `C` with base link `μ=C[0]`, remainder `Ĉ=C[1:]`, at N=∞ and `λ≥1`:

```
(1/λ)  Σ_ν W(B_ν)          # link-detour staples,     B_ν = (ν, μ, −ν) + Ĉ
 − 2    W(C)                # self / trivial term
 −(1/2λ) Σ_ν W(A_ν)        # plaquette-prepended CONTACT, A_ν = (μ, ν, −μ, −ν) + C
 −      Σ_s σ_s W(C₁)W(C₂)  # genuine self-intersection splits of C (0 for simple loops)
 = 0
```

`ν` runs over the `2(D−1)` signed directions perpendicular to `|μ|`. At N=∞ each `A_ν` is a
figure-eight touching `C` only at the base point, so it **factorizes**:
`W(A_ν) = W(plaquette_{μν}) · W(C)` — that factorization is the contact term candidate D omitted.

### Verification

`cuntz_bootstrap/lattice_loop_eq.py` (self-contained; `python -m cuntz_bootstrap.lattice_loop_eq`):

```
lam= 1.00: 324 simple loops, max|residual| = 0.000e+00
lam= 2.00: 324 simple loops, max|residual| = 0.000e+00
lam= 5.00: 324 simple loops, max|residual| = 1.301e-18
lam=10.00: 324 simple loops, max|residual| = 5.133e-19
```

All 324 non-self-intersecting D=2 loops up to length 12, applied at **every** edge (3680 equations),
machine-zero for λ≥1. (λ<1 fails because the area law itself is not exact off the strong-coupling
phase — not the regime of interest.) The oracle is the factorized N=∞ area law (free ⇒ `τ(ab)=τ(a)τ(b)`
at self-touch points).

### Consistency with the working QCD₂ solver

For the unit plaquette `C=(1,2,−1,−2)` the equation reduces to

```
(1/λ)[W_empty + W_{1×2}] − 2 W_plaq − (1/λ) W_plaq² = 0
```

— identical to `cuntz_bootstrap/qcd2_q2.py::plaquette_mm_residual` (the "Impl-32" equation the
*working* unsupervised D=2 Q2 already uses). So the exact equation is not new at D=2; the point is that
its **structure is dimension-general** and the repo's general `build_loop_system` had the wrong form.

## Why this unblocks D=3

The D=3 plaquette equation (base edge μ=1 of the 12-plane plaquette) is:

- **in-plane** detours `ν=±2`: contract → empty, extend → 1×2 rectangle (the D=2 terms);
- **out-of-plane** detours `ν=±3`: `(±3,1,∓3,2,−1,−2)` — 6-link non-planar loops **absent at D=2**;
- contacts `A_ν`: the 12-plane figure-8s **plus** 13-plane figure-8s `(1,±3,−1,∓3,1,2,−1,−2)`.

The `ν=±3` terms are exactly the perpendicular-plaquette fluctuations that correct the naive
plane-by-plane area law at **O(1/λ⁴)** — the genuine D=3 dynamics, and what should be checked against
the Münster strong-coupling character expansion.

## Next

1. Wire the D=3 plaquette MM equations (3 planes) + factorization identities into an **unsupervised**
   Q2 loss (mirror `qcd2_q2.py`, replace the single D=2 plaquette residual with the D=3 term set from
   `lattice_loop_eq.loop_equation_terms(C, D=3)`), run the operator field from random init.
2. Verify the recovered `W[plaq](λ)` against the Münster strong-coupling expansion to O(1/λ⁴).
3. Then large Wilson loops / QCD₃ vs the bootstrap bound (arXiv:2502.14421) — the actual goal.

**Deferred (not needed for the minimal Q2):** the genuine self-intersection split terms `σ_s W(C₁)W(C₂)`
have a subtle base-point/sign structure (probed, not fully pinned); the D=2 Q2 template sidesteps them
via explicit factorization identities, and D=3 will do the same.

## Files

- `cuntz_bootstrap/lattice_loop_eq.py` — exact D-general equation + self-contained D=2 verification.
- Supersedes the candidate-D form in `cluster/lattice.py::build_loop_system` (leading-order only).
