# M4 result — Kazakov–Zheng "unsolvable" two-matrix model as an operator master field

**Date:** 2026-06-25. **Model** (arXiv:2108.04830, eq. 6), transcribed verbatim:
$$S = N\,\mathrm{tr}\!\left[\tfrac12(A^2+B^2) + \tfrac{g}{4}(A^4+B^4) - \tfrac{h}{2}[A,B]^2\right]$$
couplings **g** (quartic), **h** (commutator); symmetries A↔B exchange and **Z₂×Z₂**
(A→−A, B→−B independently); normalized trace τ=(1/N)tr. This is a *second*
"unsolvable" two-matrix model — distinct from the M3 commutator+mass model — used to
show the operator master-field method generalizes.

## Derived force (verified)

`V'_A = ∂_A tr V` from the elementary identities `∂_A tr[A,B]² = 2[B,[A,B]]`,
`[B,[A,B]] = 2BAB − B²A − AB²`:
$$V'_A = A + g\,A^3 + h\,(A B^2 + B^2 A - 2 B A B),\qquad V'_B \text{ by } A\!\leftrightarrow\! B.$$
Verified by **finite difference** against `∂_A tr V` on random Hermitian A,B (max
error 5e-8 = the FD truncation error; `test_kz_force_finite_difference`). The planar
loop equation `τ(V'_A·w) = Σ_{k: w_k=A} τ(w_{<k})τ(w_{>k})` is the M3 commutator
residual **plus a quartic term `g·τ(AAA·w)`**, with commutator coefficient `h`.

## Implementation (reuses the whole M1–M3 stack)

- `loss.two_matrix_sd_core(moment, words, mass, quartic, comm)` — one general
  two-matrix loop-equation residual; commutator = (1,0,g²/2), **KZ = (1,g,h)**.
- `bootstrap_sdp._bootstrap_two_matrix(..., mass, quartic, comm)` — the SDP island,
  generalized the same way; `bootstrap_two_matrix_kz(g,h)` wrapper.
- `train.solve_kz_sparse(field, g, h, ...)` — sparse Cuntz–Fock + **suffix-shared
  moments** + Z₂×Z₂/exchange/cyclicity losses + the **fail-closed certified gate**,
  with a coupling-homotopy ramping (t·g, t·h) from the exact g=h=0 free field.

## Validation

| check | result |
|---|---|
| force `V'_A` vs `∂_A tr V` (finite difference) | max err 5e-8 ✓ |
| residual at g=h=0 (two free semicirculars) | < 1e-18 ✓ |
| SDP **g=0** reduces to the commutator island | bit-for-bit equal ✓ |
| SDP **h=0** brackets the *exact* quartic ⟨tr M²⟩ | exact 0.6312 ∈ [0.6180, 0.6329] ✓ |
| operator solve **max_word_len=2** (dim 127) | tr A²≈0.631 OUTSIDE tight island → `validated=False` (artifact, correctly rejected) |
| operator solve **max_word_len=3** (dim 1023) | **tr A²=0.55700 ∈ certified [0.53909, 0.56192]**, sd 6e-23, sym 4e-26 → **`validated=True`** |

(All at the representative point g=0.5, h=0.3; island certified by MOSEK at L=8.)

**Headline:** the KZ master field is constructed as a Cuntz–Fock operator, solving the
exact nonlinear loop equations to machine zero with positivity + Z₂×Z₂/exchange
automatic, landing inside our own MOSEK/CLARABEL-certified SDP island — exactly as for
the M3 commutator model. The method is **not specific to one model**.

## Remaining (M4)

- **Neural amortization M̂(g,h)** across the coupling plane (the other half of M4 /
  novelty-(ii)) — extend `amortized.py` to two matrices using the suffix-shared
  evaluator.
- Scan the (g,h) phase structure; match KZ's published 6-digit moments (their tables
  are in figures, not extracted here) as an external cross-check.
- Path: M5 = matrix quantum mechanics → BFSS/BMN.
