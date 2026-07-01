# Vehicle B decisive experiment — large-observable reach (KZ two-matrix, master field vs bootstrap)

**Date:** 2026-07-01 · **Branch:** `matrix-master-field` · **Status:** done — thesis **partially falsified** (honest, load-bearing)

Tests the program's core claim on a model with ground truth: does the operator/Cuntz–Fock master
field deliver *reliable* large-observable values (long words) that the convex bootstrap cannot reach?
Model: Kazakov–Zheng (arXiv:2108.04830 eq.6) `S=N·tr[½(A²+B²)+(g/4)(A⁴+B⁴)−(h/2)[A,B]²]`, g=h=1;
observable family `m[A^{2k}]` (word length 2k). Anchor: `m[A²]∈[0.4204,0.4224]` (level-12 island).
Code: `matrix_master_field/kz/kz_large_observables.py`; data `results/kz_large_observables.csv`.

The operator field is a construction: one fitted configuration → `τ(W)` for any word via matvecs.
The bootstrap can only bound `m[W]` at cutoff `≥ len(W)`, and its SDP blows up exponentially in the
cutoff. The experiment is gated on three validations before any "reach" claim.

## The decisive table

| obs | operator (deg-5, Fock-conv.) | boot@6 | boot@8 | boot@10 | verdict |
|---|---|---|---|---|---|
| m[A²] | 0.4201 | [0.390,0.557] | [0.394,0.431] | **[0.4200,0.4285]** | inside ✓ |
| m[A⁴] | 0.3342 | [0.189,0.451] | [0.313,0.357] | **[0.3300,0.3384]** | inside ✓ |
| m[A⁶] | 0.3295 | [0.238,0.596] | [0.278,0.339] | **[0.3188,0.3296]** | inside ✓ (upper edge) |
| m[A⁸] | 0.3642 | — | [0.304,0.447] | **[0.3335,0.3549]** | **OUTSIDE ✗ (+0.009)** |
| m[A¹⁰] | 0.4328 | — | — | **[0.3804,0.4306]** | **OUTSIDE ✗ (+0.002)** |
| m[A¹²…¹⁶] | 0.54 / 0.70 / 0.93 | — | — | — | bootstrap unreachable; operator **unvalidated** |

`—` = the bootstrap SDP cannot represent this moment at that cutoff. Bootstrap brackets are rigorous
*outer* bounds, so a value outside a bracket is provably wrong.

## Three validation prongs

- **(A) Ansatz convergence — FAILS for large k.** deg-3 vs deg-5 at fixed Fock-L=10 agree at
  small k (A²: 0.4197/0.4201) but diverge with k (A⁸: 0.340/0.364; A¹⁶: 0.68/0.93). The operator
  large-observables are *not* converged in polynomial degree beyond word length ~6.
- **(B) Fock convergence — OK.** deg-5 values are Fock-converged: |Δ(L=10,12)| < 1×10⁻³ through A¹⁶.
  So Fock-L is *not* the bottleneck at these lengths (the fitted A has bounded support). The
  bottleneck is the **ansatz degree**, not the Fock cutoff.
- **(C) Agreement — holds only to word length ~6.** The operator value is inside the rigorous
  bootstrap bracket for A², A⁴, A⁶ (validated; and it is a *point* where the bootstrap gives a
  width-0.011 bracket at A⁶). For A⁸ and A¹⁰ it falls **outside** the rigorous bracket — the deg-5
  operator value is **provably wrong** (overshoots).

## Conclusion — the naive thesis is falsified; the real lever is narrower

On this Hermitian matrix model the master field does **not** give reliable large observables "for
free." Its trustworthy reach is set by the fit/ansatz horizon (~word length 6 at W=3, deg-5), and
beyond it the values are rigorously wrong. Notably the deg-5 fit drives the truncated (W=3) loop
residual to machine zero yet still mispredicts A⁸ — **zero truncated-loop residual does not pin the
long-word behaviour**, the same under-determination Part C measured in moment space
(`2026-07-01-relaxation-selection.md`). Extending the reliable reach costs a richer ansatz
(higher degree / larger W) — the operator-field analogue of raising the bootstrap cutoff. No free lunch.

What *does* survive as genuine advantage (validated regime only):
1. **Point, not bracket** — a single value where the bootstrap gives an interval (A⁶: 0.3295 vs
   [0.319,0.330]).
2. **One configuration → all observables** — the bootstrap re-solves an exponentially growing SDP
   per target moment.
3. **Potential scaling** — the matrix-free/sparse-Fock lift (`cuntz_bootstrap/matfree_expm.py`) can
   reach higher effective degree/Fock at fixed cost; whether that beats the bootstrap's cutoff cost
   for a *reliable* long word is the open, quantitative question.

**Implication for QCD.** Do not assume large-Wilson-loop reach from this test — it argues the opposite
for a generic Hermitian model. The QCD case is structurally different (Wilson loops are traces of
*bounded, unitary* link operators, where a low-complexity master field may represent long loops far
better than deg-k polynomials represent high moments of an unbounded Hermitian A). That advantage, if
real, must be demonstrated with the unitary structure — it is not inherited from here.

## Files
- `matrix_master_field/kz/kz_large_observables.py` — the experiment (cached fits, memory-safe
  high-Fock evaluation, ansatz/Fock/bootstrap validation).
- `results/kz_large_observables.csv` — operator values + bootstrap brackets per word length.

## Next
Either (a) push the operator reach with higher degree/W + the matrix-free lift and re-measure the
reliable horizon vs the bootstrap's cost at equal accuracy (the quantitative "who scales better"),
or (b) move to the QCD₂ unitary-link master field where the bounded-operator structure is the actual
setting for large Wilson loops — the more direct path to the 2502.14421 comparison.
