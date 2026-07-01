# Non-relaxation constraint handling & master-field selection (massive D=2 LZ QM)

**Date:** 2026-07-01 · **Branch:** `matrix-master-field` · **Status:** minimal path (Part A + Part C) done

Controlled experiment on the massive D=2 Lin–Zheng two-matrix QM (arXiv:2507.21007),
`H = Tr[½(Π_I²+M²X_I²) − ¼[X₁,X₂]²]`, `M²=1` — non-free, not analytically solvable, but with
a level-14 rigorous island known to ~8 digits (the **referee only**, never a solver input):
`𝓔∈[1.172098376,1.172098408]`, `⟨tr Z z⟩∈[0.38900449,0.38900467]` (= ½⟨tr X_I X_I⟩).

Code: `matrix_master_field/bfss/relaxation_selection.py` (+ `test_relaxation_selection.py`, 4/4 pass).
Built on the existing engine (`lz_port.py` loop relations, `lz_pos2.build` compiled
`(const,linear,quad)` system + `Mblocks`/`Nblocks(g)` positivity). Data:
`results/selection_branches.csv`, `results/relaxation_flatness.csv`. **Double precision (SCS)** —
resolves the *trend*, not the 8-digit island (that needs SDPA-GMP + the O(D) irrep reduction; deferred).

## Part C — selection (the crux; Q2 analog) — `selection_branches.csv`

Strip positivity; characterize the exact **factorized** loop variety (double-traces → products of
single-trace moments), then test positivity pointwise.

| level | nV | #loop eqs | **variety dim** = nV−rank(J) | positive-E extent (min/max SQP) | width | point/set |
|---|---|---|---|---|---|---|
| 4 | 14 | 48 | **3** | [0.86603, 9.056] | 8.19 | **SET** |
| 6 | 94 | 326 | **8** | [0.95578, 2.970] | 2.01 | **SET** |
| 8 | 614 | 2118 | **22** | [1.16967, —¹] | —¹ | **SET** |

¹ L8 max-E did not converge in double precision (positivity is weak on the upper side; honest limit).

**Findings.**
1. **The variety is positive-dimensional; dim = 3, 8, 22 at L=4,6,8 — exactly Lin–Zheng Table I.**
   This integer match (their own free-parameter count, reproduced here as nV−rank of the *factorized*
   Jacobian) validates the assembly and quantifies "how many directions positivity must fix." It grows
   with ℓ.
2. **Positivity does NOT collapse the variety to a point at finite truncation — it leaves a positive
   SET** (a bracket), width 8.19 → 2.01 → (shrinking). Confirms the Vehicle-A underdetermination
   finding quantitatively.
3. **The set tightens toward the island as ℓ grows.** The lower edge (a rigorous lower bound) rises
   **0.86603 → 0.95578 → 1.16967**, converging up to the island's lower edge 1.172098376 (gap ~0.2% at
   L8). L4's 0.86603 = √3/2 = D·M·√3/4, the analytic positivity bound (cross-check ✓). The island lies
   inside the positive set at every level.

**⇒ Positivity is necessary but not sufficient at finite truncation. The disambiguating selector is
level-growth** (the branch/edge that survives and converges as ℓ→∞) — equivalently continuation from a
solvable limit. This is the "realistic case," and it is the concrete criterion to port to the lattice.

## Part A — what the relaxation costs — `relaxation_flatness.csv`

| level | nV | #factorization products \|Q\| | **moment-matrix rank** r_ℓ(1e-6) | factorization slack S_ℓ | width w_ℓ |
|---|---|---|---|---|---|
| 4 | 14 | **0** | 2 | 0 (nothing to relax) | 8.19 |
| 6 | 94 | **0** | 12 | 0 (nothing to relax) | 2.01 |
| 8 | 614 | **6** | 38 | 6.96¹ | —¹ |

¹ double-precision-limited (min-E vertex at L8 is SCS-inaccurate; slack magnitude not trustworthy).

**Findings.**
1. **In the single-trace-moment formulation, the QM loop equations are LINEAR up to L6** (every term
   is a single trace; the CCR [X,P]=i·1 keeps the cyclicity contractions single-trace). **Genuine
   double-trace factorization products Q=xxᵀ first appear at L8** (6 single-trace variables enter
   products). So below L8 **there is no Q=xxᵀ constraint to relax** — the *only* relaxation is the
   positivity-hierarchy truncation. (Whether factorization is even "a relaxed constraint" is thus
   basis-/level-dependent — a caveat on the Q_A framing.)
2. **The moment-matrix rank does NOT stabilize: r_ℓ = 2 → 12 → 38** through L=4,6,8. No flat extension
   (r_ℓ ≠ r_{ℓ−1}) at reachable levels ⇒ the truncated moment problem is **not yet flat** ⇒ the
   representing state is not uniquely pinned by flatness — the SDP-dual reason Part C's positive region
   is still a *set*. (L4's min-E vertex is itself rank-2 with a clean 10-order gap — locally flat *at
   that vertex*, but the hierarchy as a whole is not.)

**⇒ At reachable levels the relaxation gap is NOT flatness-controlled** (rank still climbing);
"relax and hope the island is tight" is *not* asymptotically free here. The load-bearing route is the
non-relaxation lower bound (Part C), which tightens 0.866→0.956→1.170 with exact factorization +
positivity. Confirming flatness would need L≥10 (O(D) irrep reduction) at arbitrary precision.

## Answers

- **Q_A (can factorization be handled without relaxing; what does Shor cost?)** For this QM,
  factorization is not the operative relaxation below L8 — the loop equations are linear in single-trace
  moments; the operative relaxation is positivity truncation, whose gap is **not** flatness-controlled at
  reachable levels (rank 2→12→38). So the relaxation is *not* asymptotically free at reach; the exact
  non-relaxation lower bound is what converges to the answer.
- **Q_B (selection).** Positivity leaves a positive **set** (dim 3,8,22) that shrinks toward the island;
  it does not select a unique point at finite truncation. The extra selector is **level-growth /
  continuation** — measured here, ready to port. (Convergence half — Part D, Cuntz decay — folds into
  Vehicle B, deferred.)

## Transfer to the QCD₂ program (the Q2 blocker)

The lattice null-space question — "does positivity collapse the loop-equation null space to the
physical point?" — answers **NO at finite truncation** in this controlled setting: you get a
positive-dimensional variety and a shrinking positive *bracket*, not a point. The master-field program
is nonetheless well-posed: the physical point is the limit of the bracket's converging edge, selected by
**continuation from the solvable (strong-coupling / Gross–Witten) limit** and level-growth — not by
positivity at any fixed truncation. Concretely for QCD₂: expect a positive-dimensional lattice loop
variety, impose positivity for a shrinking bracket, and continue from GW to track the physical branch
(exactly the operator-field + continuation vehicle validated on KZ).

## Deferred (out of the minimal A+C path)

- 8-digit island / flat-extension certificate: needs SDPA-GMP (or mpmath) + the O(D)-irrep singlet
  reduction for L≥10. Both unbuilt.
- Part B1 Hartree self-consistent iteration; Part D Cuntz–Fock decay (→ Vehicle B).
