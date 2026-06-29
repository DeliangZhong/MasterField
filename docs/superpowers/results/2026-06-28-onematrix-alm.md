# One-matrix ALM master field — exact factorization collapses the convex bound (result)

**Date:** 2026-06-28 · **Status:** done + tested (7/7) · **Branch:** `matrix-master-field`
**Spec:** `docs/superpowers/specs/2026-06-28-onematrix-alm-masterfield-diagnostic.md`

**One-line.** On the exactly-solvable one-matrix model $V=\tfrac12M^2+\tfrac{g}{4}M^4$, imposing the
factorized loop equations **exactly** ($Q=mm^\top$) by an augmented-Lagrangian method collapses the
convex bootstrap's wide/unbounded bracket onto the unique answer (~$10^{-4}$ wide, containing the exact
value) — the program's thesis, validated on solvable ground. Two encodings (moments+ALM and an
operator/Jacobi field) give the *same* number, so the precision is the **non-linear constraint**, not
the parametrization.

## Setup

The conventional bootstrap (`bootstrap_sdp.bootstrap_one_matrix`) encodes the factorized loop-equation
RHS $\sum_j m_j m_{n-1-j}$ by a product matrix $Q$ with $Q_{0k}=m_k$, $Q\succeq0$ — i.e. the convex
**relaxation** $Q\succeq mm^\top$ of the exact rank-1 $Q=mm^\top$. The relaxation is the entire source
of looseness. `onematrix_alm.py` imposes $Q=mm^\top$ exactly (the true quadratic loop equations) by
ALM — the non-convex problem the SDP cannot represent — selected by Hankel positivity.

## Result — same truncation ($K=6$), $\lambda=1$ quartic

| moment | exact | convex $[\text{lb},\text{ub}]$ (width) | **exact-factorization ALM** (width) |
|---|---|---|---|
| $m_2$ | 0.5162 | $[0.186, 0.592]$ (0.41) | $[0.5157, 0.5162]$ (**5e-4**) |
| $m_4$ | 0.4838 | $[0.405, 0.863]$ (0.46) | $[0.4838, 0.4843]$ (**5e-4**) |
| $m_6$ | 0.5485 | **unbounded** above | $[0.5472, 0.5487]$ (**1.5e-3**) |

(g=0.5 identical story; g=0 returns Catalan $1,2,5,14,42$ exactly, loop residual 6e-9.) For the
**Gaussian** the convex relaxation is already loose — $m_4\in[2.00,4.05]$, $m_6\in[4.08,47.5]$,
$m_8,m_{10}$ upper-unbounded — even though the answer is the unique Catalan sequence. Imposing exact
factorization pins it. (Corrects the spec's "one-matrix convex is nearly tight": the *honest* $Q$-relaxation is loose; the textbook tight one-matrix bootstrap is tight only because it secretly keeps
factorization exact via the recursion.)

## H1 vs H2 — it is the constraint

- **Method A** (operator/Jacobi master field, `method_a_jacobi`): parametrize a self-adjoint tridiagonal
  $J$ (off-diagonals $b_n$); Hankel⪰0 and factorization are then *automatic* (no penalty). Fitting the
  loop equations gives $b_n=1$ (semicircle, Catalan exact) at $g=0$, and matches the exact quartic
  moments to rel.err $\sim10^{-4}$ at $g=0.5,1$ ($b_n\to0.74,\,0.66$).
- **Method B** (moments + ALM with a Hankel-eigenvalue penalty): the brackets above.

A and B land on the **same** answer because they impose the **same** factorized loop equations. The
"master field" operator is just a convenient (positivity-automatic) encoding of the same
constraint-satisfying solution — not a source of extra power. **Verdict on the one-matrix model: H2**
(the non-linear constraint is the key, not the parametrization).

## Honest caveats

- The ~$10^{-4}$ ALM bracket is a genuine truncated-PSD min/max (it shrinks with $K$), not a certified
  zero; the soft eigenvalue penalty allows a hair of slack. As a *determination* it nails the answer.
- One-matrix is too easy to fully separate A and B — both are exact. The operator field's real edges
  (positivity-automatic, and evaluating **large loops** the moment set can't hold) only bite in the
  loose-convex **multi-matrix** case. That is the next testbed.
- Finite-$N$ control: **dropped** — the goal is strictly large $N$, where the constraint is non-linear
  (factorization) and we impose it directly. (At finite $N$ factorization does not hold and the loop
  equations are linear/convex; not our regime.)

## Next

Escalate to a **loose-convex multi-matrix model** (where the convex bootstrap is genuinely wide and the
moment set cannot be eliminated by a one-parameter recursion), then to the lattice gauge / QCD$_3$
operator master field (`cuntz_bootstrap/`). The ALM engine here (augmented Lagrangian on the loop
equations + positivity) is the reusable core.

## Files

- `matrix_master_field/onematrix_alm.py` — ALM solver (method B), Jacobi operator field (method A).
- `matrix_master_field/tests/test_onematrix_alm.py` — 7 tests (g=0 Catalan, g>0 bracket contains exact
  & ≪ convex, method A = exact).
- Reuses `one_matrix.py` (exact moments), `bootstrap_sdp.py` (convex baseline).
