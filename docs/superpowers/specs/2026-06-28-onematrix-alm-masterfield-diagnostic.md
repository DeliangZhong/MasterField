# One-matrix ALM master-field diagnostic (design)

**Date:** 2026-06-28 · **Status:** design (approved in conversation; spec for review) · **Branch:** `matrix-master-field`

## Goal

Build the augmented-Lagrangian (ALM) master-field machinery on the **exactly-solvable one-matrix
model**, and use it to answer the program's central question:

> Is the precision of the master-field approach (vs. the conventional **convex SDP** bootstrap) due to
> **(H1)** the large-$N$ master field being *special* (a unique saddle, an $N=\infty$ phenomenon), or
> **(H2)** just the **non-linear factorization constraint** being the key to precision — which would
> then work at finite $N$ too, making "master field" incidental framing?

The one-matrix model is the right arena because it is **exactly solvable at every $N$** (orthogonal
polynomials at finite $N$; closed density at $N=\infty$), so we have ground truth everywhere and can
*measure* H1 vs. H2 instead of arguing it. This also stands up the ALM engine reused everywhere above
(two-matrix, then QCD).

**Honest caveat (in scope from the start):** the one-matrix convex bootstrap is already nearly tight
(one free parameter $m_2$ after the loop-equation recursion), so the precision *gap* here is small.
One-matrix is therefore the **controlled diagnostic** — it answers "do the methods recover the exact
answer, do A and B agree, is there *any* non-linear-vs-convex gap, and is it $N$-dependent." The
*dramatic* version (a wide convex bound collapsing to a point) belongs to a **loose-convex multi-matrix
model**, which is the explicit next project, not this one.

## Conventions (pinned — from `matrix_master_field/CONVENTIONS.md`; do not re-guess)

- Hermitian $N\times N$ matrix $M$; action $S = N\,\mathrm{tr}\,V(M)$; partition function
  $Z=\int dM\,e^{-N\,\mathrm{tr}\,V(M)}$.
- Potential $V(M)=\tfrac12 M^2 + \tfrac{g}{4}M^4$, so $V'(M)=M+gM^3$, coupling $g\ge 0$.
- Normalized moments $m_k = \tfrac1N\langle \mathrm{tr}\,M^k\rangle$; $m_0=1$ (hard constraint, never a
  variable); odd moments vanish by $M\to-M$ parity.
- **Large-$N$ loop equation** (factorized Schwinger–Dyson):
  $$ m_{n+1} + g\,m_{n+3} = \sum_{k=0}^{n-1} m_k\,m_{n-1-k}, \qquad n\ge 0. $$
  ($n=0$: RHS empty $\Rightarrow m_1+g m_3=0$, consistent with parity.)
- **Finite-$N$ loop equation** (exact, un-factorized): same LHS, RHS
  $\sum_{k=0}^{n-1}\tfrac1{N^2}\langle\mathrm{tr}\,M^k\,\mathrm{tr}\,M^{n-1-k}\rangle$ — the double-trace
  moments do **not** factorize; the connected part is $O(1/N^2)$.
- $g=0$ anchor: master field is the semicircle; $m_{2k}=C_k$ (Catalan: $1,1,2,5,14,42,\dots$).
- Quartic density (CONVENTIONS.md): $\rho(x)=\tfrac{1}{2\pi}(g x^2+1+\tfrac{g a^2}{2})\sqrt{a^2-x^2}$,
  with $a^2$ fixed by $\tfrac{3}{4}g a^4 + a^2 = 4$ (normalization). Moments by integration.

## Architecture — five computations + one ground truth

All share the same word/moment bookkeeping and the same exact-answer spine.

0. **Exact ground truth** (`exact_onematrix`): finite-$N$ moments via the orthogonal-polynomial
   (three-term-recurrence) solution of $\int dM\,e^{-N\mathrm{tr}V}$; $N=\infty$ moments via the
   density above (and Catalan at $g=0$). This is the validation spine for every method, at every $N$.

1. **Convex baseline (large $N$)** — the conventional bootstrap: moments as variables, Hankel matrix
   $H_{ij}=m_{i+j}\succeq0$ (an LMI, convex), loop equations with the quadratic RHS handled the convex
   way (products treated as the relaxation / eliminated via the 1-parameter recursion). Reuse/extend
   `bootstrap_sdp.py` + `one_matrix.py`. Output: the bound on $m_2$ (and hence all moments).

2. **B(∞) — moment + non-linear ALM**: variables = moments $\{m_k\}_{k\le 2L}$; PSD enforced by the
   Cholesky parametrization $H=LL^\top$ (variables are the entries of $L$); the **factorized** loop
   equations imposed as non-linear equality constraints $h(\cdot)=0$ via the augmented Lagrangian
   $\mathcal L = \text{Obj} - \sum_i \lambda_i h_i + \tfrac{\mu}{2}\sum_i h_i^2$ (inner loop: JAX
   autograd gradient descent over $L$; outer loop: $\lambda_i\!\leftarrow\!\lambda_i-\mu h_i$, increase
   $\mu$ if $\|h\|$ stalls). Output: the unique moment vector.

3. **A — operator field + ALM (large $N$)**: parametrize a Hermitian master operator $\hat X$ on a
   truncated Cuntz/bosonic Fock space; moments $m_k=\langle\Omega|\hat X^k|\Omega\rangle$. PSD and
   factorization are **automatic** (genuine state — no Cholesky), so ALM enforces only the loop
   equations, acting on $\hat X$'s generator parameters. Output: the master-field operator + its
   moments. (Reuse `matrix_master_field/cuntz_fock.py` / `free_fisher.py` operator machinery.)

4. **B(finite $N$) — the control**: same as (2) but with the **finite-$N$** loop equations (carry the
   double-trace moments as auxiliary variables, or impose the exact finite-$N$ Schwinger–Dyson
   relations) and the finite-$N$ exact answer as ground truth. A cannot run here (it is $N=\infty$ by
   construction); B can. This is the rung that tests whether the non-linear-constraint precision is an
   $N=\infty$ effect or a general one.

5. **Readout (the deliverable)**: a table over $\{$Gaussian, quartic at a couple of $g\}$ comparing
   $m_k$ from A, B(∞), B(finite-$N$), and the convex baseline against exact — with the **H1-vs-H2
   verdict** spelled out from the comparisons in the obligations below.

## Validation obligations

| # | Claim | Check |
|---|-------|-------|
| V1 | $g=0$ anchor | every method returns Catalan $m_{2k}=C_k$ to machine precision |
| V2 | quartic correctness | A and B(∞) recover the exact large-$N$ quartic moments (vs `exact_onematrix`) to a stated tolerance |
| V3 | loop-equation residual | ALM constraint residual $\|h\|\to 0$ (report final value) for A and B |
| V4 | **H1/H2 — A vs B(∞)** | report whether A is more precise/robust than B(∞), or A$\approx$B(∞) (⇒ the constraint, not the master-field structure, carries the precision; A's unique edge is large/high-$k$ moments) |
| V5 | **H1/H2 — finite vs large $N$** | does the non-linear-vs-convex precision gain at finite $N$ match the large-$N$ gain? (general ⇒ H2; large-$N$-only ⇒ H1) |
| V6 | **artifact check (M5c scar)** | every ALM solution is **unique across $\ge 5$ random restarts** and **stable when re-validated at a larger basis** ($L\!+\!2$ moments / larger Fock) than it was optimized on — non-negotiable |

Verification medium: Python + `pytest` (numpy/scipy/JAX/cvxpy), cross-checked against `exact_onematrix`.

## Modules

- **`matrix_master_field/onematrix_alm.py` (new):** the ALM engine (shared outer/inner loop), method A
  (operator field) and method B (moment, ∞ and finite-$N$), the exact ground truth, and the experiment
  driver + readout.
- **Reuse:** `bootstrap_sdp.py` / `one_matrix.py` (convex baseline + existing one-matrix moment code);
  `cuntz_fock.py` / `free_fisher.py` (operator machinery for A).
- **Tests:** `matrix_master_field/tests/test_onematrix_alm.py`.

## Sequencing (for the plan)

Core first (answers the primary question): **0 → 1 → 2 → 3 → 5(core)**. Then the control: **4 → 5(full)**.

## Risks / out of scope

- **R1 — small gap (expected):** one-matrix convex is near-tight, so V4/V5 gaps may be small; that is
  itself the finding ("one-matrix doesn't separate H1/H2 strongly — escalate to loose-convex"). Not a
  failure.
- **R2 — non-convex local minima:** multi-start + the V6 uniqueness check are the guard.
- **R3 — finite-$N$ control complexity:** the double-trace bookkeeping is the heaviest piece; it is
  deliberately sequenced last so the core deliverable does not depend on it.
- **Out of scope:** the loose-convex multi-matrix headline model; QCD; any finite-$N$ master field
  (the master field is $N=\infty$ only — finite-$N$ appears only as the moment-bootstrap control).
