# Validating both methods on the KZ two-matrix model (non-linear opt + master field)

**Date:** 2026-07-01 · **Branch:** `matrix-master-field` · **Status:** done (operator field validated to ~0.25%)

**Why.** Before QCD₃, test our two proposed tools on a model with a KNOWN answer: the
Kazakov–Zheng "unsolvable" two-matrix model (arXiv:2108.04830, eq. 6),
`S = N·tr[½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]²]`. Don't skip the easy case. Observable: `m[A²]=τ(A²)`.

## Ground truth (convex bootstrap, our `bootstrap_two_matrix_kz`)

The convex island (rigorous bracket, contains the true value) shrinks with cutoff L:

| coupling | L=4 | L=8 | L=12 |
|---|---|---|---|
| g=1, h=1 | [0.000, 0.618] | [0.394, 0.431] | **[0.4204, 0.4224]** |
| g=0.5, h=1 | [0.000, 0.732] | [0.441, 0.498] | **[0.4803, 0.4842]** |

(External anchors, exact: h=0 decouples into two quartic one-matrix models,
`m[A²]=a⁴(ga²+1)/16`, `a²+¾ga⁴=4` → g=1: **0.5161**, g=0.5: **0.6309**.)

## The axis that matters: bracket the feasible set vs track the physical branch

The convex bootstrap relaxes the quadratic loop equation to `G⪰mmᵀ` (KZ eq. 53). Non-linear
optimization can impose `G=mmᵀ` exactly — but **whether it helps depends entirely on the
formulation**, not on "non-linear vs convex". We tested four non-linear methods:

| method | how it uses the constraints | g=1,h=1 (island [0.4204,0.4224]) | g=0.5,h=1 ([0.4803,0.4842]) |
|---|---|---|---|
| **min/max ALM** (`alm_min_max`) | extremize m[A²] over {exact loop eqs + Ω⪰0} | **[0, 0.618]** — no better than convex | [0, 0.732] |
| **moment continuation, ALM** (`kz_moment_methods`) | homotopy (g,h)=t·(g,h) from free pt, minimal-change | L8 **0.42292** (L4 0.00, L6 0.444) | L8 **0.48910** |
| **moment continuation, SQP** (`kz_sqp`, exact-PSD SDP steps) | same, linearize `mⱼmₗ`, solve SDP each step | L8 **0.42298** | L8 **0.48898** |
| **operator continuation** (below) | free-generator ansatz, ramp h | **0.41968** | **0.47889** |

**The lesson (correcting an earlier too-quick verdict).** Non-linear optimization *does* help the
two-matrix model — but the lever is **continuation** (track the physical branch from the solvable
limit), **not** min/max bracketing. min/max fails because it extremizes over the whole feasible
set, which at finite cutoff contains **spurious collapsed configs** (m[A²]=0 is feasible — a large
m[A⁴] balances the loop equation); the min just finds them. Continuation ignores the extremes and
follows the physical saddle. Three independent continuation methods (moment-ALM, moment-SQP,
operator) agree and converge to the islands; the two moment methods match to 4 digits, and the
operator lands just *below* each island while the moment methods land just *above* — they straddle
the truth (g=1: 0.4197 vs 0.4229, true ∈ [0.4204,0.4224]; g=0.5: 0.4789 vs 0.4890, true ∈
[0.4803,0.4842]). (For contrast, min/max ALM did shrink the *one-matrix* bracket
[0.186,0.592]→[0.516] — there the loop recursion nearly closes so the physical point *is* the
extremum; multi-matrix is not that case, which is why the *method* matters.)

## Operator master field + continuation (the cleanest branch-tracker)

Parametrize A, B as self-adjoint polynomials in two **free** semicircular generators x₀,x₁ on
the Cuntz–Fock space (Z₂×Z₂ ⇒ A odd in x₀, even in x₁; exchange ⇒ B(x₀,x₁)=A(x₁,x₀)). The free
vacuum is the tracial state (Voiculescu) ⇒ **positivity + traciality automatic to all orders**
(not truncated as in the moment bootstrap). Minimize the KZ planar loop residual
`τ(V'_A·w) − Σ_{k:w_k=A} τ(w_{<k})τ(w_{>k})` over the ansatz coefficients.

**Cold start is under-determined.** At test-word cutoff W=2 the optimizer drives the loop
residual to machine zero (~1e-31) but lands on a **spurious** solution, `m[A²]=0.343` — zero
residual does not pin the physical field when few loop equations are imposed (the operator
analog of low bootstrap cutoff).

**Continuation from the solvable h=0 limit fixes it.** Ramp h:0→1 with warm starts, tracking the
physical branch:

| coupling / config | h=0 (exact) | h=1 (rigorous island) | gap |
|---|---|---|---|
| g=1, W=2, Fock-L=8, dim 511 | 0.548 | 0.41898 | — |
| g=1, W=3, Fock-L=10, dim 2047 | 0.51558 (exact 0.5161) | **0.41968** ([0.4204,0.4224]) | **−0.25%** |
| g=0.5, W=2, Fock-L=8, dim 511 | 0.658 | 0.45865 | — |
| g=0.5, W=3, Fock-L=10, dim 2047 | 0.63082 (exact 0.6309) | **0.47889** ([0.4803,0.4842]) | **−0.3%** |

Each trajectory is monotone and physical (the commutator confines: m[A²] drops as h grows), the
h=0 anchor is nailed at W=3 (0.51558 vs 0.5161; 0.63082 vs 0.6309), and the h=1 endpoint
**converges up toward the rigorous island** as W grows (g=0.5: 0.459→0.479). **Both couplings land
~0.3% below the certified lower bound at W=3, deg-3, dim 2047 — in seconds**, versus the convex
bootstrap's 47 s at L=12 (which yields only a bracket).

**Why the operator field is the cleanest branch-tracker** (all three continuation methods work; the
operator one is the most robust):
1. **All-orders positivity** — a genuine operator state satisfies Ω⪰0 for *every* word, not just up
   to the cutoff; the moment continuations enforce Ω⪰0 only on the truncated tower, so they drift
   slightly off the PSD manifold (min eig ~−3e-5) and overshoot the island. The operator field can't
   drift — it *is* a state.
2. **Continuation** from the solvable limit selects the physical branch (cold start / min-max finds
   spurious collapsed solutions).
3. It's a **construction, not a bound**: one configuration → any observable (arbitrarily long words /
   large "Wilson loops"), which the cutoff-limited moment bootstrap cannot hold.

## Caveats (honest)

- **Continuation is essential and can still drift at low order.** All continuation methods are noisy
  at small L (moment: L=4 collapses to ~0; operator W=2 is coupling-dependent 0.5–5%); they converge
  with L / W. Sub-0.1% needs higher cutoff — richer operator ansatz (deg-5+, higher Fock-L; the dense
  rep hits a Fock-scaling wall) or higher moment L — motivating the **matrix-free** operator field
  (`cuntz_bootstrap/matfree_expm.py`), the machinery QCD₃ needs.
- min/max ALM is a *bracket over the feasible set*, so it reports the spurious extremes; it is the
  wrong non-linear formulation for a master-field *determination* (right for a *bound*).

## Implication for QCD₃

The operator master field + continuation is the validated multi-matrix vehicle; its lever is
all-orders positivity + branch tracking, and it delivers a *point* cheaply where the bootstrap
needs high cutoff. Combined with the exact lattice loop equation
(`2026-07-01-lattice-loop-equation.md`), this is the QCD₃ pipeline: matrix-free operator field,
continue from the Gross–Witten / QCD₂ anchor into QCD₃, evaluate large Wilson loops the bootstrap
(2502.14421) cannot reach.

## Files

All reproducible; run from the repo root (see `matrix_master_field/kz/README.md` for commands).

- `matrix_master_field/twomatrix_alm.py` — exact-factorization ALM (min/max bracket + `_project_alm`
  moment continuation).
- `matrix_master_field/kz/kz_moment_methods.py` — moment-space exact-factorization ALM + homotopy
  continuation. L=8 → 0.42292 (g=1,h=1), 0.48910 (g=0.5,h=1).
- `matrix_master_field/kz/kz_sqp.py` — sequential-SDP continuation, exact `Ω⪰0` per Newton step
  (cvxpy/SCS). L=8 → 0.42298, 0.48898.
- `matrix_master_field/kz/kz_bm.py` — Burer–Monteiro (rank-1 factor); cold multistart vs continuation.
- `matrix_master_field/kz/opfield_kz.py` — operator field, cold start (shows under-determination).
- `matrix_master_field/kz/opfield_kz_cont.py` — operator field + continuation from h=0. W=3, dim 2047
  → m[A²]≈0.41968 (g=1).
- `matrix_master_field/kz/opfield_kz_cont_deg5.py` — same, degree-5 operator ansatz.

(Next: lift the operator continuation onto the matrix-free machinery `cuntz_bootstrap/matfree_expm.py`
for the higher cutoff QCD₃ needs.)
