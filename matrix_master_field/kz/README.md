# KZ two-matrix model — non-convex optimization + operator master field

Reproducible scripts behind `docs/superpowers/results/2026-07-01-two-matrix-validation.md`.
They test our two proposed QCD₃ tools on the Kazakov–Zheng "unsolvable" two-matrix model
(arXiv:2108.04830, eq. 6) — a model with a KNOWN convex-bootstrap answer — before scaling up:

    S = N·tr[ ½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]² ]

Observable: `m[A²] = τ(A²)`. Convex ground truth (our `bootstrap_two_matrix_kz`, L=12):
g=1,h=1 → `[0.4204, 0.4224]`; g=0.5,h=1 → `[0.4803, 0.4842]`.

Run from the repo root (each script discovers the root via `Path(__file__).parents[2]`):

```bash
uv run --no-project --with numpy --with scipy --with jax python matrix_master_field/kz/opfield_kz_cont.py
uv run --no-project --with numpy --with scipy --with jax python matrix_master_field/kz/kz_moment_methods.py
uv run --no-project --with numpy --with scipy --with jax --with cvxpy python matrix_master_field/kz/kz_sqp.py
```

## The lesson

Non-linear optimization *does* beat the convex bracket on this model — but the lever is
**continuation** (track the physical branch from the solvable h=0 limit), **not** min/max
bracketing over the feasible set (that set contains spurious collapsed configs at finite cutoff).
Three independent continuation methods agree and straddle the certified islands.

## Files

Moment space (variables = planar moments; deps: `matrix_master_field.twomatrix_alm`,
`.bootstrap_sdp`):
- `kz_moment_methods.py` — exact-factorization ALM + homotopy `(g,h)=t·(g,h)`, minimal-change
  projection. L=8 → **0.42292** (g=1,h=1), **0.48910** (g=0.5,h=1).
- `kz_sqp.py` — sequential-SDP: linearize `mⱼmₗ`, solve a convex SDP with *exact* `Ω⪰0` (cvxpy/SCS)
  each Newton step, then step the homotopy. L=8 → **0.42298**, **0.48898**.
- `kz_bm.py` — Burer–Monteiro (direct rank-1 factor = the moment vector). Design test: COLD
  multistart (finds spurious solutions) vs CONTINUATION (recovers the physical branch).

Operator master field (variables = coefficients of A,B as self-adjoint polynomials in two free
semicircular generators on the Cuntz–Fock space; positivity + traciality automatic to all orders;
dep: `matrix_master_field.cuntz_fock`):
- `opfield_kz.py` — cold-start solve at fixed (g,h); shows the cold start is under-determined
  (spurious `m[A²]=0.343` at test-word cutoff W=2).
- `opfield_kz_cont.py` — **continuation from h=0** (ramp h:0→1, warm starts). W=3, Fock-L=10,
  dim 2047 → **m[A²]≈0.41968** (g=1), ~0.3% below the certified lower bound, in seconds.
- `opfield_kz_cont_deg5.py` — same, degree-5 operator ansatz (richer branch, W=2, Fock-L=12).

The operator field is the cleanest branch-tracker (all-orders positivity; it *is* a state, so it
can't drift off the PSD manifold) and is a construction, not a bound: one configuration → any
observable, including long words the cutoff-limited moment bootstrap can't reach. That is the
property QCD₃ needs.
