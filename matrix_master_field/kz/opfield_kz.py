"""Operator master field for the Kazakov-Zheng two-matrix model (fast).

A,B self-adjoint on the free (Cuntz-Fock) space, polynomials in free semicirculars
x0,x1. Vacuum state = tracial (Voiculescu) => positivity+traciality AUTOMATIC.
Z2xZ2 + exchange:  A odd in x0, even in x1;  B(x0,x1)=A(x1,x0).
Minimize KZ planar loop residual.  Ground truth m[A^2] ~ 0.421 at g=h=1.

Speed: precompute monomial matrices once; tau(...) via mat-VEC chains to |Omega>.
"""
import sys, time, itertools
import numpy as np
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from matrix_master_field.cuntz_fock import CuntzFockSpace


def sa_monomials(x0, x1, degree):
    """Self-adjoint monomials ODD in x0, EVEN in x1, up to `degree`."""
    def sym(*mats):
        P = mats[0]
        for m in mats[1:]:
            P = P @ m
        Pr = mats[-1]
        for m in reversed(mats[:-1]):
            Pr = Pr @ m
        return 0.5 * (P + Pr)
    mons = [x0]
    if degree >= 3:
        mons += [x0@x0@x0, sym(x0, x1, x1), x1@x0@x1]
    if degree >= 5:
        mons += [x0@x0@x0@x0@x0, sym(x0,x0,x0,x1,x1), sym(x0,x1,x1,x1,x1),
                 x1@x1@x0@x1@x1, sym(x0,x0,x1,x0,x1), x1@x0@x0@x0@x1]
    return mons


def make_loss(fock, g, h, W, degree):
    x0 = fock.x(0).astype(np.float64); x1 = fock.x(1).astype(np.float64)
    MA = jnp.asarray(np.stack(sa_monomials(x0, x1, degree)))   # (P,dim,dim)
    MB = jnp.asarray(np.stack(sa_monomials(x1, x0, degree)))
    nP = MA.shape[0]
    vac = jnp.asarray(fock.vacuum_state())
    words = [()]
    for L in range(1, W + 1):
        words += [tuple(c) for c in itertools.product((0, 1), repeat=L)]

    def build(c):
        A = jnp.tensordot(c, MA, axes=(0, 0)); B = jnp.tensordot(c, MB, axes=(0, 0))
        A2 = A@A; B2 = B@B
        VpA = A + g*(A2@A) + h*(A@B2 + B2@A - 2.0*(B@A@B))
        VpB = B + g*(B2@B) + h*(B@A2 + A2@B - 2.0*(A@B@A))
        return A, B, VpA, VpB

    def tau_chain(ops, vac):          # <Om| ops[0] ops[1] ... |Om>  via matvecs
        v = vac
        for op in reversed(ops):
            v = op @ v
        return v[0]

    def residuals(c):
        A, B, VpA, VpB = build(c)
        opmap = (A, B)
        res = []
        for w in words:
            wops = [opmap[k] for k in w]
            for a, Vp in ((0, VpA), (1, VpB)):
                lhs = tau_chain([Vp] + wops, vac)
                rhs = 0.0
                for k in range(len(w)):
                    if w[k] == a:
                        lhs_l = tau_chain([opmap[j] for j in w[:k]], vac)
                        lhs_r = tau_chain([opmap[j] for j in w[k+1:]], vac)
                        rhs = rhs + lhs_l * lhs_r
                res.append(lhs - rhs)
        return jnp.stack(res)

    def loss(c):
        r = residuals(c)
        return jnp.sum(r*r)

    def mA2(c):
        A, _, _, _ = build(c)
        return (A @ (A @ vac))[0]

    return jax.jit(jax.value_and_grad(loss)), jax.jit(loss), jax.jit(mA2), nP


def solve(fock, g, h, W, degree, restarts=3, seed=0, maxiter=1500):
    vg, loss, mA2, nP = make_loss(fock, g, h, W, degree)
    rng = np.random.default_rng(seed)
    best = None
    for t in range(restarts):
        c0 = np.zeros(nP); c0[0] = 1.0
        if t: c0 = c0 + rng.normal(0, 0.25, nP)
        def fg(x):
            v, gr = vg(jnp.asarray(x)); return float(v), np.asarray(gr, float)
        r = minimize(fg, c0, jac=True, method="L-BFGS-B",
                     options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-13})
        Lv = float(loss(jnp.asarray(r.x))); m = float(mA2(jnp.asarray(r.x)))
        if best is None or Lv < best[0]: best = (Lv, m, r.x)
    return best


def eval_at(fock, g, h, W, degree, c):
    _, loss, mA2, _ = make_loss(fock, g, h, W, degree)
    return float(loss(jnp.asarray(c))), float(mA2(jnp.asarray(c)))


if __name__ == "__main__":
    g, h = 1.0, 1.0; GT = 0.421
    print(f"Operator master field, KZ g={g} h={h}.  Ground truth m[A^2] in [0.4204,0.4224]\n", flush=True)
    print(f"{'FockL':>5} {'W':>2} {'deg':>3} {'dim':>5} {'loss':>11} {'m[A^2]':>9} {'err':>7} {'t(s)':>6}", flush=True)
    results = {}
    for FL, W, deg in [(8,2,3),(10,2,3),(10,3,3),(10,3,5),(10,4,5)]:
        fock = CuntzFockSpace(2, FL); t = time.time()
        Lv, m, c = solve(fock, g, h, W, deg)
        results[(W,deg)] = c
        print(f"{FL:>5} {W:>2} {deg:>3} {fock.dim:>5} {Lv:>11.3e} {m:>9.5f} {abs(m-GT):>7.4f} {time.time()-t:>6.1f}", flush=True)
    # Fock-truncation convergence: re-evaluate the best (W=3,deg=5) coeffs at higher L
    if (3,5) in results:
        c = results[(3,5)]
        print("\nFock-truncation check (fixed coeffs from W=3,deg=5):", flush=True)
        for FL in (10, 12, 14):
            fock = CuntzFockSpace(2, FL)
            Lv, m = eval_at(fock, g, h, 3, 5, c)
            print(f"  Fock-L={FL} dim={fock.dim}: loss={Lv:.3e}  m[A^2]={m:.6f}", flush=True)
