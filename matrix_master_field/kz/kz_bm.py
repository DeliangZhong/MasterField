"""Burer-Monteiro (direct rank-1 factor) on the KZ two-matrix model, moment space.

The rank-1 SDP variable is the product matrix G = m m^T; in moment space that factor
IS the moment vector m.  So BM here = minimize the exact-factorization feasibility
    F(m) = ||loop_residuals(m)||^2  +  pen * ||neg_eig(Omega(m))||^2
directly over m (smooth non-convex), with a good local optimizer.

Design-critical test: COLD multistart (does a strong optimizer find the physical
master field without continuation?) vs CONTINUATION (warm-start homotopy).

GT: g=1,h=1 -> m[A^2] in [0.4204,0.4224];  g=0.5,h=1 -> [0.4803,0.4842].
"""
import sys, time
import numpy as np
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from matrix_master_field.twomatrix_alm import build_structure, _make_pieces, residual_and_omega, free_moment
from matrix_master_field.bootstrap_sdp import _two_matrix_canon as canon


def make_F(struct, pen=1e3):
    omega_j, eqs = _make_pieces(struct)
    def F(m_free):
        res, Om = residual_and_omega(m_free, omega_j, eqs)
        neg = jnp.minimum(jnp.linalg.eigvalsh(Om), 0.0)
        return jnp.sum(res*res) + pen*jnp.sum(neg*neg)
    return jax.jit(jax.value_and_grad(F)), omega_j, eqs


def bm_min(struct, m0, pen=1e3, maxiter=3000):
    vg, omega_j, eqs = make_F(struct, pen)
    def fg(x):
        v, g = vg(jnp.asarray(x)); return float(v), np.asarray(g, float)
    r = minimize(fg, m0, jac=True, method="L-BFGS-B",
                 options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14})
    res, Om = residual_and_omega(jnp.asarray(r.x), omega_j, eqs)
    return r.x, float(np.linalg.norm(np.asarray(res))), float(np.min(np.linalg.eigvalsh(np.asarray(Om))))


def bm_cold(g, h, L, restarts=10, target=(0,0), seed=0):
    s = build_structure(L, 1.0, g, h)
    cl, cidx = s["canon_list"], s["cidx"]; tidx = cidx[canon(target)]
    m0 = np.array([float(free_moment(c)) for c in cl[1:]])
    rng = np.random.default_rng(seed)
    outs = []
    for t in range(restarts):
        init = m0 if t == 0 else m0 * rng.uniform(0.3, 1.7, size=m0.shape)
        m, r, mineig = bm_min(s, init)
        mval = float(np.concatenate([[1.0], m])[tidx])
        outs.append((r, mineig, mval))
    # keep the ones that actually satisfy the eqs and are ~PSD
    feas = [o for o in outs if o[0] < 1e-4 and o[1] > -1e-3]
    mvals = sorted(set(round(o[2], 4) for o in feas))
    return outs, feas, mvals


def bm_continuation(g, h, L, n_steps=15, target=(0,0)):
    s_t = build_structure(L, 1.0, g, h); cl, cidx = s_t["canon_list"], s_t["cidx"]
    tidx = cidx[canon(target)]
    m = np.array([float(free_moment(c)) for c in cl[1:]])
    for t in np.linspace(0, 1, n_steps+1):
        s = build_structure(L, 1.0, t*g, t*h)
        m, r, mineig = bm_min(s, m)
    return float(np.concatenate([[1.0], m])[tidx]), r, mineig


if __name__ == "__main__":
    print("BURER-MONTEIRO (direct rank-1 factor), KZ model")
    print("GT: g=1,h=1 -> [0.4204,0.4224];  g=0.5,h=1 -> [0.4803,0.4842]\n")
    for (g, h, isl) in [(1.0,1.0,"[0.4204,0.4224]"), (0.5,1.0,"[0.4803,0.4842]")]:
        print(f"=== g={g} h={h}  island {isl} ===")
        for L in (6, 8):
            t0=time.time()
            outs, feas, mvals = bm_cold(g, h, L, restarts=10)
            print(f"  COLD L={L}: {len(feas)}/{len(outs)} restarts feasible; "
                  f"distinct m[A^2] values found = {mvals}   ({time.time()-t0:.0f}s)")
        for L in (6, 8):
            t0=time.time()
            m, r, mineig = bm_continuation(g, h, L)
            print(f"  CONT L={L}: m[A^2]={m:.5f}  (res {r:.1e}, min eig {mineig:+.1e})   ({time.time()-t0:.0f}s)")
        print()
