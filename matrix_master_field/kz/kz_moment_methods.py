"""More non-linear optimization methods on the KZ two-matrix model (moment space).

Method 1 (done elsewhere): ALM min/max bracketing -> no improvement over convex.
Here:
  (A) MOMENT-SPACE CONTINUATION: exact factorization (G=mm^T) + homotopy from the
      free (g=h=0) point along (g,h)=t*(g_t,h_t), minimal-change projection each step.
      This is the direct moment-space analog of the operator-field continuation that
      hit ~0.3%.  Tests whether continuation (not the operator rep) is the key.
  (B) Newton polish: from the continuation endpoint, damped Newton on the loop
      equations restricted to the free moments (exact factorization), PSD-checked.

Ground truth (convex L=12): g=1,h=1 -> m[A^2] in [0.4204,0.4224];
                            g=0.5,h=1 -> [0.4803,0.4842].
"""
import sys, time
import numpy as np
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from matrix_master_field.twomatrix_alm import (
    build_structure, _make_pieces, residual_and_omega, free_moment,
)
from matrix_master_field.bootstrap_sdp import _two_matrix_canon as canon


def project_min_change(struct, m_prev_free, *, n_outer=60, max_inner=300, pen_psd=1e4):
    """argmin ||m - m_prev||^2  s.t. exact loop eqs = 0 (ALM) and Omega ⪰ 0 (eig penalty)."""
    omega_j, eqs = _make_pieces(struct)
    mp = jnp.asarray(m_prev_free)

    def lag(m_free, lam, rho):
        res, Om = residual_and_omega(m_free, omega_j, eqs)
        neg = jnp.minimum(jnp.linalg.eigvalsh(Om), 0.0)
        return (jnp.sum((m_free - mp) ** 2) - jnp.dot(lam, res)
                + 0.5 * rho * jnp.dot(res, res) + pen_psd * jnp.dot(neg, neg))

    vg = jax.jit(jax.value_and_grad(lag))
    m_free = np.asarray(m_prev_free, float)
    lam = np.zeros(len(eqs)); rho, prev = 1.0, np.inf
    for _ in range(n_outer):
        def fg(x):
            v, gr = vg(jnp.asarray(x), jnp.asarray(lam), jnp.asarray(rho))
            return float(v), np.asarray(gr, float)
        m_free = minimize(fg, m_free, jac=True, method="L-BFGS-B",
                          options={"maxiter": max_inner}).x
        res, Om = residual_and_omega(jnp.asarray(m_free), omega_j, eqs)
        res = np.asarray(res); nh = float(np.linalg.norm(res))
        lam = lam - rho * res
        if nh > 0.25 * prev:
            rho = min(rho * 3.0, 1e9)
        prev = nh
    res, Om = residual_and_omega(jnp.asarray(m_free), omega_j, eqs)
    mineig = float(np.min(np.linalg.eigvalsh(np.asarray(Om))))
    return m_free, float(np.linalg.norm(np.asarray(res))), mineig


def continuation(g, h, L, n_steps=20, target=(0, 0)):
    """Homotopy from free (0,0) to (g,h) along a straight line; minimal-change projection."""
    s_t = build_structure(L, 1.0, g, h)
    cl, cidx = s_t["canon_list"], s_t["cidx"]
    tidx = cidx[canon(target)]
    m_free = np.array([float(free_moment(c)) for c in cl[1:]])   # free (g=h=0) anchor
    ts = np.linspace(0.0, 1.0, n_steps + 1)
    last = None
    for t in ts:
        s = build_structure(L, 1.0, t * g, t * h)
        m_free, r, mineig = project_min_change(s, m_free)
        last = (float(t), r, mineig, float(np.concatenate([[1.0], m_free])[tidx]))
    return last, m_free  # (t=1, residual, min eig Omega, m[target])


if __name__ == "__main__":
    print("MOMENT-SPACE CONTINUATION (exact factorization + homotopy), KZ model")
    print("GT: g=1,h=1 -> [0.4204,0.4224];  g=0.5,h=1 -> [0.4803,0.4842]\n")
    for (g, h, isl) in [(1.0, 1.0, "[0.4204,0.4224]"), (0.5, 1.0, "[0.4803,0.4842]")]:
        print(f"g={g} h={h}  island {isl}:")
        for L in (4, 6, 8):
            t0 = time.time()
            (t, r, mineig, m), _ = continuation(g, h, L, n_steps=20)
            print(f"  L={L}: m[A^2]={m:.5f}  (loop res={r:.1e}, min eig Omega={mineig:+.2e})  {time.time()-t0:.0f}s")
        print()
