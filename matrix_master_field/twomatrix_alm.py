"""twomatrix_alm.py — exact-factorization (G = m mᵀ) master field via ALM, two-matrix.

Generalizes onematrix_alm to the two-matrix models whose convex bootstrap relaxes
factorization to a product matrix G ⪰ m mᵀ (bootstrap_sdp._bootstrap_two_matrix):

  • commutator+mass:  S = N·tr[½(M₀²+M₁²) − (g²/4)[M₀,M₁]²]   (mass 1, comm g²/2)
  • Kazakov–Zheng:    S = N·tr[½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]²]  (mass 1, quartic g, comm h)

Imposing G = m mᵀ EXACTLY (the true factorized loop equations) by an augmented-Lagrangian
method — the non-convex problem the SDP cannot represent — collapses the wide convex
bracket onto a sharp determination. Here the loop equations do NOT close by a
one-parameter recursion (genuine multi-matrix), so the convex relaxation is genuinely
loose: the decisive test.

Single-trace moments m[w] (real), canonicalized by cyclicity + M₀↔M₁ exchange + Z₂×Z₂
parity. At comm=0 the matrices are free; `free_moment` (non-crossing same-letter
pairings) is the exact anchor + initializer.
"""
from __future__ import annotations

import itertools
from functools import lru_cache

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from matrix_master_field.bootstrap_sdp import _two_matrix_canon as canon


@lru_cache(maxsize=None)
def free_moment(w: tuple) -> int:
    """τ(w) for two FREE semicirculars of variance 1 = number of non-crossing perfect
    matchings of the positions of w pairing equal letters. Exact at comm=0."""
    if len(w) == 0:
        return 1
    if len(w) % 2 == 1:
        return 0
    return sum(
        free_moment(w[1:j]) * free_moment(w[j + 1 :])
        for j in range(1, len(w))
        if w[j] == w[0]
    )


def build_structure(max_word_len: int, mass: float, quartic: float, comm: float):
    """Canonical-moment indexing, the Ω-Gram index map, and the factorized loop
    equations (as index arrays) for the ALM. Mirrors bootstrap_sdp._bootstrap_two_matrix
    but returns numeric structure instead of cvxpy constraints."""
    words = [()]
    for L in range(1, max_word_len + 1):
        words += [tuple(c) for c in itertools.product((0, 1), repeat=L)]
    canon_list = sorted({canon(w) for w in words} - {None}, key=lambda t: (len(t), t))
    cidx = {c: i for i, c in enumerate(canon_list)}
    nvar = len(canon_list)
    SENT = nvar  # sentinel → appended 0 (parity-forbidden / out-of-set)

    def idx(w):
        c = canon(tuple(w))
        return SENT if c is None else cidx[c]

    half = max_word_len // 2
    basis = [()]
    for L in range(1, half + 1):
        basis += [tuple(c) for c in itertools.product((0, 1), repeat=L)]
    nb = len(basis)
    omega_idx = np.full((nb, nb), SENT, dtype=int)
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            omega_idx[i, j] = idx(tuple(reversed(u)) + v)

    test_words = [()]
    for L in range(1, max(1, max_word_len - 3) + 1):
        test_words += [tuple(c) for c in itertools.product((0, 1), repeat=L)]
    loop_eqs = []  # (lhs_coeffs, lhs_idx, rhs_cl_idx, rhs_cr_idx)
    for w in test_words:
        for a in (0, 1):
            b = 1 - a
            lc, li = [mass], [idx((a,) + w)]
            if quartic != 0.0:
                lc.append(quartic); li.append(idx((a, a, a) + w))
            if comm != 0.0:
                lc += [comm, comm, -2.0 * comm]
                li += [idx((a, b, b) + w), idx((b, b, a) + w), idx((b, a, b) + w)]
            rcl, rcr = [], []
            for j in range(len(w)):
                if w[j] == a:
                    rcl.append(idx(w[:j])); rcr.append(idx(w[j + 1 :]))
            loop_eqs.append((np.array(lc), np.array(li, dtype=int),
                             np.array(rcl, dtype=int), np.array(rcr, dtype=int)))
    return dict(canon_list=canon_list, cidx=cidx, nvar=nvar, omega_idx=omega_idx,
                loop_eqs=loop_eqs)


def _make_pieces(struct):
    omega_j = jnp.asarray(struct["omega_idx"])
    eqs = [(jnp.asarray(lc), jnp.asarray(li), jnp.asarray(rcl), jnp.asarray(rcr))
           for (lc, li, rcl, rcr) in struct["loop_eqs"]]
    return omega_j, eqs


def _moments_vec(m_free):
    """[m_(), free..., 0_sentinel]; m[()] = index 0 = 1 (canon_list[0] is ())."""
    return jnp.concatenate([jnp.ones(1), m_free, jnp.zeros(1)])


def residual_and_omega(m_free, omega_j, eqs):
    m = _moments_vec(m_free)
    res = jnp.stack([jnp.sum(lc * m[li]) - jnp.sum(m[rcl] * m[rcr]) for (lc, li, rcl, rcr) in eqs])
    Om = m[omega_j]
    return res, 0.5 * (Om + Om.T)


def _alm_one(struct, target_idx, sign, m_init, *, n_outer=80, max_inner=300, pen_psd=1e4):
    omega_j, eqs = _make_pieces(struct)

    def lag(m_free, lam, rho):
        res, Om = residual_and_omega(m_free, omega_j, eqs)
        neg = jnp.minimum(jnp.linalg.eigvalsh(Om), 0.0)
        return (sign * _moments_vec(m_free)[target_idx] - jnp.dot(lam, res)
                + 0.5 * rho * jnp.dot(res, res) + pen_psd * jnp.dot(neg, neg))

    vg = jax.jit(jax.value_and_grad(lag))
    m_free = np.asarray(m_init, float)
    lam = np.zeros(len(eqs))
    rho, prev = 1.0, np.inf
    for _ in range(n_outer):
        def fg(x):
            v, gr = vg(jnp.asarray(x), jnp.asarray(lam), jnp.asarray(rho))
            return float(v), np.asarray(gr, float)
        m_free = minimize(fg, m_free, jac=True, method="L-BFGS-B",
                          options={"maxiter": max_inner}).x
        res, _ = residual_and_omega(jnp.asarray(m_free), omega_j, eqs)
        res = np.asarray(res); nh = float(np.linalg.norm(res))
        lam = lam - rho * res
        if nh > 0.25 * prev:
            rho = min(rho * 3.0, 1e9)
        prev = nh
    res, _ = residual_and_omega(jnp.asarray(m_free), omega_j, eqs)
    return np.concatenate([[1.0], m_free]), float(np.linalg.norm(np.asarray(res)))


def alm_min_max(struct, target_word, *, restarts=3):
    """[min, max] of m[target_word] over {exact loop eqs + Ω ⪰ 0}, multi-start from the
    free (comm=0) moments. Returns (lo, hi, residual)."""
    cl = struct["canon_list"]
    m0 = np.array([float(free_moment(c)) for c in cl[1:]])  # free init (drop m[()])
    tidx = struct["cidx"][canon(target_word)]
    rng = np.random.default_rng(0)
    inits = [m0] + [m0 * rng.uniform(0.7, 1.3, size=m0.shape) for _ in range(restarts - 1)]
    los, his, res = [], [], []
    for mi in inits:
        m_lo, r1 = _alm_one(struct, tidx, +1.0, mi)
        m_hi, r2 = _alm_one(struct, tidx, -1.0, mi)
        los.append(m_lo[tidx]); his.append(m_hi[tidx]); res.append(max(r1, r2))
    return float(min(los)), float(max(his)), float(max(res))


def _project_alm(struct, m_prev_free, *, n_outer=50, max_inner=200, pen_psd=1e4):
    """Minimal-change projection: argmin ||m_free − m_prev||² s.t. exact loop eqs = 0
    (ALM) and Ω ⪰ 0 (eigenvalue penalty). Tracks the physical branch under homotopy."""
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
        res, _ = residual_and_omega(jnp.asarray(m_free), omega_j, eqs)
        res = np.asarray(res); nh = float(np.linalg.norm(res))
        lam = lam - rho * res
        if nh > 0.25 * prev:
            rho = min(rho * 3.0, 1e9)
        prev = nh
    res, _ = residual_and_omega(jnp.asarray(m_free), omega_j, eqs)
    return m_free, float(np.linalg.norm(np.asarray(res)))


def homotopy_commutator(g_target, L=4, n_steps=24, target_word=(0, 0)):
    """Track the master field of the commutator model from g=0 (free) to g_target,
    minimal-change continuation. Returns [(g, m[target_word]), ...]."""
    s0 = build_structure(L, 1.0, 0.0, 0.0)
    cl, cidx = s0["canon_list"], s0["cidx"]
    tidx = cidx[canon(target_word)]
    m_free = np.array([float(free_moment(c)) for c in cl[1:]])
    traj = [(0.0, float(np.concatenate([[1.0], m_free])[tidx]))]
    for g in np.linspace(0.0, g_target, n_steps + 1)[1:]:
        s = build_structure(L, 1.0, 0.0, g * g / 2.0)
        m_free, r = _project_alm(s, m_free)
        traj.append((float(g), float(np.concatenate([[1.0], m_free])[tidx])))
    return traj


if __name__ == "__main__":
    from matrix_master_field.bootstrap_sdp import bootstrap_two_matrix, bootstrap_two_matrix_kz

    L = 4
    print("=" * 74)
    print("Two-matrix exact-factorization ALM vs convex bootstrap")
    print("=" * 74)

    # Sanity: at comm=0 the commutator model is two free Gaussians — free_moment must
    # satisfy the loop equations exactly and give Ω ⪰ 0.
    s0 = build_structure(L, mass=1.0, quartic=0.0, comm=0.0)
    m0 = np.array([float(free_moment(c)) for c in s0["canon_list"][1:]])
    omj, eqs = _make_pieces(s0)
    res0, Om0 = residual_and_omega(jnp.asarray(m0), omj, eqs)
    print(f"\n[comm=0 free anchor]  loop residual={float(np.linalg.norm(np.asarray(res0))):.1e}  "
          f"min eig(Ω)={float(np.min(np.linalg.eigvalsh(np.asarray(Om0)))):.3f}")

    # (1) commutator model
    for g in (0.5, 1.0):
        s = build_structure(L, mass=1.0, quartic=0.0, comm=g * g / 2.0)
        lo, hi, r = alm_min_max(s, (0, 0))
        clo = bootstrap_two_matrix(g, L, (0, 0), maximize=False)
        chi = bootstrap_two_matrix(g, L, (0, 0), maximize=True)
        print(f"\n[commutator g={g}]  m[M₀²]:")
        print(f"   convex = [{clo:.4f}, {chi:.4f}]  width={chi-clo:.2e}")
        print(f"   ALM    = [{lo:.4f}, {hi:.4f}]  width={hi-lo:.2e}  (res {r:.1e})")

    # (2) Kazakov–Zheng model
    for (g, h) in ((1.0, 1.0), (0.5, 1.0)):
        s = build_structure(L, mass=1.0, quartic=g, comm=h)
        lo, hi, r = alm_min_max(s, (0, 0))
        clo = bootstrap_two_matrix_kz(g, h, L, (0, 0), maximize=False)
        chi = bootstrap_two_matrix_kz(g, h, L, (0, 0), maximize=True)
        print(f"\n[KZ g={g} h={h}]  m[A²]:")
        print(f"   convex = [{clo:.4f}, {chi:.4f}]  width={chi-clo:.2e}")
        print(f"   ALM    = [{lo:.4f}, {hi:.4f}]  width={hi-lo:.2e}  (res {r:.1e})")
