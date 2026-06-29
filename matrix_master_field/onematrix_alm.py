"""onematrix_alm.py — exact-factorization (Q = m mᵀ) master-field solver via ALM.

The conventional bootstrap (bootstrap_sdp.bootstrap_one_matrix) relaxes the factorized
loop-equation RHS  Σ_j m_j m_{n-1-j}  to a convex product matrix  Q ⪰ m mᵀ, and the
bound balloons (Gaussian m₄ ∈ [2.0, 4.05], m₈⁺ unbounded). This module instead imposes
the factorization EXACTLY (Q = m mᵀ, i.e. the true quadratic loop equations) by an
augmented-Lagrangian method, selected by Hankel positivity. That is the non-convex
problem the SDP cannot represent; it pins the unique large-N moment sequence.

Model V(M)=½M²+(g/4)M⁴, V'=M+gM³.  Even moments μ_p = m_{2p}, μ₀ = 1 (odd moments 0).
Factorized loop equation:
    μ_{p+1} + g·μ_{p+2} = Σ_{q=0}^{p} μ_q μ_{p-q},   p = 0,1,2,…
Hankel (symmetric Hamburger): H[i,j] = m_{i+j} (odd entries 0), H ⪰ 0.

`alm_min_max(g, K, target_p)` returns the [min, max] of m_{2·target_p} over
{loop eqs (exact) + Hankel ⪰ 0} — the exact-factorization bracket, to compare against
the convex bracket and the exact answer.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from matrix_master_field.one_matrix import gaussian_moments, quartic_moments_from_sd


def _full_moments(mu):
    """μ = [μ₀..μ_K] (even moments) → m = [m₀..m_{2K}] with odd entries 0."""
    K = mu.shape[0] - 1
    return jnp.zeros(2 * K + 1).at[0 : 2 * K + 1 : 2].set(mu)


def loop_residuals(mu, g):
    """h_p = μ_{p+1} + g μ_{p+2} − Σ_{q=0}^p μ_q μ_{p-q},  p = 0..K−2  (needs μ_{p+2})."""
    K = mu.shape[0] - 1
    return jnp.array(
        [mu[p + 1] + g * mu[p + 2] - jnp.sum(mu[: p + 1] * mu[: p + 1][::-1]) for p in range(K - 1)]
    )


def hankel(mu):
    """Symmetric Hamburger Hankel H[i,j] = m_{i+j} (odd moments 0), size (K+1)²."""
    m = _full_moments(mu)
    K = mu.shape[0] - 1
    return jnp.array([[m[i + j] for j in range(K + 1)] for i in range(K + 1)])


def _alm_one(g, K, target_p, sign, mu_init, *, n_outer=70, max_inner=300, pen_psd=1e4):
    """One augmented-Lagrangian solve: extremize sign·m_{2·target_p} subject to the
    loop equations = 0 (multipliers λ + growing penalty ρ) and Hankel ⪰ 0 (a fixed
    hard penalty on negative eigenvalues — keeps the iterate in the PSD cone, unlike
    a barrier the extremizer can escape)."""

    def lagrangian(mu_free, lam, rho):
        mu = jnp.concatenate([jnp.ones(1), mu_free])
        h = loop_residuals(mu, g)
        neg = jnp.minimum(jnp.linalg.eigvalsh(hankel(mu)), 0.0)  # negative eigenvalues
        return (sign * mu[target_p] - jnp.dot(lam, h) + 0.5 * rho * jnp.dot(h, h)
                + pen_psd * jnp.dot(neg, neg))

    vg = jax.jit(jax.value_and_grad(lagrangian))
    mu_free = np.asarray(mu_init, float)
    lam = np.zeros(K - 1)
    rho, prev_h = 1.0, np.inf
    for _ in range(n_outer):
        def fg(x):
            v, gr = vg(jnp.asarray(x), jnp.asarray(lam), jnp.asarray(rho))
            return float(v), np.asarray(gr, float)

        mu_free = minimize(fg, mu_free, jac=True, method="L-BFGS-B",
                           options={"maxiter": max_inner}).x
        mu = np.concatenate([[1.0], mu_free])
        h = np.asarray(loop_residuals(jnp.asarray(mu), g))
        nh = float(np.linalg.norm(h))
        lam = lam - rho * h
        if nh > 0.25 * prev_h:
            rho = min(rho * 3.0, 1e9)
        prev_h = nh
    mu = np.concatenate([[1.0], mu_free])
    return mu, float(np.linalg.norm(np.asarray(loop_residuals(jnp.asarray(mu), g))))


def alm_min_max(g, K, target_p=1, *, restarts=4):
    """[min, max] of m_{2·target_p} over the exact-factorization feasible set.

    Multi-start (the V6 artifact guard); returns (lo, hi, residual, mu_at_lo, mu_at_hi).
    Init from the Catalan (semicircle) even moments — a guaranteed PD-Hankel start.
    """
    mu0 = gaussian_moments(2 * K)[0 : 2 * K + 1 : 2][1:]  # μ_1..μ_K (Catalan)
    rng = np.random.default_rng(0)
    inits = [mu0] + [np.abs(mu0 * rng.uniform(0.5, 1.5, size=K)) for _ in range(restarts - 1)]
    los, his, res = [], [], []
    for mi in inits:
        mu_lo, r1 = _alm_one(g, K, target_p, +1.0, mi)
        mu_hi, r2 = _alm_one(g, K, target_p, -1.0, mi)
        los.append((mu_lo[target_p], mu_lo)); his.append((mu_hi[target_p], mu_hi)); res.append(max(r1, r2))
    lo, mu_lo = min(los, key=lambda t: t[0])
    hi, mu_hi = max(his, key=lambda t: t[0])
    return float(lo), float(hi), float(max(res)), mu_lo, mu_hi


# ── Method A: operator (Jacobi) master field — positivity automatic ──────────


def jacobi_even_moments(b, K):
    """Even moments μ_0..μ_K, μ_p = (J^{2p})_{00}, for the symmetric tridiagonal
    Jacobi operator J (zero diagonal, off-diagonals b, size len(b)+1 ≥ K+1).
    A self-adjoint J ⟹ the moments are those of a genuine measure: Hankel ⪰ 0 and
    factorization hold BY CONSTRUCTION (no penalty needed)."""
    L = b.shape[0]
    i = jnp.arange(L)
    J = jnp.zeros((L + 1, L + 1)).at[i, i + 1].set(b).at[i + 1, i].set(b)
    u = jnp.zeros(L + 1).at[0].set(1.0)
    mus = [jnp.ones(())]
    for _ in range(K):
        u = J @ (J @ u)
        mus.append(u[0])
    return jnp.stack(mus)


def method_a_jacobi(g, K, *, L=None, w_smooth=1e-4, maxiter=4000):
    """Operator master field: minimise the loop-equation residual over the Jacobi
    off-diagonals b = exp(θ) > 0 (positivity automatic). A light smoothness penalty
    selects the regular (physical) branch among the 1-parameter truncation family.
    Returns (even moments μ_0..μ_K, loop residual, b)."""
    if L is None:
        L = K + 4

    def obj(theta):
        b = jnp.exp(theta)
        mu = jacobi_even_moments(b, K)
        return jnp.sum(loop_residuals(mu, g) ** 2) + w_smooth * jnp.sum(jnp.diff(b) ** 2)

    vg = jax.jit(jax.value_and_grad(obj))

    def fg(x):
        v, gr = vg(jnp.asarray(x))
        return float(v), np.asarray(gr, float)

    theta = minimize(fg, np.zeros(L), jac=True, method="L-BFGS-B",
                     options={"maxiter": maxiter}).x
    b = np.exp(theta)
    mu = np.asarray(jacobi_even_moments(jnp.asarray(b), K))
    return mu, float(np.linalg.norm(np.asarray(loop_residuals(jnp.asarray(mu), g)))), b


if __name__ == "__main__":
    print("=" * 74)
    print("Exact-factorization (Q=m mᵀ) ALM vs exact answer — one-matrix")
    print("=" * 74)

    # g=0 sanity: loop equations alone must pin Catalan (m₂,m₄,m₆ = 1,2,5).
    K = 6
    mu, r = _alm_one(0.0, K, 1, +1.0, gaussian_moments(2 * K)[0 : 2 * K + 1 : 2][1:])
    cat = gaussian_moments(2 * K)
    print(f"\n[g=0] ALM even moments vs Catalan  (loop residual {r:.1e}):")
    for p in range(1, K):
        print(f"  m_{2*p}: ALM={mu[p]:.6f}  exact={cat[2*p]:.6f}")

    # g>0: exact-factorization bracket vs exact vs the convex bracket from the probe.
    for g in (0.5, 1.0):
        exact = quartic_moments_from_sd(g, 2 * K)
        print(f"\n[g={g}] exact-factorization [min,max] vs exact:")
        for p in (1, 2, 3):
            lo, hi, r, *_ = alm_min_max(g, K, p)
            print(f"  m_{2*p}: ALM=[{lo:.5f}, {hi:.5f}] width={hi-lo:.2e}  exact={exact[2*p]:.5f}")
