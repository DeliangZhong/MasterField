"""Loop-equation (Schwinger-Dyson) residual losses on master-field moments.

The one-matrix SD equation for V'(M)=Σ_k v_k M^k is, at N=∞:
    Σ_k v_k m_{n+k} = Σ_{j=0}^{n-1} m_j m_{n-j-1}   (factorized RHS).
We minimize the relative-scaled mean-squared residual (moments grow fast, so
absolute residuals at high n would dominate). Differentiable in the moments.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def one_matrix_sd_residual(moments, v_prime_coeffs):
    """Mean-squared relative SD residual. `moments` = [m_0..m_K], m_0=1."""
    m = jnp.asarray(moments, dtype=jnp.float64)
    K = m.shape[0] - 1
    n_v = len(v_prime_coeffs)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n_eqs = 0
    for n in range(0, max(1, K - n_v + 1)):
        lhs = jnp.asarray(0.0, dtype=jnp.float64)
        for k in range(n_v):
            if n + k <= K:
                lhs = lhs + v_prime_coeffs[k] * m[n + k]
        rhs = jnp.asarray(0.0, dtype=jnp.float64)
        for j in range(n):
            if (n - j - 1) <= K:
                rhs = rhs + m[j] * m[n - j - 1]
        scale = jnp.maximum(jnp.abs(lhs) + jnp.abs(rhs), 1.0)
        total = total + ((lhs - rhs) / scale) ** 2
        n_eqs += 1
    return total / n_eqs
