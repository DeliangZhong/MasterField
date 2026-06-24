"""Loop-equation (Schwinger-Dyson) residual losses on master-field moments.

One-matrix V'(M)=Σ_k v_k M^k:  Σ_k v_k m_{n+k} = Σ_{j} m_j m_{n-j-1}.
Two-matrix commutator+mass model (matches schwinger_dyson.TwoMatrixSD), action
S = N·tr[½(M₁²+M₂²) − (g²/4)[M₁,M₂]²]:
    V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a − 2 M_b M_a M_b),
    ⟨tr(V'_a·w)⟩ = Σ_{j: w[j]=a} ⟨tr w_left⟩⟨tr w_right⟩  (factorized at N=∞).
Relative-scaled mean-squared residual; differentiable in the moments/operators.
"""

from itertools import product as _product

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from matrix_master_field.fock_jax import word_moment  # noqa: E402


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


def two_matrix_test_words(max_len, n=2):
    """All words (incl. empty) up to length max_len over the n-letter alphabet."""
    words = [()]
    for length in range(1, max_len + 1):
        words.extend(tuple(c) for c in _product(range(n), repeat=length))
    return words


def two_matrix_sd_residual(ops_list, test_words, g):
    """Commutator+mass SD residual. ops_list=[M̂_0,M̂_1]; g the coupling (λ=g²)."""
    g2 = float(g) * float(g)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n_eqs = 0
    for w in test_words:
        for a in (0, 1):
            b = 1 - a
            lhs = word_moment(ops_list, (a,) + w)
            if g2 != 0.0:
                lhs = lhs + (g2 / 2.0) * (
                    word_moment(ops_list, (a, b, b) + w)
                    + word_moment(ops_list, (b, b, a) + w)
                    - 2.0 * word_moment(ops_list, (b, a, b) + w)
                )
            rhs = jnp.asarray(0.0, dtype=jnp.float64)
            for j in range(len(w)):
                if w[j] == a:
                    rhs = rhs + word_moment(ops_list, w[:j]) * word_moment(ops_list, w[j + 1:])
            scale = jnp.maximum(jnp.abs(lhs) + jnp.abs(rhs), 1.0)
            total = total + ((lhs - rhs) / scale) ** 2
            n_eqs += 1
    return total / max(n_eqs, 1)
