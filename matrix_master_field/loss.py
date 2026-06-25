"""Loop-equation (Schwinger-Dyson) residual losses on master-field moments.

One-matrix V'(M)=Σ_k v_k M^k:  Σ_k v_k m_{n+k} = Σ_{j} m_j m_{n-j-1}.
Two-matrix commutator+mass model (matches schwinger_dyson.TwoMatrixSD), action
S = N·tr[½(M₁²+M₂²) − (g²/4)[M₁,M₂]²]:
    V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a − 2 M_b M_a M_b),
    ⟨tr(V'_a·w)⟩ = Σ_{j: w[j]=a} ⟨tr w_left⟩⟨tr w_right⟩  (factorized at N=∞).

Symmetry losses enforce the properties the Cuntz vacuum does NOT give for free:
trace cyclicity (the vacuum is not tracial), M₁↔M₂ exchange, and Z₂ (M→−M).
Relative-scaled mean-squared residuals; differentiable in the moments/operators.
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


def sd_residual_from_moment(moment, test_words, g):
    """Commutator+mass SD residual from a moment callable `moment(word)->scalar`.

    Backend-agnostic: pass `lambda w: word_moment(ops_list, w)` (dense) or
    `lambda w: field.word_moment(params, w)` (sparse). Both are differentiable.
    """
    g2 = float(g) * float(g)
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n_eqs = 0
    for w in test_words:
        for a in (0, 1):
            b = 1 - a
            lhs = moment((a,) + w)
            if g2 != 0.0:
                lhs = lhs + (g2 / 2.0) * (
                    moment((a, b, b) + w)
                    + moment((b, b, a) + w)
                    - 2.0 * moment((b, a, b) + w)
                )
            rhs = jnp.asarray(0.0, dtype=jnp.float64)
            for j in range(len(w)):
                if w[j] == a:
                    rhs = rhs + moment(w[:j]) * moment(w[j + 1:])
            scale = jnp.maximum(jnp.abs(lhs) + jnp.abs(rhs), 1.0)
            total = total + ((lhs - rhs) / scale) ** 2
            n_eqs += 1
    return total / max(n_eqs, 1)


def two_matrix_sd_residual(ops_list, test_words, g):
    """Commutator+mass SD residual. ops_list=[M̂_0,M̂_1]; g the coupling (λ=g²)."""
    return sd_residual_from_moment(lambda w: word_moment(ops_list, w), test_words, g)


# ─── Symmetry losses (NOT automatic on the Cuntz vacuum) ──────────────────────

def _rotations(w):
    return [w[i:] + w[:i] for i in range(len(w))]


def cyclicity_from_moment(moment, words):
    """Penalize τ(w) ≠ τ(cyclic rotation of w) — enforces a tracial state."""
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n = 0
    for w in words:
        if len(w) < 2:
            continue
        rots = _rotations(w)
        m0 = moment(rots[0])
        for r in rots[1:]:
            total = total + (moment(r) - m0) ** 2
            n += 1
    return total / max(n, 1)


def exchange_from_moment(moment, words):
    """Penalize τ(w(M₁,M₂)) ≠ τ(w(M₂,M₁)) — M₁↔M₂ exchange symmetry (n=2)."""
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n = 0
    for w in words:
        if not w:
            continue
        sw = tuple(1 - x for x in w)
        total = total + (moment(w) - moment(sw)) ** 2
        n += 1
    return total / max(n, 1)


def z2_from_moment(moment, words):
    """Z₂×Z₂ parity: each M_i→−M_i is independently a symmetry of the
    commutator+mass action, so any word with an ODD count of ANY generator has a
    vanishing moment (e.g. ⟨tr M₀M₁⟩, ⟨tr M₀³M₁⟩). Stronger than odd-total-length:
    cyclicity/exchange only RELATE such moments; they do not force them to zero.
    """
    total = jnp.asarray(0.0, dtype=jnp.float64)
    n = 0
    for w in words:
        if any(w.count(c) % 2 == 1 for c in set(w)):
            total = total + moment(w) ** 2
            n += 1
    return total / max(n, 1)


def symmetry_losses_from_moment(moment, words):
    """Sum of cyclicity + exchange + Z₂ penalties from a moment callable."""
    return (
        cyclicity_from_moment(moment, words)
        + exchange_from_moment(moment, words)
        + z2_from_moment(moment, words)
    )


def _dense_moment(ops_list):
    return lambda w: word_moment(ops_list, w)


def cyclicity_loss(ops_list, words):
    """Penalize τ(w) ≠ τ(cyclic rotation of w) — enforces a tracial state."""
    return cyclicity_from_moment(_dense_moment(ops_list), words)


def exchange_loss(ops_list, words):
    """Penalize τ(w(M₁,M₂)) ≠ τ(w(M₂,M₁)) — M₁↔M₂ exchange symmetry (n=2)."""
    return exchange_from_moment(_dense_moment(ops_list), words)


def z2_loss(ops_list, words):
    """Z₂×Z₂ parity: any word with an ODD count of ANY generator vanishes."""
    return z2_from_moment(_dense_moment(ops_list), words)


def symmetry_losses(ops_list, words):
    """Sum of cyclicity + exchange + Z₂ penalties."""
    return symmetry_losses_from_moment(_dense_moment(ops_list), words)
