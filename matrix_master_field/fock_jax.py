"""JAX moment evaluation on the truncated Cuntz-Fock space.

Provides constant creation/annihilation operators as jnp matrices and
differentiable vacuum-expectation (moment) evaluators. The master-field
moments are τ(word) = ⟨Ω| M̂_{w₁}…M̂_{w_k} |Ω⟩; positivity of τ is automatic
because it is a vacuum vector state.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from matrix_master_field.cuntz_fock import CuntzFockSpace  # noqa: E402


class FockOps:
    """Constant â_i, â†_i on the truncated Cuntz-Fock space, as jnp matrices."""

    def __init__(self, n_matrices: int, max_length: int):
        base = CuntzFockSpace(n_matrices, max_length)
        self.n = n_matrices
        self.max_length = max_length  # Fock word-length cutoff
        self.D = base.dim
        self.a = [jnp.asarray(base.a(i), dtype=jnp.float64) for i in range(n_matrices)]
        self.adag = [jnp.asarray(base.adag(i), dtype=jnp.float64) for i in range(n_matrices)]
        self.vacuum = jnp.zeros(self.D, dtype=jnp.float64).at[0].set(1.0)


def word_moment(ops_list, word):
    """⟨Ω| M̂_{w₀} … M̂_{w_{k-1}} |Ω⟩ via right-to-left matvecs (differentiable)."""
    D = ops_list[0].shape[0]
    v = jnp.zeros(D, dtype=jnp.float64).at[0].set(1.0)  # |Ω⟩
    for idx in reversed(tuple(word)):
        v = ops_list[idx] @ v
    return v[0]


def power_moments(M, K):
    """[⟨Ω|Mᵖ|Ω⟩]_{p=0..K} for a single operator M (differentiable)."""
    D = M.shape[0]
    v = jnp.zeros(D, dtype=jnp.float64).at[0].set(1.0)
    out = [v[0]]
    for _ in range(K):
        v = M @ v
        out.append(v[0])
    return jnp.stack(out)
