"""Sparse Cuntz-Fock moment evaluator — scales to large cutoffs (max_word_len ≥ 5,
dim ≥ 8191) where dense D×D operators are infeasible.

The basic a_i, a†_i and the monomials â†_u â_v are 0/1 partial permutations on the
word basis: each basis |w⟩ maps to at most one basis state. Concretely (matching
`cuntz_fock`: â†_i|w⟩=|iw⟩, â_i|jw⟩=δ_ij|w⟩, â†=âᵀ),

    â†_u â_v |w⟩ = |u · w[|v|:]⟩   iff   w[:|v|] == reverse(v)  and  |result| ≤ cutoff,

else 0. We precompute these transitions as (src→tgt) index arrays and apply
M̂_i = (A_i + A_iᵀ)/2 with A_i = Σ_k c_{i,k} monomial_k via JAX scatter-add — fully
differentiable in the coefficients c. Same monomial set / ordering / param layout
as `MultiMonomialAnsatz`, and validated against it at small cutoff.
"""

from itertools import product as _product

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _words(n, max_len):
    out = {0: [()]}
    for L in range(1, max_len + 1):
        out[L] = [tuple(c) for c in _product(range(n), repeat=L)]
    return out


class SparseMonomialField:
    """Sparse multi-matrix monomial ansatz + moment evaluator on the truncated
    Cuntz-Fock space. Mirrors `MultiMonomialAnsatz` (same monomials, ordering and
    params shape (n, n_monomials)) but never materializes D×D matrices."""

    def __init__(self, n_matrices: int, cutoff: int, degree: int):
        self.n = n_matrices
        self.cutoff = cutoff
        self.degree = degree

        # basis: words of length 0..cutoff (vacuum = index 0), matching enumerate_words
        basis = [()]
        for L in range(1, cutoff + 1):
            basis += [tuple(c) for c in _product(range(n_matrices), repeat=L)]
        self.D = len(basis)
        idx = {w: i for i, w in enumerate(basis)}

        # monomials (u,v) with |u|+|v| ≤ degree, in MultiMonomialAnsatz.keys order
        words = _words(n_matrices, degree)
        keys = []
        for total in range(degree + 1):
            for p in range(total + 1):
                q = total - p
                for u in words[p]:
                    for v in words[q]:
                        keys.append((u, v))
        self.keys = keys
        self.n_monomials = len(keys)
        self.n_params = self.n * self.n_monomials
        self._free_idx = [keys.index(((), (i,))) for i in range(n_matrices)]

        # transitions src -> tgt with coefficient-index cidx (one row per nonzero entry)
        src, tgt, cidx = [], [], []
        for k, (u, v) in enumerate(keys):
            q = len(v)
            rv = tuple(reversed(v))
            for w, i in idx.items():
                if w[:q] == rv:
                    res = u + w[q:]
                    if len(res) <= cutoff:
                        src.append(i)
                        tgt.append(idx[res])
                        cidx.append(k)
        self.src = jnp.asarray(src, dtype=jnp.int32)
        self.tgt = jnp.asarray(tgt, dtype=jnp.int32)
        self.cidx = jnp.asarray(cidx, dtype=jnp.int32)
        self.n_transitions = len(src)

    def init_params(self, key):
        return 0.01 * jax.random.normal(key, (self.n, self.n_monomials), dtype=jnp.float64)

    def params_for_free_field(self):
        """M̂_i = â_i + â†_i: coeff 2 on the â_i monomial (u=(), v=(i,)) per matrix."""
        p = np.zeros((self.n, self.n_monomials))
        for i in range(self.n):
            p[i, self._free_idx[i]] = 2.0
        return jnp.asarray(p, dtype=jnp.float64)

    def apply_Mi(self, c_i, x):
        """M̂_i x = (A_i x + A_iᵀ x)/2, A_i = Σ_k c_{i,k} monomial_k (sparse scatter-add)."""
        coeff = c_i[self.cidx]  # coefficient on each transition
        zero = jnp.zeros(self.D, dtype=x.dtype)
        ax = zero.at[self.tgt].add(coeff * x[self.src])   # (A_i x)[tgt] += c · x[src]
        atx = zero.at[self.src].add(coeff * x[self.tgt])  # (A_iᵀ x)[src] += c · x[tgt]
        return 0.5 * (ax + atx)

    def word_moment(self, params, word):
        """τ(word) = ⟨Ω| M̂_{w₀} … M̂_{w_{k-1}} |Ω⟩ via right-to-left sparse applies."""
        x = jnp.zeros(self.D, dtype=jnp.float64).at[0].set(1.0)  # |Ω⟩
        for i in reversed(tuple(word)):
            x = self.apply_Mi(params[i], x)
        return x[0]

    def build_dense_operators(self, params):
        """Dense M̂_i (for cross-checking only; do NOT use at large cutoff)."""
        ops = []
        for i in range(self.n):
            A = np.zeros((self.D, self.D))
            ci = np.asarray(params[i])
            s, t, c = np.asarray(self.src), np.asarray(self.tgt), np.asarray(self.cidx)
            np.add.at(A, (t, s), ci[c])
            ops.append((A + A.T) / 2.0)
        return ops
