"""Operator ansätze for the master field — pluggable, all Hermitian.

Each ansatz exposes:
    init_params(key) -> pytree
    build_operators(params) -> list[jnp D×D Hermitian]   (one per matrix)

All guarantee M̂_i = M̂_i†, so the vacuum state τ(·)=⟨Ω|·|Ω⟩ is automatically a
positive state. Three single-matrix ansätze were compared in Milestone 2
(monomial wins); MultiMonomialAnsatz is the Milestone-3 multi-matrix extension.
"""

from itertools import product as _product

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from matrix_master_field.fock_jax import FockOps  # noqa: E402


class MonomialAnsatz:
    """M̂ = Σ_k c_k H_k with H_k = (â†)^p â^q + h.c. (p ≤ q, p+q ≤ degree).

    Each H_k is real-symmetric (â† = âᵀ here), so M̂ is Hermitian for real c_k.
    Single-matrix (n_matrices == 1).
    """

    def __init__(self, fock_ops: FockOps, degree: int):
        if fock_ops.n != 1:
            raise NotImplementedError("MonomialAnsatz is single-matrix; use MultiMonomialAnsatz")
        self.ops = fock_ops
        self.degree = degree
        D = fock_ops.D
        a = np.asarray(fock_ops.a[0])
        adag = np.asarray(fock_ops.adag[0])

        a_pow = [np.eye(D)]
        adag_pow = [np.eye(D)]
        for _ in range(degree):
            a_pow.append(a_pow[-1] @ a)
            adag_pow.append(adag_pow[-1] @ adag)

        keys, mats = [], []
        for q in range(degree + 1):
            for p in range(q + 1):
                if p + q > degree:
                    continue
                mono = adag_pow[p] @ a_pow[q]
                H = mono if p == q else mono + adag_pow[q] @ a_pow[p]
                keys.append((p, q))
                mats.append(H)

        self.keys = keys
        self.H = jnp.asarray(np.stack(mats), dtype=jnp.float64)
        self.n_params = len(keys)
        self._free_idx = keys.index((0, 1))

    def init_params(self, key):
        return 0.01 * jax.random.normal(key, (self.n_params,), dtype=jnp.float64)

    def params_for_free_field(self):
        return jnp.zeros(self.n_params, dtype=jnp.float64).at[self._free_idx].set(1.0)

    def build_operators(self, params):
        return [jnp.tensordot(params, self.H, axes=1)]


class DenseHermitianAnsatz:
    """M̂ = (W + Wᵀ)/2 — any real-symmetric D×D operator (maximal flexibility).

    Single-matrix comparison baseline (Milestone 2).
    """

    def __init__(self, fock_ops: FockOps):
        if fock_ops.n != 1:
            raise NotImplementedError("DenseHermitianAnsatz is single-matrix")
        self.ops = fock_ops
        self.D = fock_ops.D
        self.n_params = self.D * self.D

    def init_params(self, key):
        return 0.01 * jax.random.normal(key, (self.n_params,), dtype=jnp.float64)

    def build_operators(self, params):
        W = params.reshape(self.D, self.D)
        return [(W + W.T) / 2.0]


def _words(n, max_len):
    out = {0: [()]}
    for L in range(1, max_len + 1):
        out[L] = [tuple(c) for c in _product(range(n), repeat=L)]
    return out


class MultiMonomialAnsatz:
    """Multi-matrix monomial ansatz (Milestone 3).

    For each matrix i: A_i = Σ_k c^{(i)}_k · (â†_u â_v),  M̂_i = (A_i + A_iᵀ)/2,
    over all word-monomials (u,v) with |u|+|v| ≤ degree in the n-letter alphabet.
    The symmetrization makes M̂_i Hermitian (â† = âᵀ ⇒ real symmetric). params is
    an array of shape (n, n_monomials).
    """

    def __init__(self, fock_ops: FockOps, degree: int):
        self.ops = fock_ops
        self.n = fock_ops.n
        self.degree = degree
        D = fock_ops.D
        a = [np.asarray(fock_ops.a[i]) for i in range(self.n)]
        adag = [np.asarray(fock_ops.adag[i]) for i in range(self.n)]
        I = np.eye(D)

        def word_mat(letters, ops_list):
            m = I
            for x in letters:
                m = m @ ops_list[x]
            return m

        words = _words(self.n, degree)
        keys, mats = [], []
        for total in range(degree + 1):
            for p in range(total + 1):
                q = total - p
                for u in words[p]:
                    for v in words[q]:
                        mats.append(word_mat(u, adag) @ word_mat(v, a))  # â†_u â_v
                        keys.append((u, v))

        self.keys = keys
        self.M = jnp.asarray(np.stack(mats), dtype=jnp.float64)  # [P, D, D]
        self.n_monomials = len(keys)
        self.n_params = self.n * self.n_monomials
        # index of the pure-annihilation monomial â_i = (u=(), v=(i,)) per matrix
        self._free_idx = [keys.index(((), (i,))) for i in range(self.n)]

    def init_params(self, key):
        return 0.01 * jax.random.normal(key, (self.n, self.n_monomials), dtype=jnp.float64)

    def params_for_free_field(self):
        """M̂_i = â_i + â†_i:  A_i = 2 â_i  ⇒  (A_i+A_iᵀ)/2 = â_i + â_iᵀ."""
        p = np.zeros((self.n, self.n_monomials))
        for i in range(self.n):
            p[i, self._free_idx[i]] = 2.0
        return jnp.asarray(p, dtype=jnp.float64)

    def build_operators(self, params):
        ops = []
        for i in range(self.n):
            A = jnp.tensordot(params[i], self.M, axes=1)
            ops.append((A + A.T) / 2.0)
        return ops


class MultiDenseHermitianAnsatz:
    """Multi-matrix maximal-flexibility ansatz: M̂_i = (W_i + W_iᵀ)/2, any
    real-symmetric D×D operator per matrix (D·D params each).

    Cross-check for the bounded-degree MultiMonomialAnsatz: if both land at the
    same in-island moment, the monomial result is not an artifact of its
    restricted form (the Milestone-3 spurious-solution guard). Being fully
    flexible, it can equally represent (and so stress-test) a spurious, non-tracial
    minimum — which is why the symmetry penalties matter here.

    NOTE: this ansatz is full-rank, NOT bounded-degree, so the monomial
    exactness/cutoff guard in `train.solve_two_matrix` does not apply — word
    moments are exact on the truncated D-dim space by construction (no intermediate
    state is projected out). It therefore exposes no `degree` attribute (the guard
    defaults it to 1, which trivially passes).
    """

    def __init__(self, fock_ops: FockOps):
        self.ops = fock_ops
        self.n = fock_ops.n
        self.D = fock_ops.D
        self.n_params = self.n * self.D * self.D
        # free field M̂_i = â_i + â†_i = a_i + a_iᵀ (â† = âᵀ), already symmetric.
        self._free = jnp.stack(
            [fock_ops.a[i] + fock_ops.adag[i] for i in range(self.n)]
        )

    def init_params(self, key):
        return 0.01 * jax.random.normal(key, (self.n, self.D, self.D), dtype=jnp.float64)

    def params_for_free_field(self):
        return self._free  # W_i = a_i + a_iᵀ ⇒ (W_i + W_iᵀ)/2 = â_i + â†_i

    def build_operators(self, params):
        return [(params[i] + params[i].T) / 2.0 for i in range(self.n)]
