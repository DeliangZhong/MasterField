"""Operator ansätze for the master field — pluggable, all Hermitian.

Each ansatz exposes:
    init_params(key) -> pytree
    build_operators(params) -> list[jnp D×D Hermitian]   (one per matrix)

Milestone 2 compares three: MonomialAnsatz (here), DenseHermitianAnsatz,
and the amortized network (amortized.py). All guarantee M̂_i = M̂_i†, so the
vacuum state τ(·)=⟨Ω|·|Ω⟩ is automatically a positive state.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from matrix_master_field.fock_jax import FockOps  # noqa: E402


class MonomialAnsatz:
    """M̂ = Σ_k c_k H_k with H_k = (â†)^p â^q + h.c. (p ≤ q, p+q ≤ degree).

    Each H_k is real-symmetric (â† = âᵀ here), so M̂ is Hermitian for real c_k.
    Implemented for n_matrices == 1 (Milestone 2); multi-matrix is Milestone 3.
    """

    def __init__(self, fock_ops: FockOps, degree: int):
        if fock_ops.n != 1:
            raise NotImplementedError("MonomialAnsatz multi-matrix: Milestone 3")
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
            for p in range(q + 1):  # p <= q
                if p + q > degree:
                    continue
                mono = adag_pow[p] @ a_pow[q]  # (â†)^p â^q
                H = mono if p == q else mono + adag_pow[q] @ a_pow[p]  # + h.c.
                keys.append((p, q))
                mats.append(H)

        self.keys = keys
        self.H = jnp.asarray(np.stack(mats), dtype=jnp.float64)  # [P, D, D]
        self.n_params = len(keys)
        self._free_idx = keys.index((0, 1))  # the â + â† combination

    def init_params(self, key):
        return 0.01 * jax.random.normal(key, (self.n_params,), dtype=jnp.float64)

    def params_for_free_field(self):
        """Coefficients selecting M̂ = â + â† (the free/Gaussian master field)."""
        return jnp.zeros(self.n_params, dtype=jnp.float64).at[self._free_idx].set(1.0)

    def build_operators(self, params):
        return [jnp.tensordot(params, self.H, axes=1)]
