"""JAX-compatible Cuntz-Fock space.

Truncated at word length L_trunc. Labels 0..n_labels-1. For lattice YM with
D dimensions, use n_labels = 2D: forward direction +μ maps to label
2(μ-1) and backward -μ to 2(μ-1)+1.

Operators â_i and â†_i are stored as dense complex128 JAX arrays of shape
(dim, dim). They satisfy the Cuntz algebra in the INTERIOR (on words
whose length can still be incremented by â†_i without exceeding L_trunc):

    â_i â†_j = δ_{ij} · I          (interior)
    Σ_i â†_i â_i = I - |Ω⟩⟨Ω|       (interior, non-vacuum)

At the truncation boundary (|w| = L_trunc) â†_i acts as zero to avoid
leaving the truncated space.
"""
from __future__ import annotations

import itertools

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np


class CuntzFockJAX:
    """Truncated Cuntz-Fock space with JAX-array operators."""

    def __init__(self, n_labels: int, L_trunc: int) -> None:
        if n_labels < 1:
            raise ValueError("n_labels must be >= 1")
        if L_trunc < 0:
            raise ValueError("L_trunc must be >= 0")
        self.n_labels: int = int(n_labels)
        self.L_trunc: int = int(L_trunc)
        self._build_basis()
        self._build_operators()

    def _build_basis(self) -> None:
        basis: list[tuple[int, ...]] = [()]
        for k in range(1, self.L_trunc + 1):
            for w in itertools.product(range(self.n_labels), repeat=k):
                basis.append(tuple(w))
        self.basis: list[tuple[int, ...]] = basis
        self.dim: int = len(basis)
        self.basis_to_idx: dict[tuple[int, ...], int] = {
            w: i for i, w in enumerate(basis)
        }

    def _build_operators(self) -> None:
        adag_np = [
            np.zeros((self.dim, self.dim), dtype=np.complex128)
            for _ in range(self.n_labels)
        ]
        a_np = [
            np.zeros((self.dim, self.dim), dtype=np.complex128)
            for _ in range(self.n_labels)
        ]
        for j, w in enumerate(self.basis):
            # â†_i |w⟩ = |i, w⟩ if len(w) < L_trunc, else 0
            if len(w) < self.L_trunc:
                for i in range(self.n_labels):
                    new_w = (i,) + w
                    new_j = self.basis_to_idx[new_w]
                    adag_np[i][new_j, j] = 1.0
            # â_i |w⟩ = |w[1:]⟩ if w[0] == i, else 0
            if len(w) >= 1:
                i0 = w[0]
                new_w = w[1:]
                new_j = self.basis_to_idx[new_w]
                a_np[i0][new_j, j] = 1.0
        self.adag: list[jnp.ndarray] = [jnp.asarray(m) for m in adag_np]
        self.a: list[jnp.ndarray] = [jnp.asarray(m) for m in a_np]

    def vacuum_state(self) -> jnp.ndarray:
        v = np.zeros(self.dim, dtype=np.complex128)
        v[0] = 1.0
        return jnp.asarray(v)

    def vacuum_projector(self) -> jnp.ndarray:
        P = np.zeros((self.dim, self.dim), dtype=np.complex128)
        P[0, 0] = 1.0
        return jnp.asarray(P)
