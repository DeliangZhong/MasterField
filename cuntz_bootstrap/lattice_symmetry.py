"""Lattice symmetry residuals for the Cuntz-Fock master field.

The hyperoctahedral group B_D acts on signed directions {+-1, ..., +-D} by
axis permutations and sign flips. The physical master field satisfies
W[sigma(C)] = W[C] for every sigma in B_D. Like cyclicity, this is NOT
automatic in the Cuntz-Fock construction — the coefficients must encode
the symmetry.

This module provides a small set of GENERATORS of B_D (adjacent axis swaps
and individual sign flips) that together span the group. Iterating over
generators alone gives a lightweight symmetry loss; to enforce the full
group one can compose generators or use the full element list.

For a closed loop, axis reversal (mu -> -mu everywhere) equals reading the
loop in reverse direction, which by adjointness gives W[reversed] = conj(W).
For SU(N) at N = infinity and real Wilson-action master field, W is real, so
adjoint symmetry adds no new constraint beyond cyclicity + axis-permutation.
We therefore focus on axis-permutation and per-axis sign-flip symmetries as
the primary content of L_sym.
"""
from __future__ import annotations

from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX
from .wilson_loops import wilson_loop


def _apply_axis_flip(axis: int) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """Return sigma(mu) = -mu if |mu|==axis else mu."""

    def flip(word: tuple[int, ...]) -> tuple[int, ...]:
        return tuple((-mu if abs(mu) == axis else mu) for mu in word)

    return flip


def _apply_axis_swap(
    i: int, j: int
) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """Return sigma that swaps axes i and j (preserves sign)."""

    def swap(word: tuple[int, ...]) -> tuple[int, ...]:
        out = []
        for mu in word:
            abs_mu = abs(mu)
            if abs_mu == i:
                new_abs = j
            elif abs_mu == j:
                new_abs = i
            else:
                new_abs = abs_mu
            out.append(new_abs if mu > 0 else -new_abs)
        return tuple(out)

    return swap


def b_d_generators(D: int) -> list[Callable[[tuple[int, ...]], tuple[int, ...]]]:
    """Return a list of B_D generators (axis flips + adjacent swaps).

    For D=2: 2 flips + 1 swap = 3 generators. Compositions generate B_2 (order 8).
    For D=4: 4 flips + 3 adjacent swaps = 7 generators, generating B_4 (order 384).
    """
    if D < 1:
        raise ValueError("D must be >= 1")
    gens: list[Callable[[tuple[int, ...]], tuple[int, ...]]] = []
    # Sign flips on each axis
    for axis in range(1, D + 1):
        gens.append(_apply_axis_flip(axis))
    # Adjacent axis swaps
    for i in range(1, D):
        gens.append(_apply_axis_swap(i, i + 1))
    return gens


def lattice_symmetry_loss(
    U_list: list[jnp.ndarray],
    test_loops: list[tuple[int, ...]],
    symmetries: list[Callable[[tuple[int, ...]], tuple[int, ...]]],
    fock: CuntzFockJAX,
    D: int,
) -> jnp.ndarray:
    """L_sym = sum_{C, sigma} |W[sigma(C)] - W[C]|^2."""
    total = jnp.zeros((), dtype=jnp.float64)
    for C in test_loops:
        if not C:
            continue
        W0 = wilson_loop(U_list, C, fock, D)
        for sigma in symmetries:
            C_sym = sigma(C)
            W_sym = wilson_loop(U_list, C_sym, fock, D)
            total = total + jnp.abs(W_sym - W0) ** 2
    return total
