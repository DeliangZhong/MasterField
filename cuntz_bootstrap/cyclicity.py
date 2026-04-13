"""Cyclicity / traciality residuals for the Cuntz-Fock master field.

At N = infinity, the Wilson-loop "trace" W[C] = <Omega| U_{mu_1} ... U_{mu_k}
|Omega> is invariant under cyclic rotations of the loop word C. This is the
planar-trace property, not an automatic consequence of Hilbert-space
positivity of the Cuntz-Fock construction. It must be imposed as a constraint.

For each loop C in a user-supplied test set, for each cyclic rotation C_i,
this module contributes |W[C_i] - W[C_0]|^2 to the loss.

The default test set is all canonical loops in a LoopSystem with length
at least `min_length` (default 3, since length-2 loops {(mu, -mu)} are
trivially cyclic).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX
from .wilson_loops import wilson_loop


def build_cyclicity_test_loops(
    loop_sys, min_length: int = 3
) -> list[tuple[int, ...]]:
    """Select loops of length >= min_length from a LoopSystem."""
    return [C for C in loop_sys.loops if len(C) >= min_length]


def _cyclic_rotation(C: tuple[int, ...], i: int) -> tuple[int, ...]:
    """Rotate C by i positions: (C[i], C[i+1], ..., C[-1], C[0], ..., C[i-1])."""
    i = i % len(C)
    return C[i:] + C[:i]


def cyclicity_loss(
    U_list: list[jnp.ndarray],
    test_loops: list[tuple[int, ...]],
    fock: CuntzFockJAX,
    D: int,
) -> jnp.ndarray:
    """Sum over test loops of sum_{i=1..|C|-1} |W[C_i] - W[C_0]|^2."""
    total = jnp.zeros((), dtype=jnp.float64)
    for C in test_loops:
        if len(C) < 2:
            continue
        W0 = wilson_loop(U_list, C, fock, D)
        for i in range(1, len(C)):
            Ci = _cyclic_rotation(C, i)
            Wi = wilson_loop(U_list, Ci, fock, D)
            total = total + jnp.abs(Wi - W0) ** 2
    return total


def cyclicity_loss_matfree(
    params: list[jnp.ndarray],
    test_loops: list[tuple[int, ...]],
    fock: CuntzFockJAX,
    word_pairs,
    D: int,
    order: int = 25,
) -> jnp.ndarray:
    """Matrix-free variant of cyclicity_loss.

    Same loss structure but uses `wilson_loop_matfree` (Taylor-series
    expm-v) instead of dense U@v. Signature takes the raw parameter list
    (one h vector per direction) rather than a list of pre-assembled U's.
    """
    from .wilson_loops import wilson_loop_matfree  # lazy import

    total = jnp.zeros((), dtype=jnp.float64)
    for C in test_loops:
        if len(C) < 2:
            continue
        W0 = wilson_loop_matfree(params, C, fock, word_pairs, D, order)
        for i in range(1, len(C)):
            Ci = _cyclic_rotation(C, i)
            Wi = wilson_loop_matfree(params, Ci, fock, word_pairs, D, order)
            total = total + jnp.abs(Wi - W0) ** 2
    return total
