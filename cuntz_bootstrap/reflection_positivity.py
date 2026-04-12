"""Reflection positivity (RP) residuals for the Cuntz-Fock master field.

Pick a reflection plane (default: axis mu = D, treated as "time"). For a set
of open paths p_1, ..., p_K that live entirely in the upper half (no steps
along -D), the reflection positivity condition requires the K x K overlap
matrix

    R_{ij} = <Omega| U_{theta(p_i)}^dag · U_{p_j} |Omega>

to be positive semidefinite, where theta(p) = reverse(p) with time-axis
signs flipped. R is a Gram matrix of the vectors v_{p_j} = U_{p_j} |Omega>
inner-producted with their reflected partners v_{theta(p_j)}.

We penalise negative eigenvalues of R:

    L_RP = sum_{k : lambda_k(R) < 0} lambda_k(R)^2

Equivalently (continuous everywhere, differentiable away from lambda = 0):

    L_RP = sum_k (relu(-lambda_k(R)))^2

At Uhat = I, R_{ij} = <Omega| Omega> = 1 for all (i, j), so R = (1..1)^T (1..1)
is rank-1 PSD, L_RP = 0.
"""
from __future__ import annotations

from itertools import product
from typing import Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.nn
import jax.numpy as jnp

from .fock import CuntzFockJAX


def reflect_path(p: tuple[int, ...], time_axis: int) -> tuple[int, ...]:
    """theta(p) = reverse(p) with time-axis signs flipped."""
    return tuple(
        (-mu if abs(mu) == time_axis else mu) for mu in reversed(p)
    )


def positive_half_open_paths(
    D: int, length_cutoff: int, time_axis: Optional[int] = None
) -> list[tuple[int, ...]]:
    """Enumerate open paths with no step in the -time_axis direction.

    Includes the empty path. Allows all signed non-time-axis steps and only
    the +time_axis step.
    """
    if time_axis is None:
        time_axis = D
    allowed: list[int] = []
    for axis in range(1, D + 1):
        if axis == time_axis:
            allowed.append(+axis)  # forward time only
        else:
            allowed.extend([+axis, -axis])

    paths: list[tuple[int, ...]] = [()]
    for k in range(1, length_cutoff + 1):
        for steps in product(allowed, repeat=k):
            paths.append(tuple(steps))
    return paths


def _path_vector(
    U_list: list[jnp.ndarray],
    p: tuple[int, ...],
    fock: CuntzFockJAX,
) -> jnp.ndarray:
    """v_p = U_{p[0]} U_{p[1]} ... U_{p[k-1]} |Omega>.

    Matches the operator ordering used by wilson_loop: the rightmost factor
    acts on |Omega> first, then factors to the left.
    """
    v = fock.vacuum_state()
    for mu in reversed(p):
        if mu == 0:
            raise ValueError("path step cannot be mu=0")
        U = U_list[abs(mu) - 1]
        if mu < 0:
            U = U.conj().T
        v = U @ v
    return v


def reflection_overlap_matrix(
    U_list: list[jnp.ndarray],
    paths: list[tuple[int, ...]],
    fock: CuntzFockJAX,
    D: int,
    time_axis: Optional[int] = None,
) -> jnp.ndarray:
    """Compute R_{ij} = <v_{theta(p_i)} | v_{p_j}>."""
    if time_axis is None:
        time_axis = D
    K = len(paths)
    vs = [_path_vector(U_list, p, fock) for p in paths]
    theta_vs = [
        _path_vector(U_list, reflect_path(p, time_axis), fock) for p in paths
    ]
    # R_{ij} = <theta_v_i, v_j> = sum_k conj(theta_v_i[k]) * v_j[k]
    V = jnp.stack(vs, axis=1)  # shape (dim, K)
    Theta_V = jnp.stack(theta_vs, axis=1)
    R = Theta_V.conj().T @ V  # shape (K, K)
    return R


def reflection_positivity_loss(
    U_list: list[jnp.ndarray],
    paths: list[tuple[int, ...]],
    fock: CuntzFockJAX,
    D: int,
    time_axis: Optional[int] = None,
) -> jnp.ndarray:
    """Sum of squared negative eigenvalues of the Hermitian part of R."""
    R = reflection_overlap_matrix(U_list, paths, fock, D, time_axis)
    R_herm = 0.5 * (R + R.conj().T)
    evals = jnp.linalg.eigvalsh(R_herm)
    neg_part = jax.nn.relu(-evals)
    return jnp.sum(neg_part ** 2)
