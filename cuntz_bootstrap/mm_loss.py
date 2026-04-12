"""Makeenko-Migdal loop-equation residuals (Phase 4 v2).

Reuses master_field/lattice.LoopSystem for the MM equation index tables.
Wilson loops are computed from a pre-assembled list of unitary link operators
via the evaluator from wilson_loops.py, so the residuals are differentiable
w.r.t. the underlying parameters (Hermitian generators h).

MM equation convention (Kazakov-Zheng candidate D):

    (1/lambda) * sum_{k in lhs} w[k]  =  c_self * w[loop] + sum_{(i,j)} w[i] * w[j]

with c_self = 2. The residual per equation is LHS - RHS. total_loss.py sums
the squared residuals into L_MM.

This module exports:

- `_load_loop_system(D, L_max)` — thin wrapper around
  `master_field.lattice.build_loop_system`, hiding the sys.path dance.
- `compute_all_wilson_loops(U_list, loop_sys, fock, D)` — evaluate Re[W[C]]
  for every canonical loop in loop_sys.
- `default_area_law_target(loop_sys, lam)` — GW area-law target (D=2 only),
  used when the optional supervised anchor is enabled.
- `make_mm_residuals_fn(loop_sys, fock, D)` — returns a closure
  `residuals(U_list, lam) -> jnp.ndarray of shape (n_equations,)`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import LoopSystem, build_loop_system  # noqa: E402
from mm_equations import gw_w_plus  # noqa: E402

from .fock import CuntzFockJAX
from .wilson_loops import wilson_loop


def _load_loop_system(D: int, L_max: int) -> LoopSystem:
    """Wrapper around master_field.lattice.build_loop_system."""
    return build_loop_system(D=D, L_max=L_max, mm_form="D")


def compute_all_wilson_loops(
    U_list: list[jnp.ndarray],
    loop_sys: LoopSystem,
    fock: CuntzFockJAX,
    D: int,
) -> jnp.ndarray:
    """Real-part Wilson-loop vector indexed by loop_sys.loops.

    Wilson loops are real in SU(N) at N=∞ for the physical master field;
    taking Re(·) suppresses numerical roundoff in the imaginary part.
    """
    vals = [jnp.real(wilson_loop(U_list, C, fock, D)) for C in loop_sys.loops]
    return jnp.stack(vals)


def default_area_law_target(loop_sys: LoopSystem, lam: float) -> jnp.ndarray:
    """GW lattice area-law target W[C] = w_+^{Area(C)} (D=2 only)."""
    if loop_sys.areas is None:
        raise ValueError("Supervised target requires loop_sys.areas (D=2 only)")
    w_plus = gw_w_plus(lam)
    target = jnp.zeros(loop_sys.K, dtype=jnp.float64)
    for i in range(loop_sys.K):
        area = loop_sys.areas.get(i, 0)
        target = target.at[i].set(w_plus ** area)
    return target


def make_mm_residuals_fn(
    loop_sys: LoopSystem,
    fock: CuntzFockJAX,
    D: int,
) -> Callable[[list[jnp.ndarray], float], jnp.ndarray]:
    """Factory returning a closure residuals(U_list, lam) -> (n_eqs,) array.

    Each entry is LHS − RHS of one Makeenko-Migdal equation. L_MM is the sum
    of squares. Keeping per-equation residuals exposed makes it easy to
    inspect which equations are violated worst during optimisation.
    """
    equations = loop_sys.mm_equations

    def residuals_fn(U_list: list[jnp.ndarray], lam: float) -> jnp.ndarray:
        W = compute_all_wilson_loops(U_list, loop_sys, fock, D)
        res_list: list[jnp.ndarray] = []
        for eq in equations:
            if eq.lhs_loop_indices:
                lhs = jnp.sum(W[jnp.array(eq.lhs_loop_indices)]) / lam
            else:
                lhs = jnp.zeros((), dtype=jnp.float64)
            rhs = eq.rhs_self_coeff * W[eq.loop_idx]
            for (i, j) in eq.rhs_split_pairs:
                rhs = rhs + W[i] * W[j]
            res_list.append(lhs - rhs)
        return jnp.stack(res_list)

    return residuals_fn
