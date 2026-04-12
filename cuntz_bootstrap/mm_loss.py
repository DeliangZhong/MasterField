"""Full training loss: unitarity + Makeenko-Migdal + (optional) supervised anchor.

Reuses master_field/lattice.LoopSystem for MM equation index tables. The
Wilson loops W[C] are computed from the Cuntz-Fock coefficient parameters,
making the entire loss differentiable w.r.t. the coefficients.

MM equation convention (candidate D from master_field/mm_equations.py):

    (1/λ) Σ_{k ∈ lhs} w[k]  =  c_self · w[loop]  +  Σ_{(i,j) ∈ splits} w[i] · w[j]

with c_self = 2. The residual per equation is LHS − RHS; the total MM loss
is the sum of squared residuals over all equations of loops up to L_max.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Make master_field/ importable without __init__ mutation.
_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import LoopSystem, build_loop_system  # noqa: E402
from mm_equations import gw_w_plus  # noqa: E402

from .fock import CuntzFockJAX
from .master_operator import assemble_master_operator
from .unitarity import unitarity_loss
from .wilson_loops import wilson_loop


def _load_loop_system(D: int, L_max: int) -> LoopSystem:
    """Thin wrapper so test code can request a LoopSystem without duplicating the sys.path dance."""
    return build_loop_system(D=D, L_max=L_max, mm_form="D")


def _compute_all_wilson_loops(
    U_list: list[jnp.ndarray],
    loop_sys: LoopSystem,
    fock: CuntzFockJAX,
    D: int,
) -> jnp.ndarray:
    """Return real-part Wilson loop vector indexed by loop_sys.loops."""
    vals = [jnp.real(wilson_loop(U_list, C, fock, D)) for C in loop_sys.loops]
    return jnp.stack(vals)


def _default_area_law_target(loop_sys: LoopSystem, lam: float) -> jnp.ndarray:
    """Default supervised target: GW lattice area law W[C] = w_+^{Area}. D=2 only."""
    if loop_sys.areas is None:
        raise ValueError("Supervised target requires loop_sys.areas (D=2 only)")
    w_plus = gw_w_plus(lam)
    target = jnp.zeros(loop_sys.K, dtype=jnp.float64)
    for i in range(loop_sys.K):
        area = loop_sys.areas.get(i, 0)
        target = target.at[i].set(w_plus ** area)
    return target


def make_cuntz_mm_loss_fn(
    loop_sys: LoopSystem,
    fock: CuntzFockJAX,
    D: int,
    w_unit: float = 1.0,
    w_mm: float = 1.0,
    w_sup: float = 0.0,
    sup_target_fn: Optional[Callable[[float], jnp.ndarray]] = None,
    return_components: bool = False,
) -> Callable:
    """Build a differentiable loss function L(params, lam) → scalar.

    Parameters
    ----------
    loop_sys : LoopSystem from master_field.lattice
    fock : CuntzFockJAX
    D : spacetime dimension
    w_unit, w_mm, w_sup : loss-component weights
    sup_target_fn : optional callable λ → target-W vector of length loop_sys.K.
        If w_sup > 0 and sup_target_fn is None, defaults to the D=2 area law
        W[C] = w_+^{Area(C)} with w_+ = gw_w_plus(λ).
    return_components : if True, return (total, L_unit, L_mm, L_sup) tuple.
    """
    equations = loop_sys.mm_equations

    def loss_fn(params: list[jnp.ndarray], lam: float):
        U_list = [assemble_master_operator(c, fock) for c in params]
        L_unit = unitarity_loss(U_list)
        W = _compute_all_wilson_loops(U_list, loop_sys, fock, D)

        L_mm = jnp.zeros((), dtype=jnp.float64)
        for eq in equations:
            if eq.lhs_loop_indices:
                lhs = jnp.sum(W[jnp.array(eq.lhs_loop_indices)]) / lam
            else:
                lhs = jnp.zeros((), dtype=jnp.float64)
            rhs = eq.rhs_self_coeff * W[eq.loop_idx]
            for (i, j) in eq.rhs_split_pairs:
                rhs = rhs + W[i] * W[j]
            residual = lhs - rhs
            L_mm = L_mm + residual ** 2

        if w_sup > 0.0:
            target = (
                sup_target_fn(lam)
                if sup_target_fn is not None
                else _default_area_law_target(loop_sys, lam)
            )
            L_sup = jnp.sum((W - target) ** 2)
        else:
            L_sup = jnp.zeros((), dtype=jnp.float64)

        total = w_unit * L_unit + w_mm * L_mm + w_sup * L_sup
        if return_components:
            return total, L_unit, L_mm, L_sup
        return total

    return loss_fn
