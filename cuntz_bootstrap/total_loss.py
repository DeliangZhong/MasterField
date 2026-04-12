"""Total-loss factory for the Phase 4 Cuntz-Fock bootstrap.

Composes the four component losses (Makeenko-Migdal, cyclicity, reflection
positivity, lattice symmetry) plus an optional supervised anchor into a
single differentiable closure:

    L(params, lam) = w_MM * L_MM + w_cyc * L_cyc + w_RP * L_RP + w_sym * L_sym
                    [ + w_sup * L_sup ]

The optimiser calls L(params, lam) to get a scalar; debugging code calls with
return_components=True to inspect the pieces.
"""
from __future__ import annotations

from typing import Callable, NamedTuple, Optional

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .cyclicity import build_cyclicity_test_loops, cyclicity_loss
from .fock import CuntzFockJAX
from .hermitian_operator import build_forward_link_ops
from .lattice_symmetry import b_d_generators, lattice_symmetry_loss
from .mm_loss import (
    compute_all_wilson_loops,
    default_area_law_target,
    make_mm_residuals_fn,
)
from .reflection_positivity import (
    positive_half_open_paths,
    reflection_positivity_loss,
)


class LossComponents(NamedTuple):
    total: jnp.ndarray
    L_MM: jnp.ndarray
    L_cyc: jnp.ndarray
    L_RP: jnp.ndarray
    L_sym: jnp.ndarray
    L_sup: jnp.ndarray


def make_total_loss_fn(
    loop_sys,
    fock: CuntzFockJAX,
    D: int,
    weights: Optional[dict] = None,
    cyc_test_loops: Optional[list[tuple[int, ...]]] = None,
    sym_generators: Optional[list[Callable]] = None,
    rp_paths: Optional[list[tuple[int, ...]]] = None,
    rp_time_axis: Optional[int] = None,
    sup_target_fn: Optional[Callable[[float], jnp.ndarray]] = None,
    return_components: bool = False,
) -> Callable:
    """Return a closure loss_fn(params, lam) -> scalar (or LossComponents).

    weights keys: 'mm', 'cyc', 'rp', 'sym', 'sup'. Missing keys default to 0.
    Defaults for the test sets / generators are built from loop_sys and D.
    """
    w = dict(weights) if weights is not None else {}
    w_MM = float(w.get("mm", 0.0))
    w_cyc = float(w.get("cyc", 0.0))
    w_RP = float(w.get("rp", 0.0))
    w_sym = float(w.get("sym", 0.0))
    w_sup = float(w.get("sup", 0.0))

    if cyc_test_loops is None:
        cyc_test_loops = build_cyclicity_test_loops(loop_sys, min_length=3)
    if sym_generators is None:
        sym_generators = b_d_generators(D)
    if rp_paths is None:
        rp_paths = positive_half_open_paths(
            D=D, length_cutoff=2, time_axis=rp_time_axis if rp_time_axis else D
        )
    if rp_time_axis is None:
        rp_time_axis = D

    residuals_fn = make_mm_residuals_fn(loop_sys=loop_sys, fock=fock, D=D)

    def loss_fn(params: list[jnp.ndarray], lam: float):
        U_list = build_forward_link_ops(params, fock)

        L_MM = jnp.zeros((), dtype=jnp.float64)
        if w_MM > 0.0:
            L_MM = jnp.sum(residuals_fn(U_list, lam) ** 2)

        L_cyc = jnp.zeros((), dtype=jnp.float64)
        if w_cyc > 0.0:
            L_cyc = cyclicity_loss(U_list, cyc_test_loops, fock, D)

        L_RP = jnp.zeros((), dtype=jnp.float64)
        if w_RP > 0.0:
            L_RP = reflection_positivity_loss(
                U_list, rp_paths, fock, D, time_axis=rp_time_axis
            )

        L_sym = jnp.zeros((), dtype=jnp.float64)
        if w_sym > 0.0:
            L_sym = lattice_symmetry_loss(
                U_list, cyc_test_loops, sym_generators, fock, D
            )

        L_sup = jnp.zeros((), dtype=jnp.float64)
        if w_sup > 0.0:
            target = (
                sup_target_fn(lam)
                if sup_target_fn is not None
                else default_area_law_target(loop_sys, lam)
            )
            W = compute_all_wilson_loops(U_list, loop_sys, fock, D)
            L_sup = jnp.sum((W - target) ** 2)

        total = (
            w_MM * L_MM
            + w_cyc * L_cyc
            + w_RP * L_RP
            + w_sym * L_sym
            + w_sup * L_sup
        )
        if return_components:
            return LossComponents(
                total=total, L_MM=L_MM, L_cyc=L_cyc,
                L_RP=L_RP, L_sym=L_sym, L_sup=L_sup,
            )
        return total

    return loss_fn
