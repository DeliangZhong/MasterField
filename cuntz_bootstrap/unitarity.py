"""Unitarity loss L_unit = Σ_μ ||Û_μ Û_μ† - I||²_F."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX
from .master_operator import assemble_master_operator


def unitarity_loss(U_list: list[jnp.ndarray]) -> jnp.ndarray:
    """Sum of Frobenius-norm-squared deviations from unitarity.

    Returns a real-valued scalar (|x|² is real for complex x).
    """
    d = U_list[0].shape[0]
    dtype = U_list[0].dtype
    I = jnp.eye(d, dtype=dtype)
    total = jnp.zeros((), dtype=jnp.float64)
    for U in U_list:
        diff = U @ U.conj().T - I
        total = total + jnp.sum(jnp.abs(diff) ** 2)
    return total


def unitarity_loss_from_params(
    params: list[jnp.ndarray], fock: CuntzFockJAX
) -> jnp.ndarray:
    """Assemble operators from coefficient vectors and compute unitarity loss."""
    U_list = [assemble_master_operator(c, fock) for c in params]
    return unitarity_loss(U_list)
