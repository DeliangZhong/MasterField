"""Truncation diagnostics for the Cuntz-Fock bootstrap.

Monitor:
- boundary_norm(v, fock):   |components of v at Fock basis words of maximal length|².
  If ~ 1e-2 or larger, Fock truncation is too small for the computation.
- interior_unitarity(U, fock):   ||U U† - I||_F restricted to words of length < L_trunc.
  For exp-Hermitian Û, this should be machine precision (~1e-10).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX


def boundary_norm(v: jnp.ndarray, fock: CuntzFockJAX) -> float:
    """Sum |v_w|² over basis words of maximal length L_trunc.

    If this is >> 0, the Fock truncation is too small to contain the state.
    """
    L = fock.L_trunc
    boundary_idx = [i for i, w in enumerate(fock.basis) if len(w) == L]
    if not boundary_idx:
        return 0.0
    idx_arr = jnp.asarray(boundary_idx)
    return float(jnp.sum(jnp.abs(v[idx_arr]) ** 2))


def interior_unitarity(U: jnp.ndarray, fock: CuntzFockJAX) -> float:
    """||U U† - I||_F restricted to basis words of length < L_trunc.

    For Padé-computed expm(iĤ), this should be ~1e-10 (machine precision).
    If larger, something is wrong with the Hermitian assembly or expm.
    """
    L = fock.L_trunc
    interior_idx = jnp.asarray(
        [i for i, w in enumerate(fock.basis) if len(w) < L]
    )
    I = jnp.eye(fock.dim, dtype=U.dtype)
    diff = U @ U.conj().T - I
    sub = diff[jnp.ix_(interior_idx, interior_idx)]
    return float(jnp.sqrt(jnp.sum(jnp.abs(sub) ** 2)))


def probe_truncation_adequate(
    U_list: list[jnp.ndarray],
    fock: CuntzFockJAX,
    threshold: float = 1e-2,
) -> dict:
    """Check U_μ |Ω⟩ boundary-norm for every direction.

    Returns a dict {mu -> boundary_norm}. If any value > threshold, the
    current L_trunc is too small for this Û.
    """
    vac = fock.vacuum_state()
    out = {}
    for mu, U in enumerate(U_list, start=1):
        v = U @ vac
        out[mu] = boundary_norm(v, fock)
    return out
