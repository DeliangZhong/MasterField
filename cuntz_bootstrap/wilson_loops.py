"""Wilson loop evaluator on the Cuntz-Fock space.

W[C] = <Omega| U_{mu_1} U_{mu_2} ... U_{mu_k} |Omega>

for a closed loop C = (mu_1, ..., mu_k) with mu_j in {±1, ..., ±D}.

Conventions:
- mu > 0: apply U_{mu} (from U_list[mu - 1])
- mu < 0: apply U_{mu}† (adjoint of U_list[|mu| - 1])
- Empty loop → 1 (vacuum-to-vacuum amplitude of the identity).

Implementation: left-to-right matrix-vector sweep starting from
v_0 = |Omega>. The rightmost factor acts first per the operator ordering
U_{mu_1} U_{mu_2} ... U_{mu_k} |Omega> (standard QM right-to-left action).
Final W = <Omega|v_k> = v_k[0].
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX


def _apply_step(
    v: jnp.ndarray, U_list: list[jnp.ndarray], mu: int
) -> jnp.ndarray:
    if mu == 0:
        raise ValueError("Loop step cannot be mu=0")
    U = U_list[abs(mu) - 1]
    if mu < 0:
        U = U.conj().T
    return U @ v


def wilson_loop(
    U_list: list[jnp.ndarray],
    loop: tuple[int, ...],
    fock: CuntzFockJAX,
    D: int,
) -> jnp.ndarray:
    """Compute W[C] for the given loop word."""
    for mu in loop:
        if mu == 0:
            raise ValueError("Loop step cannot be mu=0")
        if abs(mu) > D:
            raise ValueError(f"|mu|={abs(mu)} exceeds D={D}")
    v = fock.vacuum_state()
    for mu in reversed(loop):
        v = _apply_step(v, U_list, mu)
    return v[0]


"""Note: `build_forward_link_ops` lives in `hermitian_operator.py` for v2.
Import it from there instead of this module."""
