"""Polynomial parametrisation of master link operators on Cuntz-Fock space.

    Û_μ = Σ_{|w|<=L} c^{(+)}_{μ,w} · (â†_{w_1} … â†_{w_k})
        + Σ_{|v|>=1, |v|<=L} c^{(-)}_{μ,v} · (â_{v_k} … â_{v_1})

Flat coefficient layout per matrix (length 2*d_L - 1, complex128):
    [c^{(+)}_{empty},          # index 0 → identity
     c^{(+)}_{basis[1]}, ...,  # creation for non-trivial words basis[1..d-1]
     c^{(-)}_{basis[1]}, ...]  # annihilation for words basis[1..d-1]

Creation op for word w = (w_1, ..., w_k):   C_w = â†_{w_1} @ â†_{w_2} @ ... @ â†_{w_k}
  applied to |Ω⟩ yields |w_1, ..., w_k⟩.
Annihilation op for word v = (v_1, ..., v_k): A_v = â_{v_k} @ â_{v_{k-1}} @ ... @ â_{v_1}
  applied to |v⟩ yields |Ω⟩ and to any other basis state yields 0 (mod truncation).

Orientation reversal: Û_{-μ} = Û_μ†. Store only D matrices; adjoint computed
on demand by callers (see wilson_loops._apply_step).
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX


def direction_to_label(mu: int, D: int) -> int:
    """Map signed direction μ ∈ {±1, ..., ±D} to Cuntz label ∈ {0, ..., 2D-1}."""
    if mu == 0:
        raise ValueError("mu cannot be 0")
    abs_mu = abs(mu)
    if abs_mu > D:
        raise ValueError(f"|mu|={abs_mu} exceeds D={D}")
    sign_bit = 0 if mu > 0 else 1
    return 2 * (abs_mu - 1) + sign_bit


def init_master_operator_params(
    n_matrices: int,
    fock: CuntzFockJAX,
    seed: int = 0,
    scale: float = 0.01,
) -> list[jnp.ndarray]:
    """Initialise n_matrices complex coefficient vectors, each length 2*d_L - 1."""
    key = jax.random.PRNGKey(seed)
    size = 2 * fock.dim - 1
    params: list[jnp.ndarray] = []
    for _ in range(n_matrices):
        key, k_re = jax.random.split(key)
        key, k_im = jax.random.split(key)
        re = jax.random.normal(k_re, (size,)) * scale
        im = jax.random.normal(k_im, (size,)) * scale
        params.append((re + 1j * im).astype(jnp.complex128))
    return params


def _build_basis_operators(
    fock: CuntzFockJAX,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """Return (creation_ops, annihilation_ops) aligned with fock.basis.

    create_ops[j] = â†_{w_1} @ ... @ â†_{w_k} for w = fock.basis[j]
                    (identity for j = 0, i.e. empty word)
    annihilate_ops[j] = â_{w_k} @ ... @ â_{w_1} for w = fock.basis[j]
                       (identity sentinel for j = 0; never used because
                        annihilation sum starts at j = 1)
    """
    eye = jnp.eye(fock.dim, dtype=jnp.complex128)
    create_ops: list[jnp.ndarray] = []
    annihilate_ops: list[jnp.ndarray] = []
    for w in fock.basis:
        if len(w) == 0:
            create_ops.append(eye)
            annihilate_ops.append(eye)
            continue
        c_op = fock.adag[w[0]]
        for idx in w[1:]:
            c_op = c_op @ fock.adag[idx]
        create_ops.append(c_op)
        a_op = fock.a[w[-1]]
        for idx in reversed(w[:-1]):
            a_op = a_op @ fock.a[idx]
        annihilate_ops.append(a_op)
    return create_ops, annihilate_ops


def assemble_master_operator(c: jnp.ndarray, fock: CuntzFockJAX) -> jnp.ndarray:
    """Assemble a d_L × d_L master link matrix from flat coefficient vector c."""
    d = fock.dim
    if c.shape != (2 * d - 1,):
        raise ValueError(f"c shape {c.shape} does not match expected ({2 * d - 1},)")
    create_ops, annihilate_ops = _build_basis_operators(fock)
    U = jnp.zeros((d, d), dtype=jnp.complex128)
    # Creation: indices 0..d-1
    for j in range(d):
        U = U + c[j] * create_ops[j]
    # Annihilation: indices d..2d-2, mapped to basis words 1..d-1
    for j in range(1, d):
        U = U + c[d + j - 1] * annihilate_ops[j]
    return U
