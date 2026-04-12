"""Exponential-Hermitian parametrisation of master link operators (Phase 4 v2).

For lattice direction mu with n = 2D creation labels on the truncated Cuntz-
Fock space:

    Hhat_mu = Σ_{|w|<=L_poly} h_{mu,w} * (adag_{w_1} ... adag_{w_k})
                                       + conj(h_{mu,w}) * (a_{w_k} ... a_{w_1})
    Uhat_mu = expm(i * Hhat_mu)
    Uhat_{-mu} = Uhat_mu^dag       (orientation reversal)

Parameter layout per matrix: ONE complex vector `h` of length d_L = fock.dim,
with one coefficient per basis word (including the empty word, which gives the
identity contribution). For the empty word the imaginary part of h_empty
cancels against the h.c. contribution, so it contributes only a real scalar to
Hhat; the imaginary DOF is unused but harmless.

Unitarity of Uhat = expm(i * Hhat) is automatic for Hermitian Hhat. We build
Uhat via eigen-decomposition of Hhat: eigh is numerically stable, gives
machine-precision unitarity, and is differentiable in JAX.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg

from .fock import CuntzFockJAX


def direction_to_label(mu: int, D: int) -> int:
    """Map signed direction mu in {+-1, ..., +-D} to Cuntz label in {0, ..., 2D-1}.

    +mu -> 2*(mu-1),   -mu -> 2*(mu-1) + 1.
    """
    if mu == 0:
        raise ValueError("mu cannot be 0")
    abs_mu = abs(mu)
    if abs_mu > D:
        raise ValueError(f"|mu|={abs_mu} exceeds D={D}")
    sign_bit = 0 if mu > 0 else 1
    return 2 * (abs_mu - 1) + sign_bit


def init_hermitian_params(
    n_matrices: int,
    fock: CuntzFockJAX,
    seed: int = 0,
    scale: float = 0.01,
) -> list[jnp.ndarray]:
    """Initialise n_matrices complex coefficient vectors, each length d_L."""
    key = jax.random.PRNGKey(seed)
    size = fock.dim
    params: list[jnp.ndarray] = []
    for _ in range(n_matrices):
        key, k_re = jax.random.split(key)
        key, k_im = jax.random.split(key)
        re = jax.random.normal(k_re, (size,)) * scale
        im = jax.random.normal(k_im, (size,)) * scale
        params.append((re + 1j * im).astype(jnp.complex128))
    return params


def _build_word_operators(
    fock: CuntzFockJAX,
) -> tuple[list[jnp.ndarray], list[jnp.ndarray]]:
    """For each basis word w, return (creation_op, annihilation_op).

    create_ops[j] = adag_{w_1} @ ... @ adag_{w_k}  for w = fock.basis[j]
    annihilate_ops[j] = a_{w_k} @ ... @ a_{w_1}    (adjoint of create_ops[j])

    Both are the identity for the empty word (j = 0).
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


def assemble_hermitian(h: jnp.ndarray, fock: CuntzFockJAX) -> jnp.ndarray:
    """Ĥ = Σ_w (h_w · C_w + conj(h_w) · A_w). Hermitian by construction."""
    if h.shape != (fock.dim,):
        raise ValueError(f"h shape {h.shape} does not match ({fock.dim},)")
    create_ops, annihilate_ops = _build_word_operators(fock)
    H = jnp.zeros((fock.dim, fock.dim), dtype=jnp.complex128)
    for j in range(fock.dim):
        H = H + h[j] * create_ops[j] + jnp.conj(h[j]) * annihilate_ops[j]
    return H


def assemble_unitary(h: jnp.ndarray, fock: CuntzFockJAX) -> jnp.ndarray:
    """Û = expm(i · Ĥ) via jax.scipy.linalg.expm (Padé approximation).

    Padé is numerically stable AND differentiable through every parameter
    without the nearly-degenerate eigenvalue singularity that plagues
    the eigh-based expm backward pass. For Hermitian Ĥ, expm(iĤ) is unitary
    to machine precision (Padé error is ~1e-14 for norm(Ĥ) ~ O(1)).
    """
    H = assemble_hermitian(h, fock)
    H = 0.5 * (H + H.conj().T)  # symmetrise to suppress roundoff asymmetry
    return jax.scipy.linalg.expm(1j * H)


def build_forward_link_ops(
    params: list[jnp.ndarray], fock: CuntzFockJAX
) -> list[jnp.ndarray]:
    """Assemble all D forward link operators Û_μ from coefficient vectors."""
    return [assemble_unitary(h, fock) for h in params]
