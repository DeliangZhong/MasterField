"""observables.py — Gauge-invariant observables on the TEK master field.

All observables take the list of link matrices U = [U_1, ..., U_D] (already
built via `tek.build_link_matrices`) and return JAX scalars.

Plaquette observable and Polyakov loop are fully supported.

`wilson_loop_rectangular` requires the rectangular twist phase f(R,T) which is
GATED on transcribing the exact formula from:
    - arXiv:1708.00841 (García Pérez–González-Arroyo–Okawa 2017), or
    - Phys. Rev. D 27 (1983) 2397 (González-Arroyo–Okawa, eq. 3.5).
Calling it before transcription raises NotImplementedError.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def wilson_loop_plaquette(
    U: list[jnp.ndarray],
    mu: int,
    nu: int,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Single-plaquette Wilson loop:
        W[□]_{μν} = Re[z_μν · Tr(U_μ U_ν U_μ† U_ν†)] / N

    Real-valued.
    """
    if mu == nu:
        raise ValueError(f"mu and nu must differ; got mu=nu={mu}")
    N = U[0].shape[0]
    Umu = U[mu]
    Unu = U[nu]
    plaq = Umu @ Unu @ jnp.conj(Umu.T) @ jnp.conj(Unu.T)
    return jnp.real(z[mu, nu] * jnp.trace(plaq)) / N


def polyakov_loop(U: list[jnp.ndarray], mu: int, L_t: int = 1) -> jnp.ndarray:
    """Polyakov loop in direction μ:  P_μ = Tr(U_μ^{L_t}) / N.

    For a center-symmetric configuration, P_μ = 0 (confined). Nonzero ⟨P⟩
    signals center-symmetry breaking.
    """
    N = U[0].shape[0]
    Umu_Lt = jnp.linalg.matrix_power(U[mu], L_t)
    return jnp.trace(Umu_Lt) / N


def eigenvalue_phases(U: list[jnp.ndarray], mu: int) -> jnp.ndarray:
    """Eigenvalue phases of U_μ (angles in (−π, π])."""
    eigs = jnp.linalg.eigvals(U[mu])
    return jnp.angle(eigs)


def eigenvalue_density(
    U: list[jnp.ndarray],
    mu: int,
    n_bins: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Histogram of eigenvalue phases on (−π, π].

    Returns (counts_normalized, bin_edges). At the center-symmetric saddle,
    counts_normalized ≈ 1/(2π) (uniform on the circle).
    """
    phases = eigenvalue_phases(U, mu)
    counts, edges = jnp.histogram(phases, bins=n_bins, range=(-jnp.pi, jnp.pi))
    N = U[0].shape[0]
    return counts / N, edges


def center_symmetry_order(U: list[jnp.ndarray]) -> jnp.ndarray:
    """Σ_μ |Tr(U_μ)/N|² — a scalar measuring center-symmetry breaking.

    0 if center symmetry is unbroken, positive otherwise.
    """
    N = U[0].shape[0]
    traces = jnp.stack([jnp.trace(Umu) / N for Umu in U])
    return jnp.sum(jnp.abs(traces) ** 2)


def wilson_loop_rectangular(
    U: list[jnp.ndarray],
    z: jnp.ndarray,
    R: int,
    T: int,
    mu: int = 0,
    nu: int = 1,
) -> jnp.ndarray:
    """Rectangular R×T Wilson loop in the (μ,ν) plane.

    On a single-site TEK lattice, U_μ shifts by 1 lattice unit in direction μ,
    so a naive single-site form is:
        W_naive = Tr(U_μ^R U_ν^T U_μ^{-R} U_ν^{-T}) / N

    With the twist, the correct form picks up a twist phase f(R, T, z_μν) from
    the non-commutativity of the twist eaters. The formula is in:
        - García Pérez–González-Arroyo–Okawa, arXiv:1708.00841 (2017)
        - González-Arroyo–Okawa, Phys. Rev. D 27 (1983) 2397, eq. (3.5)

    DO NOT GUESS. Until the exact formula is transcribed and tested, this
    function raises NotImplementedError. The plaquette case R=T=1 is handled
    correctly by `wilson_loop_plaquette`.
    """
    raise NotImplementedError(
        f"wilson_loop_rectangular(R={R}, T={T}): twist phase f(R,T) not yet "
        "transcribed from arXiv:1708.00841 or PRD 27 (1983) eq. (3.5). "
        "Use wilson_loop_plaquette for R=T=1."
    )


def creutz_ratio(
    U: list[jnp.ndarray],
    z: jnp.ndarray,
    R: int,
    T: int,
    mu: int = 0,
    nu: int = 1,
) -> jnp.ndarray:
    """Creutz ratio  χ(R,T) = −log[ W(R,T)·W(R−1,T−1) / (W(R,T−1)·W(R−1,T)) ].

    At large R, T the ratio → σ (string tension in lattice units). Gated on
    `wilson_loop_rectangular` being implemented.
    """

    def W(r: int, t: int) -> jnp.ndarray:
        return wilson_loop_rectangular(U, z, r, t, mu, nu)

    num = W(R, T) * W(R - 1, T - 1)
    den = W(R, T - 1) * W(R - 1, T)
    return -jnp.log(num / den)
