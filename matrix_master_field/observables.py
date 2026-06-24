"""Operator observables of the two-matrix master field (Milestone 3, Task 6).

Everything is read directly off the optimized Cuntz-Fock operators M̂_i:
  - ρ(λ): eigenvalue density of a single Hermitian M̂_i in the vacuum state
    τ = ⟨Ω|·|Ω⟩ (weights |⟨Ω|v_i⟩|²); reduces to the Wigner semicircle at g=0.
  - ⟨tr[M̂0,M̂1]²⟩: the non-commutativity order parameter (≤ 0, → 0 as g→0).
  - complex spectrum of the non-normal M̂0+iM̂1: a Brown-measure PROXY (see caveat).

These are the observables a bound (SDP) cannot give — the point of constructing
the master field as an operator rather than a list of moments.
"""

import numpy as np


def vacuum_spectral_measure(M_hat):
    """Eigenvalue spectral measure of a Hermitian M̂ in the vacuum state.

    Returns (eigs, weights) with weights_i = |⟨Ω|v_i⟩|² (Ω = basis state 0). By
    construction Σ_i weights_i · λ_i^k = ⟨Ω|M̂^k|Ω⟩ — i.e. (eigs, weights) is the
    eigenvalue density ρ(λ) of M̂ as seen by the master-field trace, and its
    moments are exactly the vacuum moments we optimize/validate.
    """
    M = np.asarray(M_hat, dtype=float)
    M = 0.5 * (M + M.T)  # symmetrize against round-off before eigh
    eigs, vecs = np.linalg.eigh(M)
    weights = vecs[0, :] ** 2  # |⟨Ω|v_i⟩|², Ω = e_0; Σ weights = ||Ω||² = 1
    return eigs, weights


def density_histogram(eigs, weights, bins=60, span=None):
    """Binned eigenvalue density (centers, ρ) from a spectral measure; ∫ρ dλ = 1."""
    eigs = np.asarray(eigs)
    weights = np.asarray(weights)
    if span is None:
        lo, hi = float(eigs.min()), float(eigs.max())
        pad = 0.05 * (hi - lo + 1e-12)
        span = (lo - pad, hi + pad)
    rho, edges = np.histogram(eigs, bins=bins, range=span, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, rho


def commutator_sq_expectation(ops):
    """⟨tr[M̂0,M̂1]²⟩ = 2(⟨Ω|M0M1M0M1|Ω⟩ − ⟨Ω|M0²M1²|Ω⟩).

    [M0,M1] is anti-Hermitian so tr[M0,M1]² ≤ 0; → 0 as g→0 (commuting free
    semicirculars). Identity: tr[A,B]² = 2(tr ABAB − tr A²B²) by cyclicity.
    """
    from matrix_master_field.fock_jax import word_moment

    abab = float(word_moment(ops, (0, 1, 0, 1)))
    aabb = float(word_moment(ops, (0, 0, 1, 1)))
    return 2.0 * (abab - aabb)


def complex_spectrum(M0, M1):
    """Complex eigenvalues of the non-normal operator T = M̂0 + i M̂1.

    A PROXY for the Brown measure of T (the spectral distribution of X+iY in the
    trace). CAVEAT — this is NOT the Brown measure proper: the Brown measure is
    taken w.r.t. the tracial state τ = ⟨Ω|·|Ω⟩, whereas the uniform distribution
    over these D matrix eigenvalues corresponds to the normalized matrix trace
    (1/D)Tr ≠ τ. So treat this as a qualitative view of the non-Hermitian
    spectrum; a Brown measure w.r.t. τ needs τ-functional calculus and is
    deferred. Returns the D complex eigenvalues of T.
    """
    T = np.asarray(M0, dtype=float) + 1j * np.asarray(M1, dtype=float)
    return np.linalg.eigvals(T)
