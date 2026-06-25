"""M5b — collective-field master field of single-matrix QM H = Tr P² + Tr X² + (g/N) Tr X⁴.

Large-N singlet sector = N free fermions; the master field is the rescaled eigenvalue
density σ(y) (y=λ/√N), minimizing
    E/N² [σ] = ∫ [ π²σ³/3 + (y² + g y⁴) σ ] dy,  ∫σ = 1,  σ ≥ 0.
Analytic minimizer σ(y)=(1/π)√(μ−y²−g y⁴), μ fixed by normalization. The finite-N
free-fermion referee fills the N lowest single-particle levels of h=p²+λ²+(g/N)λ⁴.
See docs/superpowers/specs/2026-06-25-m5b-single-matrix-qm-design.md.
"""
import numpy as np
from scipy import integrate, optimize


def _support_umax(mu, g):
    """Largest u=y² with u + g u² = μ (the density support edge)."""
    return (-1.0 + np.sqrt(1.0 + 4.0 * g * mu)) / (2.0 * g) if g > 0 else mu


def collective_master_field(g):
    """Analytic large-N ground state: σ=(1/π)√(μ−y²−g y⁴), μ from ∫σ=1.

    Returns dict(mu, energy=E/N², m2=⟨X̃²⟩, m4=⟨X̃⁴⟩, ys, density). g=0 ⇒ energy=1, m2=½.
    """
    def sig(y, mu):
        v = mu - y ** 2 - g * y ** 4
        return np.sqrt(np.maximum(v, 0.0)) / np.pi

    def norm(mu):
        ym = np.sqrt(_support_umax(mu, g))
        return integrate.quad(lambda y: sig(y, mu), -ym, ym)[0]

    mu = optimize.brentq(lambda m: norm(m) - 1.0, 1e-3, 100.0)
    ym = np.sqrt(_support_umax(mu, g))
    ekin = integrate.quad(lambda y: np.pi ** 2 * sig(y, mu) ** 3 / 3.0, -ym, ym)[0]
    epot = integrate.quad(lambda y: (y ** 2 + g * y ** 4) * sig(y, mu), -ym, ym)[0]
    m2 = integrate.quad(lambda y: y ** 2 * sig(y, mu), -ym, ym)[0]
    m4 = integrate.quad(lambda y: y ** 4 * sig(y, mu), -ym, ym)[0]
    ys = np.linspace(-ym, ym, 400)
    return {"mu": mu, "energy": ekin + epot, "m2": m2, "m4": m4,
            "ys": ys, "density": sig(ys, mu)}


def collective_energy_density(sigma, ys, g):
    """E/N²[σ] = ∫[π²σ³/3 + (y²+g y⁴)σ] dy on a grid (trapezoid). Used by the variational
    upper bound (Task 2)."""
    sigma = np.asarray(sigma, dtype=float)
    ys = np.asarray(ys, dtype=float)
    integrand = np.pi ** 2 * sigma ** 3 / 3.0 + (ys ** 2 + g * ys ** 4) * sigma
    return float(np.trapezoid(integrand, ys))


def free_fermion_energy(g, N, n_basis=None):
    """Finite-N referee: E/N² = (1/N²) Σ of the N lowest single-particle levels of
    h = p² + λ² + (g/N) λ⁴ (= the M5a qm_fock Hamiltonian with coupling g/N). At g=0,
    h has levels 2n+1 so Σ_{0}^{N-1}(2n+1)=N² ⇒ E/N²=1 exactly at any N.
    """
    from matrix_master_field.qm_fock import hamiltonian_anharmonic

    M = n_basis if n_basis is not None else 4 * N + 40
    h = np.asarray(hamiltonian_anharmonic(M, g / N))
    w = np.sort(np.linalg.eigvalsh(0.5 * (h + h.conj().T)).real)
    return float(np.sum(w[:N]) / N ** 2)
