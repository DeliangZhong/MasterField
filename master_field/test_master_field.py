"""Automated test suite for master field computation.

Run: python test_master_field.py

All tests must pass before proceeding to ML optimisation.
"""

from math import comb

import numpy as np


def test_cuntz_algebra():
    """a_i a†_j = δ_ij in truncated Fock space."""
    from cuntz_fock import CuntzFockSpace

    for n in [1, 2, 3]:
        for L in [4, 6]:
            fock = CuntzFockSpace(n, L)
            assert fock.verify_cuntz_relations(), f"Failed for n={n}, L={L}"
    print("✓ Cuntz algebra")


def test_gaussian_moments():
    """M̂ = â + â† reproduces Catalan numbers."""
    from cuntz_fock import CuntzFockSpace

    fock = CuntzFockSpace(1, 10)
    M = fock.x(0)
    moments = fock.compute_moments(M, 10)
    for k in range(6):
        expected = comb(2 * k, k) / (k + 1)
        assert abs(moments[2 * k] - expected) < 1e-10, (
            f"m_{2 * k}: got {moments[2 * k]}, expected {expected}"
        )
    print("✓ Gaussian moments (Catalan)")


def test_gaussian_free_cumulants():
    """Gaussian has κ_2=1, all others zero."""
    from one_matrix import gaussian_moments, r_transform_from_moments

    m = gaussian_moments(10)
    kappa = r_transform_from_moments(m)
    assert abs(kappa[2] - 1.0) < 1e-10, f"κ_2 = {kappa[2]}"
    for k in [1, 3, 4, 5]:
        assert abs(kappa[k]) < 1e-8, f"κ_{k} = {kappa[k]} (should be 0)"
    print("✓ Gaussian free cumulants")


def test_sd_indexing():
    """Verify SD equation: m_{n+1} = Σ_{j=0}^{n-1} m_j m_{n-j-1} for Gaussian."""
    catalan = [comb(2 * k, k) / (k + 1) for k in range(8)]
    full_m = [0.0] * 15
    for k in range(8):
        full_m[2 * k] = catalan[k]

    for n in range(7):
        lhs = full_m[n + 1]  # V'=M → v_1=1 → index n+1
        rhs = sum(full_m[j] * full_m[n - j - 1] for j in range(n))
        assert abs(lhs - rhs) < 1e-12, f"n={n}: LHS={lhs}, RHS={rhs}"
    print("✓ SD indexing verified")


def test_quartic_sd_consistency():
    """Quartic moments from density satisfy SD equations."""
    from one_matrix import quartic_moments_from_sd
    from schwinger_dyson import OneMatrixSD

    for g in [0.1, 0.5, 1.0, 2.0]:
        m = quartic_moments_from_sd(g, 12)
        sd = OneMatrixSD([0, 1.0, 0, g], max_word_length=10)
        omega = np.zeros(sd.n_vars)
        for i, w in enumerate(sd.words):
            k = len(w)
            if k <= 12:
                omega[i] = m[k]
        res = sd.sd_residuals(omega)
        maxr = np.max(np.abs(res))
        assert maxr < 1e-4, f"g={g}: max residual = {maxr}"
    print("✓ Quartic SD consistency")


def test_free_product():
    """tr[M1 M2 M1 M2] = 0 for free semicirculars with zero mean.
    tr[M1² M2²] = tr[M1²]·tr[M2²] = 1.
    """
    from cuntz_fock import CuntzFockSpace

    fock = CuntzFockSpace(2, 5)
    M1, M2 = fock.x(0), fock.x(1)

    val = fock.vev(M1 @ M2 @ M1 @ M2)
    assert abs(val) < 1e-12, f"tr[M1 M2 M1 M2] = {val}, expected 0"

    val2 = fock.vev(M1 @ M1 @ M2 @ M2)
    assert abs(val2 - 1.0) < 1e-10, f"tr[M1² M2²] = {val2}, expected 1"
    print("✓ Free product relations")


def test_psd_constraint():
    """Gaussian moment matrix is PSD."""
    from one_matrix import gaussian_moments
    from schwinger_dyson import LoopMomentMatrix

    m = gaussian_moments(10)
    lmm = LoopMomentMatrix(1, 8)

    def moment_func(word):
        k = len(word)
        if k == 0:
            return 1.0
        if k <= 10:
            return m[k]
        return 0.0

    is_psd, min_eig = lmm.check_psd(moment_func)
    assert is_psd, f"Gaussian moment matrix not PSD: min_eig = {min_eig}"
    print(f"✓ PSD constraint (min eigenvalue = {min_eig:.6f})")


def test_voiculescu_roundtrip():
    """moments → free cumulants → Voiculescu coeffs → Fock VEVs → moments."""
    from cuntz_fock import CuntzFockSpace
    from one_matrix import gaussian_moments, r_transform_from_moments, voiculescu_coefficients

    m = gaussian_moments(8)
    kappa = r_transform_from_moments(m[:9])
    v_coeffs = voiculescu_coefficients(kappa)

    # L must exceed max moment order to avoid truncation artifacts
    fock = CuntzFockSpace(1, 10)
    M_hat = fock.build_master_field_voiculescu(v_coeffs[:7])
    m_fock = fock.compute_moments(M_hat, 8)

    for k in range(0, 9, 2):
        assert abs(m_fock[k] - m[k]) < 1e-6, f"Roundtrip failed at m_{k}: {m_fock[k]} vs {m[k]}"
    print("✓ Voiculescu roundtrip")


def test_quartic_density_normalisation():
    """Quartic eigenvalue density integrates to 1."""
    from one_matrix import quartic_eigenvalue_density

    for g in [0.1, 0.5, 1.0, 5.0]:
        x, rho = quartic_eigenvalue_density(g)
        norm = np.trapezoid(rho, x)
        assert abs(norm - 1.0) < 1e-3, f"g={g}: ∫ρ = {norm}"
    print("✓ Quartic density normalisation")


def test_moment_matrix_dimensions():
    """Moment matrix has correct dimensions."""
    from schwinger_dyson import LoopMomentMatrix

    # n=1, L=6: basis = {∅, (0), (0,0), (0,0,0)} → dim=4
    lmm = LoopMomentMatrix(1, 6)
    assert lmm.basis_dim == 4, f"Expected 4, got {lmm.basis_dim}"

    # n=2, L=4: basis = {∅, (0), (1), (0,0), (0,1), (1,0), (1,1)} → dim=7
    lmm2 = LoopMomentMatrix(2, 4)
    assert lmm2.basis_dim == 7, f"Expected 7, got {lmm2.basis_dim}"
    print("✓ Moment matrix dimensions")


if __name__ == "__main__":
    print("=" * 50)
    print("Master Field — Test Suite")
    print("=" * 50 + "\n")

    test_cuntz_algebra()
    test_gaussian_moments()
    test_gaussian_free_cumulants()
    test_sd_indexing()
    test_quartic_sd_consistency()
    test_quartic_density_normalisation()
    test_free_product()
    test_psd_constraint()
    test_moment_matrix_dimensions()
    test_voiculescu_roundtrip()

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED ✓")
    print("=" * 50)
