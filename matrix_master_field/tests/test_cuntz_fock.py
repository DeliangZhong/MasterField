import numpy as np

from matrix_master_field.cuntz_fock import CuntzFockSpace


def test_cuntz_relations_hold_in_interior():
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    # verify_cuntz_relations returns True if interior relations a_i a†_j = δ_ij hold
    assert fock.verify_cuntz_relations(tol=1e-12) is True


def test_gaussian_operator_gives_catalan():
    fock = CuntzFockSpace(n_matrices=1, max_length=8)
    M = fock.x(0)  # a + a† = free semicircular
    m = fock.compute_moments(M, max_power=8)
    for k, cat in [(0, 1), (2, 1), (4, 2), (6, 5), (8, 14)]:
        assert np.isclose(m[k], cat), f"tr[M^{k}]={m[k]} expected {cat}"


def test_free_semicircular_mixed_moments():
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    M1, M2 = fock.x(0), fock.x(1)
    # Freeness: an alternating centered word vanishes. By the free-Wick rule,
    # tau(x1 x2 x1 x2) sums over non-crossing pairings of EQUAL indices; both
    # NC pairings of (1,2,1,2) pair unequal indices, so the moment is 0.
    assert np.isclose(fock.vev(M1 @ M2 @ M1 @ M2), 0.0, atol=1e-12)
    # Free factorization: tau(x1^2 x2^2) = tau(x1^2) tau(x2^2) = 1.
    assert np.isclose(fock.vev(M1 @ M1 @ M2 @ M2), 1.0, atol=1e-12)
