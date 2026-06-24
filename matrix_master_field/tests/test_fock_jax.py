import numpy as np

from matrix_master_field.cuntz_fock import CuntzFockSpace
from matrix_master_field.fock_jax import FockOps, power_moments, word_moment


def test_jax_matches_numpy_gaussian_catalan():
    ops = FockOps(n_matrices=1, max_length=8)
    M = ops.a[0] + ops.adag[0]  # x̂ = â + â†
    m = np.asarray(power_moments(M, 8))
    for k, cat in [(0, 1), (2, 1), (4, 2), (6, 5), (8, 14)]:
        assert np.isclose(m[k], cat), f"tr[M^{k}]={m[k]} expected {cat}"


def test_jax_word_moment_matches_numpy():
    npf = CuntzFockSpace(n_matrices=2, max_length=4)
    ops = FockOps(n_matrices=2, max_length=4)
    M1n, M2n = npf.x(0), npf.x(1)
    M1, M2 = ops.a[0] + ops.adag[0], ops.a[1] + ops.adag[1]
    cases = [
        ((0, 1, 0, 1), M1n @ M2n @ M1n @ M2n),  # alternating -> 0
        ((0, 0, 1, 1), M1n @ M1n @ M2n @ M2n),  # factorized -> 1
    ]
    for word, Mn_prod in cases:
        got = float(word_moment([M1, M2], word))
        assert np.isclose(got, npf.vev(Mn_prod), atol=1e-12), f"{word}: {got}"
