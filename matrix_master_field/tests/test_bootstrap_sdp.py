import pytest

from matrix_master_field import bootstrap_sdp as bs
from matrix_master_field import one_matrix as om

pytestmark = pytest.mark.skipif(not bs.HAS_CVXPY, reason="cvxpy not installed")


def test_gaussian_bounds_bracket_exact_m2():
    exact = om.gaussian_moments(8)[2]  # = 1
    lb = bs.bootstrap_one_matrix([0.0, 1.0], max_moment=8, target_moment=2, maximize=False)
    ub = bs.bootstrap_one_matrix([0.0, 1.0], max_moment=8, target_moment=2, maximize=True)
    assert lb is not None and ub is not None
    assert lb - 1e-4 <= exact <= ub + 1e-4, f"exact {exact} not in [{lb}, {ub}]"


def test_quartic_bounds_bracket_exact_m2():
    exact = om.quartic_moments_from_sd(0.5, max_power=8)[2]
    lb = bs.bootstrap_one_matrix([0.0, 1.0, 0.0, 0.5], max_moment=8, target_moment=2, maximize=False)
    ub = bs.bootstrap_one_matrix([0.0, 1.0, 0.0, 0.5], max_moment=8, target_moment=2, maximize=True)
    assert lb is not None and ub is not None
    assert lb - 1e-3 <= exact <= ub + 1e-3, f"exact {exact} not in [{lb}, {ub}]"


def test_gaussian_bounds_bracket_higher_catalan_moments():
    # Regression for the SD-linearization bug (Codex F1): the old code forced
    # m4 = 3*m2 = 3, EXCLUDING the Catalan value m4 = 2. The product-matrix
    # relaxation must bracket the true Catalan moments m4=2, m6=5.
    for target, val in [(4, 2.0), (6, 5.0)]:
        lb = bs.bootstrap_one_matrix([0.0, 1.0], max_moment=10, target_moment=target, maximize=False)
        ub = bs.bootstrap_one_matrix([0.0, 1.0], max_moment=10, target_moment=target, maximize=True)
        assert lb is not None and ub is not None
        assert lb - 0.05 <= val <= ub + 0.05, f"m{target}={val} not in [{lb:.4f}, {ub:.4f}]"
