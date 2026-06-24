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
