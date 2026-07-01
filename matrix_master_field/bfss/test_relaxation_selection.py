"""Regression tests for the Part-A/Part-C non-relaxation & selection diagnostics
on the massive D=2 Lin-Zheng two-matrix QM.

Run:  uv run --no-project --with numpy --with scipy --with cvxpy \
          python matrix_master_field/bfss/test_relaxation_selection.py
(or via pytest with the same deps).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import relaxation_selection as rs  # noqa: E402


def test_variety_dimension_matches_table_I():
    """The exact factorized loop-variety dimension equals Lin-Zheng Table I's free
    count (3, 8 at level 4, 6). This is the number of directions positivity must fix."""
    for level, expect in [(4, 3), (6, 8)]:
        S = rs.System(level, 1.0)
        d = rs.variety_dimension(S, n_probe=6, verbose=False)
        assert d["dim"] == expect, f"level {level}: dim={d['dim']}, expected {expect}"


def test_dimension_stable_under_reseeding():
    """The Jacobian rank (hence variety dimension) is a smooth-locus invariant: it must
    not depend on the random seeds used to reach the variety (Part-C stability check)."""
    S = rs.System(4, 1.0)
    d0 = rs.variety_dimension(S, seed=0, n_probe=6, verbose=False)
    d1 = rs.variety_dimension(S, seed=123, n_probe=6, verbose=False)
    assert d0["dim"] == d1["dim"], f"dim unstable: {d0['dim']} vs {d1['dim']}"
    assert d0["rank_min"] == d0["rank_max"], "rank not constant across projected seeds"


def test_analytic_positivity_lower_bound_L4():
    """Minimising E over {exact factorization + M>=0 + N>=0} at level 4 reproduces the
    analytic positivity lower bound E >= D*M*sqrt(3)/4 = sqrt(3)/2 (D=2, M=1)."""
    S = rs.System(4, 1.0)
    ext = rs.positive_extent(S, n_starts=2, verbose=False)
    bound = math.sqrt(3) / 2  # = 0.8660254...
    assert abs(ext["E_min_pos"] - bound) < 5e-3, \
        f"E_min={ext['E_min_pos']} vs analytic {bound}"


def test_positive_region_is_a_set_not_a_point_L4():
    """The crux: at finite truncation positivity does NOT collapse the loop variety to a
    point — the positive E-extent has finite width (a SET), with the rigorous island
    strictly inside it. (Selection needs the extra level-growth criterion.)"""
    S = rs.System(4, 1.0)
    ext = rs.positive_extent(S, n_starts=2, verbose=False)
    assert ext["pos_width"] > 0.5, f"expected a wide positive set, got {ext['pos_width']}"
    assert not ext["pos_is_point"], "positivity unexpectedly collapsed to a point at L4"
    assert ext["E_min_pos"] <= rs.ISLAND_E[0] + 1e-6, "island below the positive set"


if __name__ == "__main__":
    tests = [test_variety_dimension_matches_table_I,
             test_dimension_stable_under_reseeding,
             test_analytic_positivity_lower_bound_L4,
             test_positive_region_is_a_set_not_a_point_L4]
    fail = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            fail += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - fail}/{len(tests)} passed")
    sys.exit(1 if fail else 0)
