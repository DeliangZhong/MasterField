"""M5b — collective-field master field + free-fermion referee for single-matrix QM."""
import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from matrix_master_field.qm_collective import (  # noqa: E402
    collective_master_field,
    free_fermion_energy,
)

# Verified exact large-N anchors (E/N^2, <X~^2>, <X~^4>); see the M5b spec de-risk.
EXACT = {0.0: (1.0, 0.5, 0.5), 0.5: (1.18049, 0.37943, 0.28110),
         1.0: (1.30190, 0.33143, 0.21301), 2.0: (1.48047, 0.28161, 0.15288)}


def test_collective_matches_exact_table():
    for g, (e, m2, m4) in EXACT.items():
        r = collective_master_field(g)
        assert abs(r["energy"] - e) < 1e-4
        assert abs(r["m2"] - m2) < 1e-4
        assert abs(r["m4"] - m4) < 1e-4


def test_g0_exact_energy_one():
    r = collective_master_field(0.0)
    assert abs(r["energy"] - 1.0) < 1e-6
    assert abs(r["m2"] - 0.5) < 1e-6


def test_free_fermion_converges_to_collective():
    for g in (0.0, 1.0):
        e_inf = collective_master_field(g)["energy"]
        e_N = [free_fermion_energy(g, N) for N in (20, 40, 80)]
        assert abs(e_N[-1] - e_inf) < abs(e_N[0] - e_inf) + 1e-9  # converging in N
        assert abs(e_N[-1] - e_inf) < 5e-3
    assert abs(free_fermion_energy(0.0, 50) - 1.0) < 1e-12  # g=0 exact at any N
