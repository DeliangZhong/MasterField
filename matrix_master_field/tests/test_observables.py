import numpy as np

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.fock_jax import FockOps
from matrix_master_field.observables import (
    commutator_sq_expectation,
    complex_spectrum,
    density_histogram,
    vacuum_spectral_measure,
)


def _free_ops(cut=8):
    ops = FockOps(2, cut)
    ans = MultiMonomialAnsatz(ops, degree=2)
    return [np.asarray(o) for o in ans.build_operators(ans.params_for_free_field())]


def test_vacuum_spectral_measure_reproduces_free_moments():
    # At g=0 the marginal of M̂0 is a unit Wigner semicircle: the vacuum spectral
    # measure must have total mass 1 and moments τ(M0²)=1, τ(M0⁴)=2 (Catalan).
    M0 = _free_ops()[0]
    eigs, w = vacuum_spectral_measure(M0)
    assert np.isclose(w.sum(), 1.0, atol=1e-9)
    assert np.isclose((w * eigs**2).sum(), 1.0, atol=1e-9)
    assert np.isclose((w * eigs**4).sum(), 2.0, atol=1e-9)
    # support inside the semicircle [-2,2] (compression of a norm-2 operator)
    assert eigs.min() > -2.01 and eigs.max() < 2.01


def test_commutator_vanishes_structure_at_free_field():
    # free semicirculars: τ(M0M1M0M1)=0, τ(M0²M1²)=1 ⇒ ⟨tr[M0,M1]²⟩ = 2(0−1) = −2.
    ops = _free_ops()
    assert np.isclose(commutator_sq_expectation(ops), -2.0, atol=1e-9)


def test_complex_spectrum_shape_and_finite():
    M0, M1 = _free_ops()
    z = complex_spectrum(M0, M1)
    assert z.shape[0] == M0.shape[0]
    assert np.all(np.isfinite(z.real)) and np.all(np.isfinite(z.imag))


def test_density_histogram_is_normalized():
    M0 = _free_ops()[0]
    eigs, w = vacuum_spectral_measure(M0)
    centers, rho = density_histogram(eigs, w, bins=40)
    width = centers[1] - centers[0]
    assert np.isclose((rho * width).sum(), 1.0, atol=1e-6)
