import itertools

import jax
import numpy as np

from matrix_master_field.ansatz import MultiMonomialAnsatz
from matrix_master_field.fock_jax import FockOps, word_moment
from matrix_master_field.sparse_fock import SparseMonomialField


def _all_words(n, max_len):
    out = [()]
    for L in range(1, max_len + 1):
        out += [tuple(c) for c in itertools.product(range(n), repeat=L)]
    return out


def test_sparse_monomials_match_dense_ordering():
    field = SparseMonomialField(2, cutoff=5, degree=3)
    ops = FockOps(2, 5)
    dense = MultiMonomialAnsatz(ops, degree=3)
    assert field.keys == dense.keys
    assert field.n_params == dense.n_params
    assert field.D == ops.D


def test_sparse_operators_match_dense_elementwise():
    # Same basis ordering as cuntz_fock, so the assembled operators must be identical.
    field = SparseMonomialField(2, cutoff=5, degree=3)
    ops = FockOps(2, 5)
    dense = MultiMonomialAnsatz(ops, degree=3)
    p = jax.random.normal(jax.random.PRNGKey(0), (2, field.n_monomials))
    sp_ops = field.build_dense_operators(p)
    dn_ops = [np.asarray(o) for o in dense.build_operators(p)]
    for a, b in zip(sp_ops, dn_ops):
        assert np.allclose(a, b, atol=1e-12)


def test_sparse_word_moments_match_dense():
    # The decisive check: sparse scatter-add moments == dense matvec moments, for
    # random coefficients over many words (incl. lengths that hit the cutoff).
    field = SparseMonomialField(2, cutoff=5, degree=3)
    ops = FockOps(2, 5)
    dense = MultiMonomialAnsatz(ops, degree=3)
    p = 0.3 * jax.random.normal(jax.random.PRNGKey(1), (2, field.n_monomials))
    dn_ops = dense.build_operators(p)
    for w in _all_words(2, 6):
        s = float(field.word_moment(p, w))
        d = float(word_moment(dn_ops, w))
        assert abs(s - d) < 1e-9, f"word {w}: sparse {s} != dense {d}"


def test_sparse_free_field_reproduces_free_wick():
    field = SparseMonomialField(2, cutoff=6, degree=2)
    p = field.params_for_free_field()
    assert abs(float(field.word_moment(p, (0, 1, 0, 1))) - 0.0) < 1e-12
    assert abs(float(field.word_moment(p, (0, 0, 1, 1))) - 1.0) < 1e-12
    assert abs(float(field.word_moment(p, (0, 0))) - 1.0) < 1e-12
    assert abs(float(field.word_moment(p, (1, 1, 1, 1))) - 2.0) < 1e-12
