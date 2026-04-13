"""Tests for the matrix-free expm-v module."""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np
import pytest

from .fock import CuntzFockJAX
from .hermitian_operator import assemble_hermitian, init_hermitian_params
from .matfree_expm import (
    WordPairs,
    build_word_pairs,
    expm_iH_v,
    expm_iH_v_norm_check,
    h_matvec,
)


# --------------------------------------------------------------
# build_word_pairs
# --------------------------------------------------------------


@pytest.mark.unit
def test_word_pairs_nnz_counts():
    """Total nnz across all C_w must match the analytic sum.

    For n_labels = n, L_trunc = L, and word w of length k:
      # valid preimages u = # basis states of length <= L - k
                           = sum_{j=0}^{L-k} n^j
    Total across all w:
      sum_{k=0}^{L} n^k * (sum_{j=0}^{L-k} n^j)
    """
    for (n, L) in [(1, 4), (2, 3), (4, 3)]:
        fock = CuntzFockJAX(n_labels=n, L_trunc=L)
        wp = build_word_pairs(fock)
        # analytic
        counts_by_len = [n ** k for k in range(L + 1)]
        expected = 0
        for k in range(L + 1):
            n_words_k = counts_by_len[k]
            preimages_k = sum(counts_by_len[: L - k + 1])
            expected += n_words_k * preimages_k
        assert wp.n_nnz == expected, (
            f"n={n} L={L}: got nnz={wp.n_nnz}, expected {expected}"
        )


@pytest.mark.unit
def test_word_pairs_specific_counts():
    """Known values: n=4, L=3 -> nnz=313; n=4, L=4 -> nnz=1593."""
    fock_34 = CuntzFockJAX(n_labels=4, L_trunc=3)
    assert build_word_pairs(fock_34).n_nnz == 313
    fock_44 = CuntzFockJAX(n_labels=4, L_trunc=4)
    assert build_word_pairs(fock_44).n_nnz == 1593


@pytest.mark.unit
def test_word_pairs_indices_in_range():
    """All src, tgt, w indices are within [0, dim)."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    assert int(jnp.min(wp.all_src)) >= 0
    assert int(jnp.max(wp.all_src)) < fock.dim
    assert int(jnp.min(wp.all_tgt)) >= 0
    assert int(jnp.max(wp.all_tgt)) < fock.dim
    assert int(jnp.min(wp.all_w)) >= 0
    assert int(jnp.max(wp.all_w)) < fock.dim


# --------------------------------------------------------------
# h_matvec
# --------------------------------------------------------------


def _random_h(fock: CuntzFockJAX, seed: int, scale: float) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(fock.dim) * scale
    im = rng.standard_normal(fock.dim) * scale
    return jnp.asarray(re + 1j * im).astype(jnp.complex128)


def _random_v(fock: CuntzFockJAX, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    re = rng.standard_normal(fock.dim)
    im = rng.standard_normal(fock.dim)
    v = re + 1j * im
    v = v / np.linalg.norm(v)
    return jnp.asarray(v).astype(jnp.complex128)


@pytest.mark.unit
def test_h_matvec_agrees_with_dense_small():
    """h_matvec(h, v) == assemble_hermitian(h) @ v to 1e-12."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=1, scale=0.1)
    v = _random_v(fock, seed=2)
    H_dense = assemble_hermitian(h, fock)
    v_dense = H_dense @ v
    v_matfree = h_matvec(h, v, wp)
    err = float(jnp.max(jnp.abs(v_dense - v_matfree)))
    assert err < 1e-12, f"h_matvec disagrees: max err {err:.3e}"


@pytest.mark.unit
def test_h_matvec_agrees_with_dense_at_boundary():
    """Agreement holds even for h peaked at long-word coefficients."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    # Random but non-trivial h
    h = _random_h(fock, seed=42, scale=0.5)
    v = _random_v(fock, seed=43)
    H_dense = assemble_hermitian(h, fock)
    err = float(jnp.max(jnp.abs(H_dense @ v - h_matvec(h, v, wp))))
    assert err < 1e-12


@pytest.mark.unit
def test_h_matvec_linearity():
    """h_matvec is linear in v (fixed h)."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=1, scale=0.1)
    v1 = _random_v(fock, seed=2)
    v2 = _random_v(fock, seed=3)
    a, b = 0.7, -1.3
    lhs = h_matvec(h, a * v1 + b * v2, wp)
    rhs = a * h_matvec(h, v1, wp) + b * h_matvec(h, v2, wp)
    err = float(jnp.max(jnp.abs(lhs - rhs)))
    assert err < 1e-12


# --------------------------------------------------------------
# expm_iH_v
# --------------------------------------------------------------


@pytest.mark.unit
def test_expm_iH_v_agrees_with_dense_small_h():
    """At ||h||=0.05, Taylor order=25 should match dense expm to 1e-10."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=7, scale=0.05)
    v = _random_v(fock, seed=8)
    H = assemble_hermitian(h, fock)
    U = jax.scipy.linalg.expm(1j * H)
    dense = U @ v
    matfree = expm_iH_v(h, v, wp, order=25, sign=+1.0)
    err = float(jnp.max(jnp.abs(dense - matfree)))
    assert err < 1e-10, f"expm_iH_v small h disagrees: {err:.3e}"


@pytest.mark.unit
def test_expm_iH_v_agrees_with_dense_moderate_h():
    """At ||h||=0.5, Taylor order=30 should still match to 1e-9."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=11, scale=0.5)
    v = _random_v(fock, seed=12)
    H = assemble_hermitian(h, fock)
    U = jax.scipy.linalg.expm(1j * H)
    dense = U @ v
    matfree = expm_iH_v(h, v, wp, order=30, sign=+1.0)
    err = float(jnp.max(jnp.abs(dense - matfree)))
    assert err < 1e-9, f"expm_iH_v moderate h disagrees: {err:.3e}"


@pytest.mark.unit
def test_expm_iH_v_dagger_agrees_with_dense():
    """sign=-1 path gives e^{-iH} v = (e^{iH})^dag v."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=17, scale=0.1)
    v = _random_v(fock, seed=18)
    H = assemble_hermitian(h, fock)
    U = jax.scipy.linalg.expm(1j * H)
    dense_dag = U.conj().T @ v
    matfree = expm_iH_v(h, v, wp, order=25, sign=-1.0)
    err = float(jnp.max(jnp.abs(dense_dag - matfree)))
    assert err < 1e-10


@pytest.mark.unit
def test_expm_iH_v_preserves_norm_small_h():
    """Taylor e^{iH} is approximately unitary: ||e^{iH}v|| = ||v|| to 1e-10
    at small ||h||."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=21, scale=0.05)
    v = _random_v(fock, seed=22)
    v_out = expm_iH_v(h, v, wp, order=25)
    nv = float(jnp.linalg.norm(v))
    no = float(jnp.linalg.norm(v_out))
    assert abs(nv - no) < 1e-10


@pytest.mark.unit
def test_expm_iH_v_norm_check_diagnostic():
    """expm_iH_v_norm_check's last-term relative norm is a convergence
    diagnostic."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=31, scale=0.1)
    v = _random_v(fock, seed=32)
    _, last_rel = expm_iH_v_norm_check(h, v, wp, order=25)
    assert last_rel < 1e-15, (
        f"order=25 at ||h||=0.1 should converge to machine zero, got {last_rel}"
    )


# --------------------------------------------------------------
# autodiff through expm_iH_v
# --------------------------------------------------------------


@pytest.mark.unit
def test_grad_through_expm_iH_v_matches_fd():
    """jax.grad of a scalar loss through expm_iH_v matches finite
    differences to 1e-6."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    v = _random_v(fock, seed=101)
    # Target (constant) to make the loss real scalar
    target_real = 0.07
    target_imag = 0.0

    def loss(h):
        u = expm_iH_v(h, v, wp, order=20)
        # take the 0th component real/imag
        w_real = jnp.real(u[0])
        w_imag = jnp.imag(u[0])
        return (w_real - target_real) ** 2 + (w_imag - target_imag) ** 2

    h0 = _random_h(fock, seed=102, scale=0.05)
    grad_fn = jax.grad(loss)
    # JAX grad on real scalar of complex h returns (df/dx - i df/dy);
    # compare its real and imag parts to finite differences.
    g_jax = grad_fn(h0)
    eps = 1e-5

    # finite diff on a few random components
    rng = np.random.default_rng(99)
    idxs = rng.choice(fock.dim, size=5, replace=False)
    for idx in idxs:
        # real part
        dh = np.zeros(fock.dim, dtype=np.complex128)
        dh[idx] = eps
        dh_j = jnp.asarray(dh)
        f_plus = float(loss(h0 + dh_j))
        f_minus = float(loss(h0 - dh_j))
        fd_real = (f_plus - f_minus) / (2 * eps)
        # imag part
        dh = np.zeros(fock.dim, dtype=np.complex128)
        dh[idx] = 1j * eps
        dh_j = jnp.asarray(dh)
        f_plus = float(loss(h0 + dh_j))
        f_minus = float(loss(h0 - dh_j))
        fd_imag = (f_plus - f_minus) / (2 * eps)

        # JAX complex grad convention: g_jax = df/dx - i df/dy
        # So real(g_jax) = df/dx = fd_real
        #    -imag(g_jax) = df/dy = fd_imag
        jax_dx = float(jnp.real(g_jax[idx]))
        jax_dy = -float(jnp.imag(g_jax[idx]))
        assert abs(jax_dx - fd_real) < 1e-5, (
            f"idx={idx}: jax df/dx={jax_dx}, FD={fd_real}"
        )
        assert abs(jax_dy - fd_imag) < 1e-5, (
            f"idx={idx}: jax df/dy={jax_dy}, FD={fd_imag}"
        )


# --------------------------------------------------------------
# JIT compatibility
# --------------------------------------------------------------


@pytest.mark.unit
def test_expm_iH_v_jit_compiles():
    """expm_iH_v should be JIT-compatible (fixed Taylor order unrolls)."""
    fock = CuntzFockJAX(n_labels=4, L_trunc=3)
    wp = build_word_pairs(fock)
    h = _random_h(fock, seed=201, scale=0.05)
    v = _random_v(fock, seed=202)

    # Compile
    f = jax.jit(lambda hh, vv: expm_iH_v(hh, vv, wp, order=15))
    out1 = f(h, v)
    out2 = f(h, v)
    # Same result twice, no error
    assert jnp.allclose(out1, out2)
    # Matches non-jit
    out_nojit = expm_iH_v(h, v, wp, order=15)
    assert jnp.allclose(out1, out_nojit, atol=1e-14)
