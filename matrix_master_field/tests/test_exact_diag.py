# matrix_master_field/tests/test_exact_diag.py
import numpy as np
import pytest

from matrix_master_field.exact_diag import hermitian_basis, structure_constants


def _commutator(A, B):
    return A @ B - B @ A


def test_basis_orthonormal_and_trace_mode():
    for N in (2, 3):
        T = hermitian_basis(N)
        assert T.shape == (N * N, N, N)
        # orthonormality Tr(T_a T_b) = delta_ab
        gram = np.einsum("aij,bji->ab", T, T)
        assert np.allclose(gram, np.eye(N * N), atol=1e-12)
        # index 0 is I/sqrt(N); the rest are traceless and Hermitian
        assert np.allclose(T[0], np.eye(N) / np.sqrt(N), atol=1e-12)
        for a in range(1, N * N):
            assert abs(np.trace(T[a])) < 1e-12
            assert np.allclose(T[a], T[a].conj().T, atol=1e-12)


def test_basis_n2_is_pauli():
    T = hermitian_basis(2)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(T[1], sx / np.sqrt(2), atol=1e-12)
    assert np.allclose(T[2], sy / np.sqrt(2), atol=1e-12)
    assert np.allclose(T[3], sz / np.sqrt(2), atol=1e-12)


def test_structure_constants_real_and_antisymmetric():
    for N in (2, 3):
        f = structure_constants(N)
        assert f.shape == (N * N,) * 3
        assert f.dtype == np.float64
        # totally antisymmetric
        assert np.allclose(f, -np.transpose(f, (1, 0, 2)), atol=1e-10)
        assert np.allclose(f, -np.transpose(f, (0, 2, 1)), atol=1e-10)
        # trace mode commutes with everything -> any f with a 0 index vanishes
        assert np.allclose(f[0], 0.0, atol=1e-12)
        assert np.allclose(f[:, 0], 0.0, atol=1e-12)
        assert np.allclose(f[:, :, 0], 0.0, atol=1e-12)


def test_structure_constants_reconstruct_commutator():
    for N in (2, 3):
        T = hermitian_basis(N)
        f = structure_constants(N)
        for a in range(N * N):
            for b in range(N * N):
                lhs = _commutator(T[a], T[b])
                rhs = 1j * np.einsum("c,cij->ij", f[a, b], T)
                assert np.allclose(lhs, rhs, atol=1e-10)


def test_structure_constants_n2_value():
    f = structure_constants(2)
    # [T_1,T_2]=i sqrt(2) T_3 for Pauli/sqrt(2): f_{123}=sqrt(2)
    assert np.isclose(f[1, 2, 3], np.sqrt(2.0), atol=1e-10)
