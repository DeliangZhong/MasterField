# matrix_master_field/tests/test_exact_diag.py
from math import comb

import numpy as np
import pytest
import scipy.sparse as sp

from matrix_master_field.exact_diag import (
    build_two_matrix_qm_hamiltonian,
    fock_ladder_ops,
    ground_energy,
    hermitian_basis,
    occupation_basis,
    quartic_potential_value,
    structure_constants,
)


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


def test_quartic_matches_minus_tr_commutator_sq():
    rng = np.random.default_rng(0)
    for N in (2, 3):
        T = hermitian_basis(N)
        for _ in range(5):
            x = rng.standard_normal(N * N)
            y = rng.standard_normal(N * N)
            X = np.einsum("a,aij->ij", x, T)
            Y = np.einsum("a,aij->ij", y, T)
            comm = X @ Y - Y @ X
            ref = -np.trace(comm @ comm)  # = +sum_c L_c^2, real and >= 0
            assert abs(ref.imag) < 1e-10
            val = quartic_potential_value(N, x, y)
            assert np.isclose(val, ref.real, atol=1e-10)
            assert val >= -1e-12  # positive (confining)


def test_occupation_basis_size_and_bound():
    for n_modes, K in [(6, 4), (3, 5), (2, 7)]:
        occ = occupation_basis(n_modes, K)
        assert occ.shape == (comb(K + n_modes, n_modes), n_modes)
        assert occ.sum(axis=1).max() <= K
        assert occ.min() >= 0
        # all rows distinct
        assert len({tuple(r) for r in occ}) == occ.shape[0]


def test_ladder_commutator_interior():
    n_modes, K = 3, 5
    occ, ops = fock_ladder_ops(n_modes, K)
    D = occ.shape[0]
    for i in range(n_modes):
        a = ops[i]
        adag = a.transpose()
        comm = (a @ adag - adag @ a).toarray()
        # [a_i, a_i^dag] = 1 on states with total quanta < K (interior, not truncated)
        for r in range(D):
            if occ[r].sum() < K:
                assert np.isclose(comm[r, r], 1.0, atol=1e-12)


def test_number_operator_eigenvalues():
    n_modes, K = 4, 4
    occ, ops = fock_ladder_ops(n_modes, K)
    for i in range(n_modes):
        num = (ops[i].transpose() @ ops[i]).diagonal()
        assert np.allclose(num, occ[:, i], atol=1e-12)


def test_ladder_lowers_one_quantum():
    n_modes, K = 2, 3
    occ, ops = fock_ladder_ops(n_modes, K)
    index = {tuple(occ[r]): r for r in range(occ.shape[0])}
    a0 = ops[0]
    for r in range(occ.shape[0]):
        if occ[r, 0] > 0:
            tgt = occ[r].copy(); tgt[0] -= 1
            rt = index[tuple(tgt)]
            assert np.isclose(a0[rt, r], np.sqrt(occ[r, 0]), atol=1e-12)


def test_hamiltonian_hermitian_and_dim():
    N, m, K = 2, 1.0, 3
    H = build_two_matrix_qm_hamiltonian(N, m, g=0.7, K=K)
    n_modes = 2 * (N * N - 1)
    assert H.shape[0] == comb(K + n_modes, n_modes)
    assert abs((H - H.transpose()).max()) < 1e-12  # real symmetric


def test_hamiltonian_g0_is_diagonal_free_spectrum():
    # g=0: H_interacting diagonal, lowest entry = m * 2(N^2-1) (all interacting modes n=0)
    N, m, K = 2, 1.0, 4
    H = build_two_matrix_qm_hamiltonian(N, m, g=0.0, K=K).toarray()
    n_int = 2 * (N * N - 1)
    assert np.allclose(H - np.diag(np.diag(H)), 0.0, atol=1e-12)
    assert np.isclose(np.min(np.diag(H)), m * n_int, atol=1e-12)


def test_hamiltonian_quartic_is_psd_shift():
    # the quartic g^2 sum_c L_c^2 is PSD: H(g) - H(0) has nonnegative eigenvalues
    N, m, K = 2, 1.0, 3
    H0 = build_two_matrix_qm_hamiltonian(N, m, 0.0, K).toarray()
    Hg = build_two_matrix_qm_hamiltonian(N, m, 0.9, K).toarray()
    w = np.linalg.eigvalsh(Hg - H0)
    assert w.min() > -1e-9


def test_g0_anchor_exact_all_N_K():
    # g=0 => E/N^2 = 2m exactly, any N, any K (the hard check, V2)
    for N, K in [(2, 2), (2, 5), (3, 2), (3, 3)]:
        res = ground_energy(N, m=1.0, g=0.0, K=K)
        assert np.isclose(res["E_over_N2"], 2.0, atol=1e-9)
        assert res["basis_dim"] == comb(K + 2 * (N * N - 1), 2 * (N * N - 1))


def test_g0_anchor_scales_with_m():
    res = ground_energy(2, m=1.7, g=0.0, K=3)
    assert np.isclose(res["E_over_N2"], 2 * 1.7, atol=1e-9)


def test_ground_energy_above_2m_when_interacting():
    res = ground_energy(2, m=1.0, g=0.8, K=6)
    assert res["E_over_N2"] >= 2.0 - 1e-9  # Rayleigh-Ritz: E_trunc >= E_true >= 2m
