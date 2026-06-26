# matrix_master_field/tests/test_exact_diag.py
from math import comb

import numpy as np
import pytest
import scipy.sparse as sp

from matrix_master_field.exact_diag import (
    build_two_matrix_qm_hamiltonian,
    casimir_of_ground_state,
    converge_in_K,
    fock_ladder_ops,
    gaussian_upper,
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


def test_K_convergence_monotone_and_settles():
    # variational in K: E/N^2(K) non-increasing (Rayleigh-Ritz, rigorous), and the
    # successive gaps shrink -- the signature of convergence (V3). N=2, lambda=1.
    N, m = 2, 1.0
    g = np.sqrt(1.0 / N)  # lambda = N g^2 = 1
    out = converge_in_K(N, m, g, K_list=[4, 6, 8, 10])
    vals = [e for _, e in out["series"]]
    for lo, hi in zip(vals[1:], vals[:-1]):
        assert lo <= hi + 1e-9          # non-increasing (guaranteed)
    assert vals[-1] >= 2.0 - 1e-9       # still >= 2m (guaranteed)
    gaps = [abs(b - a) for a, b in zip(vals[:-1], vals[1:])]
    assert gaps[-1] <= gaps[0] + 1e-12  # converging: last gap no larger than the first
    assert out["tail"] == pytest.approx(gaps[-1], abs=1e-12)


def test_K_convergence_g0_flat_at_2m():
    out = converge_in_K(2, 1.0, 0.0, K_list=[2, 4, 6])
    assert all(np.isclose(e, 2.0, atol=1e-9) for _, e in out["series"])


def test_gaussian_g0_is_2m():
    for N in (2, 3):
        out = gaussian_upper(N, m=1.0, g=0.0)
        assert np.isclose(out["E_over_N2"], 2.0, atol=1e-9)
        assert np.isclose(out["omega"], 1.0, atol=1e-6)  # optimal omega = m at g=0


def test_v4a_bracket_finite_N():
    # 2m <= E_exact(converged K) <= same-N Gaussian, for g>0, N=2, lambda=1 (V4a)
    N, m = 2, 1.0
    g = np.sqrt(1.0 / N)
    e_exact = converge_in_K(N, m, g, K_list=[8, 10, 12])["value"]
    e_gauss = gaussian_upper(N, m, g)["E_over_N2"]
    assert 2.0 - 1e-9 <= e_exact
    assert e_exact <= e_gauss + 1e-6   # converged exact sits below the same-N Gaussian


def test_casimir_singlet_g0_vacuum():
    # g=0 ground state = full vacuum, annihilated by every generator => Casimir ~ 0
    c2 = casimir_of_ground_state(2, m=1.0, g=0.0, K=3)
    assert abs(c2) < 1e-9


def test_casimir_ground_state_is_singlet():
    # interacting ground state is a singlet (gauge-invariant H, invariant vacuum) -> ~0
    c2 = casimir_of_ground_state(2, m=1.0, g=0.8, K=6)
    assert abs(c2) < 1e-6


def test_casimir_nonzero_on_non_singlet():
    # a single-quantum state in one mode is NOT a singlet -> Casimir > 0 (sanity)
    N, m, K = 2, 1.0, 3
    occ, _ = fock_ladder_ops(2 * (N * N - 1), K)
    psi = np.zeros(occ.shape[0])
    one = np.zeros(2 * (N * N - 1), dtype=np.int64); one[0] = 1
    idx = {tuple(occ[r]): r for r in range(occ.shape[0])}[tuple(one)]
    psi[idx] = 1.0
    c2 = casimir_of_ground_state(N, m, g=0.0, K=K, ground_state=psi)
    assert c2 > 1e-3
