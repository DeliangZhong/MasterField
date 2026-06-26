# matrix_master_field/exact_diag.py
"""M5c follow-up — exact diagonalization of the two-matrix QM (ground truth).

Model (HHK arXiv:2004.10212 Eq 17):
    H = Tr(P_X^2 + P_Y^2 + m^2 (X^2 + Y^2) - g^2 [X,Y]^2),
X,Y Hermitian NxN (U(N), trace mode included), hbar=1, [X_ij,P_X,kl]=i delta_il delta_jk.
't Hooft coupling lambda = N g^2; report E/N^2.

Method: orthonormal Hermitian mode basis {T_a} -> 2N^2 decoupled oscillators + positive
quartic g^2 sum_c L_c^2 with L_c = sum_ab f_abc x_a y_b. Diagonalize over the bosonic Fock
space of the 2(N^2-1) traceless modes with total-quanta truncation sum n_i <= K (the 2 trace
modes add 2m analytically). See docs/superpowers/specs/2026-06-26-m5c-exact-diag-ground-truth-design.md.

Conventions are pinned in the plan's Global Constraints and CONVENTIONS.md. Float64 throughout.
"""
import numpy as np


def hermitian_basis(N):
    """Orthonormal Hermitian basis {T_a}, Tr(T_a T_b)=delta_ab, index 0 = I/sqrt(N).

    Generalized Gell-Mann basis: trace I/sqrt(N), then symmetric off-diagonal,
    antisymmetric off-diagonal, and diagonal (Cartan) traceless generators.
    """
    mats = [np.eye(N, dtype=np.complex128) / np.sqrt(N)]
    # symmetric and antisymmetric off-diagonal
    for j in range(N):
        for k in range(j + 1, N):
            S = np.zeros((N, N), dtype=np.complex128)
            S[j, k] = S[k, j] = 1.0 / np.sqrt(2.0)
            mats.append(S)
            A = np.zeros((N, N), dtype=np.complex128)
            A[j, k] = -1j / np.sqrt(2.0)
            A[k, j] = 1j / np.sqrt(2.0)
            mats.append(A)
    # diagonal Cartan generators D_l, l=1..N-1
    for l in range(1, N):
        D = np.zeros((N, N), dtype=np.complex128)
        for j in range(l):
            D[j, j] = 1.0
        D[l, l] = -l
        D = D / np.sqrt(l * (l + 1))
        mats.append(D)
    return np.stack(mats, axis=0)


def structure_constants(N):
    """f[a,b,c] = -i Tr([T_a,T_b] T_c), real and totally antisymmetric."""
    T = hermitian_basis(N)
    n = N * N
    f = np.zeros((n, n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            comm = T[a] @ T[b] - T[b] @ T[a]
            for c in range(n):
                val = -1j * np.trace(comm @ T[c])
                assert abs(val.imag) < 1e-10, "f must be real"
                f[a, b, c] = val.real
    return f


def quartic_potential_value(N, x_vec, y_vec):
    """sum_c (sum_ab f_abc x_a y_b)^2  ==  -Tr[X,Y]^2 (classical, c-number x,y)."""
    f = structure_constants(N)
    L = np.einsum("abc,a,b->c", f, np.asarray(x_vec, float), np.asarray(y_vec, float))
    return float(np.dot(L, L))
