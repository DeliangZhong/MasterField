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
import scipy.sparse as sp


def occupation_basis(n_modes, K):
    """All occupation tuples (n_0,...,n_{n_modes-1}), n_i>=0, sum n_i <= K."""
    rows = []

    def rec(prefix, remaining):
        if len(prefix) == n_modes:
            rows.append(prefix)
            return
        for v in range(remaining + 1):
            rec(prefix + (v,), remaining - v)

    rec((), K)
    return np.array(rows, dtype=np.int64)


def _radix_key(occ_row, base):
    """Mixed-radix integer encoding of an occupation row (base = K+1) for O(1) lookup."""
    key = 0
    for v in occ_row[::-1]:
        key = key * base + int(v)
    return key


def fock_ladder_ops(n_modes, K):
    """(occ, ops): occupation basis and sparse annihilation operators a_i (a_i^dag = ops[i].T)."""
    occ = occupation_basis(n_modes, K)
    D = occ.shape[0]
    base = K + 1
    index = {_radix_key(occ[r], base): r for r in range(D)}
    ops = []
    for i in range(n_modes):
        rows, cols, data = [], [], []
        for c in range(D):
            ni = occ[c, i]
            if ni > 0:
                tgt = occ[c].copy()
                tgt[i] -= 1
                rows.append(index[_radix_key(tgt, base)])
                cols.append(c)
                data.append(np.sqrt(float(ni)))
        ops.append(sp.csr_matrix((data, (rows, cols)), shape=(D, D), dtype=np.float64))
    return occ, ops


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


def build_two_matrix_qm_hamiltonian(N, m, g, K):
    """Interacting H over the 2(N^2-1)-mode Fock space (trace 2m added by ground_energy).

    H = m*(2*sum_i n_i + n_modes) + g^2 * sum_c L_c^2
    where L_c = sum_{a,b=1}^{N^2-1} f[a,b,c] x_a y_b,
    x_a = (ops[a-1] + ops[a-1]^dag) / sqrt(2m),
    y_b = (ops[n_tl+(b-1)] + ops[n_tl+(b-1)]^dag) / sqrt(2m).

    Mode layout: x-modes a=1..n_tl -> ladder slot a-1; y-modes b=1..n_tl -> slot n_tl+(b-1).
    """
    n_tl = N * N - 1            # traceless modes per matrix
    n_modes = 2 * n_tl
    occ, ops = fock_ladder_ops(n_modes, K)
    s2m = np.sqrt(2.0 * m)

    def x_op(a):  # a = 1..n_tl
        A = ops[a - 1]
        return (A + A.transpose()) / s2m

    def y_op(b):  # b = 1..n_tl
        A = ops[n_tl + (b - 1)]
        return (A + A.transpose()) / s2m

    # free part: m * (2 sum n_i + n_modes), diagonal
    total = occ.sum(axis=1)
    H = sp.diags(m * (2.0 * total + n_modes), format="csr", dtype=np.float64)

    if g != 0.0:
        f = structure_constants(N)
        xs = [None] + [x_op(a) for a in range(1, n_tl + 1)]
        ys = [None] + [y_op(b) for b in range(1, n_tl + 1)]
        for c in range(1, n_tl + 1):
            terms = []
            for a in range(1, n_tl + 1):
                for b in range(1, n_tl + 1):
                    fabc = f[a, b, c]
                    if fabc != 0.0:
                        terms.append(fabc * (xs[a] @ ys[b]))
            if terms:
                Lc = terms[0]
                for t in terms[1:]:
                    Lc = Lc + t
                # Symmetrize: in exact theory [x_a, y_b]=0 so L_c is Hermitian;
                # truncation breaks this at the K-quanta boundary.
                # Use (L_c + L_c^dag)/2 to restore Hermiticity before squaring.
                Lc = 0.5 * (Lc + Lc.transpose())
                H = H + (g * g) * (Lc @ Lc)

    H = 0.5 * (H + H.transpose())
    return H.tocsr()
