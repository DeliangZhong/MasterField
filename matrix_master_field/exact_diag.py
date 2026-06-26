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
import scipy.sparse.linalg as spla


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


def build_two_matrix_qm_hamiltonian(N, m, g, K, pad=2):
    """Exact Galerkin projection P_K H P_K of the interacting H over the 2(N^2-1)-mode Fock
    space, truncated to total quanta <= K (the 2m trace contribution is added by ground_energy).

    L_c = sum_ab f_abc x_a y_b is Hermitian in the FULL space (x_a, y_b act on disjoint mode
    sets). Building it from K-truncated operators introduces a spurious middle projector that
    breaks Hermiticity at the boundary and makes L_c^2 non-PSD. Fix (as in qm_fock.py): build on
    the padded (K+pad) basis so the quartic's intermediates (total <= K+2) are represented, then
    restrict to the canonical total<=K basis. The <=K block of L_c^2-built-on-(K+pad) equals
    P_K L_c^2 P_K exactly -> Hermitian, PSD, variational upper bound monotone in K. pad=2 suffices.
    Returned H is ordered to match occupation_basis(n_modes, K) so ground_energy/casimir agree.
    """
    n_tl = N * N - 1
    n_modes = 2 * n_tl
    Kp = K + pad

    occ_p, ops_p = fock_ladder_ops(n_modes, Kp)
    s2m = np.sqrt(2.0 * m)
    xs = [None] + [(ops_p[a - 1] + ops_p[a - 1].transpose()) / s2m for a in range(1, n_tl + 1)]
    ys = [None] + [(ops_p[n_tl + (b - 1)] + ops_p[n_tl + (b - 1)].transpose()) / s2m
                   for b in range(1, n_tl + 1)]

    total_p = occ_p.sum(axis=1)
    H = sp.diags(m * (2.0 * total_p + n_modes), format="csr", dtype=np.float64)

    if g != 0.0:
        f = structure_constants(N)
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
                H = H + (g * g) * (Lc @ Lc)

    occ_K = occupation_basis(n_modes, K)
    base_p = Kp + 1
    idx_p = {_radix_key(occ_p[r], base_p): r for r in range(occ_p.shape[0])}
    keep = np.array([idx_p[_radix_key(occ_K[t], base_p)] for t in range(occ_K.shape[0])],
                    dtype=np.int64)
    H = H.tocsr()[keep][:, keep]
    H = 0.5 * (H + H.transpose())
    return H.tocsr()


def converge_in_K(N, m, g, K_list):
    """E/N^2 vs truncation K (variational, non-increasing). Returns series, value, tail."""
    series = []
    for K in K_list:
        series.append((K, ground_energy(N, m, g, K)["E_over_N2"]))
    vals = [e for _, e in series]
    tail = abs(vals[-1] - vals[-2]) if len(vals) >= 2 else float("inf")
    return {"series": series, "value": vals[-1], "tail": tail}


def ground_energy(N, m, g, K):
    """Ground-state energy density E/N^2 of the two-matrix QM by exact diagonalization.

    The interacting Hamiltonian H covers the 2(N^2-1) traceless modes (truncated to total
    quanta <= K). The two trace modes each contribute m (their zero-point energy) analytically,
    so E = E_interacting + 2m.

    Eigensolver: dense np.linalg.eigh when D < 50 (avoids Lanczos failures on tiny matrices),
    otherwise sparse eigsh(k=1, which='SA') with a fixed v0=ones/sqrt(D) for reproducibility.
    The ground-state vector is returned in occupation_basis(2(N^2-1), K) order (canonical order
    produced by build_two_matrix_qm_hamiltonian), as required by Task 8 (Casimir).
    """
    H = build_two_matrix_qm_hamiltonian(N, m, g, K)
    D = H.shape[0]
    if D < 50:
        w, v = np.linalg.eigh(H.toarray())
        e_int = float(w[0])
        gs = v[:, 0]
    else:
        v0 = np.ones(D) / np.sqrt(D)
        vals, vecs = spla.eigsh(H, k=1, which="SA", v0=v0)
        e_int = float(vals[0])
        gs = vecs[:, 0]
    E = e_int + 2.0 * m   # two trace modes, each ground energy m
    return {
        "E_over_N2": E / (N * N),
        "E": E,
        "E_interacting": e_int,
        "K": K,
        "basis_dim": D,
        "n_modes": 2 * (N * N - 1),
        "ground_state": np.asarray(gs, dtype=np.float64),
    }
