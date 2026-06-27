"""Decisive numerical test of single-trace moment-flow CLOSURE (S1) for the matrix
master field — companion to derivations/sp2-moment-flow-closure.md.

Single Hermitian matrix X̃ + conjugate P̃ on a bosonic Fock space (the 'X' sector of
HHK Eq 17, arXiv:2004.10212), with the 't Hooft-scaled CCR [X̃_ab,P̃_cd]=(i/N) δ_ad δ_bc.
Reference |G> = Gaussian ground state of Tr(P̃²+X̃²) (m=1) — the g=0 master-field state.

We compute the EXACT finite-N Heisenberg flow τ_s[w]=<G|e^{-isÂ}(τ w)e^{isÂ}|G> by acting
with expm_multiply on the state, and compare to the PLANAR (large-N) moment-flow ODE.

The variational generator carries an explicit factor N (DERIVED in the writeup):
Â = N·Tr(single-trace word), so that dτ[w]/ds = O(1).

Reuses the bosonic-Fock construction (Hermitian mode basis + occupation-truncated
ladder operators) of matrix_master_field/exact_diag.py and qm_fock.py.

Run:
  uv run --no-project --with numpy --with scipy python sp2_flow_test.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


# ------------------------- Fock construction (sparse) -------------------------
def hermitian_basis(N):
    """Orthonormal Hermitian basis, Tr(T_a T_b)=δ_ab (generalized Gell-Mann; same as
    exact_diag.hermitian_basis). Index 0 = I/√N (the trace mode)."""
    mats = [np.eye(N, dtype=complex) / np.sqrt(N)]
    for j in range(N):
        for k in range(j + 1, N):
            S = np.zeros((N, N), complex); S[j, k] = S[k, j] = 1 / np.sqrt(2); mats.append(S)
            A = np.zeros((N, N), complex); A[j, k] = -1j / np.sqrt(2); A[k, j] = 1j / np.sqrt(2); mats.append(A)
    for l in range(1, N):
        D = np.zeros((N, N), complex)
        for j in range(l):
            D[j, j] = 1.0
        D[l, l] = -l; D /= np.sqrt(l * (l + 1)); mats.append(D)
    return np.stack(mats, 0)


def fock_ladders(n_modes, K):
    """Occupation basis (Σ n_i ≤ K) + sparse annihilation operators a_i (a_i† = a[i].getH())."""
    rows = []

    def rec(pre, rem):
        if len(pre) == n_modes:
            rows.append(pre); return
        for v in range(rem + 1):
            rec(pre + (v,), rem - v)

    rec((), K)
    occ = np.array(rows, dtype=int); D = len(occ)
    idx = {tuple(o): r for r, o in enumerate(occ)}
    a = []
    for i in range(n_modes):
        r, c, d = [], [], []
        for col in range(D):
            ni = occ[col, i]
            if ni > 0:
                t = occ[col].copy(); t[i] -= 1
                r.append(idx[tuple(t)]); c.append(col); d.append(np.sqrt(ni))
        a.append(sp.csr_matrix((d, (r, c)), shape=(D, D), dtype=complex))
    return occ, a, idx, D


def build(N, K):
    """Matrix-valued X̃,P̃ as N×N lists of sparse D×D operators, the Gaussian reference
    state g, and the Fock dim D. x_a=(a+a†)/√2, p_a=−i(a−a†)/√2 (frequency m=1), then
    X̃_ij=(1/√N) Σ_a (T_a)_ij x_a (the 1/√N is the 't Hooft scaling X̃=X/√N)."""
    T = hermitian_basis(N); n = N * N
    occ, a, idx, D = fock_ladders(n, K)
    xm = [(a[i] + a[i].T) / np.sqrt(2) for i in range(n)]
    pm = [(-1j * (a[i] - a[i].T)) / np.sqrt(2) for i in range(n)]
    sN = np.sqrt(N)
    Xt = [[sp.csr_matrix((D, D), dtype=complex) for _ in range(N)] for _ in range(N)]
    Pt = [[sp.csr_matrix((D, D), dtype=complex) for _ in range(N)] for _ in range(N)]
    for am in range(n):
        Ta = T[am]
        for i in range(N):
            for j in range(N):
                c = Ta[i, j]
                if c != 0:
                    Xt[i][j] = Xt[i][j] + (c / sN) * xm[am]
                    Pt[i][j] = Pt[i][j] + (c / sN) * pm[am]
    g = np.zeros(D, complex); g[idx[(0,) * n]] = 1.0
    return Xt, Pt, g, D


# ----------------- word -> sparse Tr operator (matrix-valued product) ----------------
def _ident_mat(Xt):
    N = len(Xt); D = Xt[0][0].shape[0]
    I = sp.identity(D, dtype=complex, format='csr'); Z = sp.csr_matrix((D, D), dtype=complex)
    return [[I if i == j else Z for j in range(N)] for i in range(N)]


def word_mat(Xt, Pt, word):
    """N×N array of sparse D×D operators = product of letters along the matrix index
    (0=X̃, 1=P̃). Operators on disjoint matrix entries do NOT commute (CCR), so order
    is preserved throughout — no cyclicity is ever used."""
    N = len(Xt)
    if len(word) == 0:
        return _ident_mat(Xt)
    letters = [Xt if c == 0 else Pt for c in word]
    cur = letters[0]
    for nxt in letters[1:]:
        new = [[None] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                acc = None
                for k in range(N):
                    term = cur[i][k] @ nxt[k][j]
                    acc = term if acc is None else acc + term
                new[i][j] = acc
        cur = new
    return cur


def Tr_op(Xt, Pt, word):
    cur = word_mat(Xt, Pt, word); N = len(Xt)
    acc = cur[0][0]
    for i in range(1, N):
        acc = acc + cur[i][i]
    return acc


def tau_op(Xt, Pt, word):
    return Tr_op(Xt, Pt, word) / len(Xt)


def expval(g, Op):
    return complex(g.conj() @ (Op @ g))


def gen_TrX3(Xt, Pt):
    return len(Xt) * Tr_op(Xt, Pt, (0, 0, 0))                                   # Â = N Tr(X̃³)


def gen_X2P_sym(Xt, Pt):
    return (len(Xt) / 2.0) * (Tr_op(Xt, Pt, (0, 0, 1)) + Tr_op(Xt, Pt, (1, 0, 0)))  # Â=(N/2)Tr(X̃²P̃+P̃X̃²)


def herm_err(A):
    d = (A - A.getH()); return abs(d).max() if d.nnz else 0.0


def exact_flow(Xt, Pt, g, A, word, svals):
    """τ_s[word] via |ψ(s)>=e^{isA}|G>:  O(s)=e^{-isA}O e^{isA} ⟹ <O>(s)=<ψ(s)|O|ψ(s)>."""
    Op = tau_op(Xt, Pt, word).tocsc(); A = A.tocsc()
    out = []
    for s in svals:
        psi = expm_multiply((1j * s) * A, g)
        out.append(complex(psi.conj() @ (Op @ psi)))
    return np.array(out)


# ====================================================================== TEST A
# Â = N Tr(X̃³).  DERIVED (exact, finite N): dX̃/ds=0, dP̃/ds=3X̃² ⟹ P̃(s)=P̃₀+3sX̃².
#   τ(P̃²)(s) = τ(P̃²) + 3s[τ(P̃X̃²)+τ(X̃²P̃)] + 9 s² τ(X̃⁴).      (closed, single-trace)
#   τ(X̃²)(s) frozen.   Validates generator normalization N¹, sign, and machinery.
def test_A(N, K):
    Xt, Pt, g, D = build(N, K)
    A = gen_TrX3(Xt, Pt)
    svals = np.linspace(0, 0.15, 7)
    fP2 = exact_flow(Xt, Pt, g, A, (1, 1), svals)
    fX2 = exact_flow(Xt, Pt, g, A, (0, 0), svals)
    tP2 = expval(g, tau_op(Xt, Pt, (1, 1)))
    tPX2 = expval(g, tau_op(Xt, Pt, (1, 0, 0)))
    tX2P = expval(g, tau_op(Xt, Pt, (0, 0, 1)))
    tX4 = expval(g, tau_op(Xt, Pt, (0, 0, 0, 0)))
    pred = tP2 + 3 * svals * (tPX2 + tX2P) + 9 * svals ** 2 * tX4
    eP2 = np.max(np.abs(fP2 - pred)); eX2 = np.max(np.abs(fX2 - fX2[0]))
    print(f"  [A] N={N} K={K} D={D}  herm(Â)={herm_err(A):.0e}  "
          f"max|exact τ(P̃²) − closed-form ODE|={eP2:.2e}  frozen|Δτ(X̃²)|={eX2:.2e}")
    return eP2, eX2


# ====================================================================== TEST B
# Â=(N/2)Tr(X̃²P̃+P̃X̃²). DERIVED: dX̃/ds=−X̃² (matrix Riccati) ⟹ X̃(s)=X̃₀(1+sX̃₀)⁻¹.
#   Planar single-trace ODE:  dτ(X̃^k)/ds = −k τ(X̃^{k+1}).   (closed, single-trace, linear)
# Compares the EXACT finite-N flow of τ(X̃²) to this ODE (the closure stress-test on a
# genuinely NONLINEAR canonical flow).
def test_B(N, K, kmax=12):
    Xt, Pt, g, D = build(N, K)
    A = gen_X2P_sym(Xt, Pt)
    svals = np.linspace(0, 0.10, 6)
    fX2 = exact_flow(Xt, Pt, g, A, (0, 0), svals)
    m0 = np.array([expval(g, tau_op(Xt, Pt, (0,) * k)).real for k in range(kmax + 2)])

    def rhs(mv):
        d = np.zeros_like(mv)
        for k in range(1, kmax + 1):
            d[k] = -k * mv[k + 1]
        return d

    def integ(s, nst=300):
        mv = m0.copy(); h = s / nst
        for _ in range(nst):
            k1 = rhs(mv); k2 = rhs(mv + h / 2 * k1); k3 = rhs(mv + h / 2 * k2); k4 = rhs(mv + h * k3)
            mv = mv + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return mv[2]

    pred = np.array([integ(s) for s in svals])
    eODE = np.max(np.abs(fX2.real - pred))
    print(f"  [B] N={N} K={K} D={D}  herm(Â)={herm_err(A):.0e}  "
          f"max|exact τ(X̃²) − planar ODE (kmax={kmax})|={eODE:.2e}  "
          f"(end: exact={fX2[-1].real:.5f} ODE={pred[-1]:.5f})")
    return eODE


# ====================================================================== TEST C
# Connected double-trace C_N(w1,w2)=⟨τ(w1)τ(w2)⟩−⟨τ(w1)⟩⟨τ(w2)⟩ in |G>.  Large-N
# FACTORIZATION predicts C_N = O(1/N²).  This is WHY the moment-flow RHS (which contains
# genuine Tr·Tr from two simultaneous CCR firings) closes on PRODUCTS of single-traces.
def test_C(N_K_list):
    pairs = [((0, 0), (0, 0)), ((0, 0), (0, 0, 0, 0)), ((1, 1), (1, 1))]
    names = ["τ(X̃²)τ(X̃²)", "τ(X̃²)τ(X̃⁴)", "τ(P̃²)τ(P̃²)"]
    print("      connected C_N = ⟨τ(w1)τ(w2)⟩ − ⟨τ(w1)⟩⟨τ(w2)⟩   pairs: " + ", ".join(names))
    rows = {}
    for N, K in N_K_list:
        Xt, Pt, g, D = build(N, K)
        row = []
        for w1, w2 in pairs:
            O1 = tau_op(Xt, Pt, w1); O2 = tau_op(Xt, Pt, w2)
            row.append((expval(g, O1 @ O2) - expval(g, O1) * expval(g, O2)).real)
        rows[(N, K)] = row
        print(f"        N={N} (K={K},D={D}):  " + "  ".join(f"{v:+.3e}" for v in row))
    print("      C_N · N²  (flat across N ⟹ C_N ∝ 1/N²  ⟹ factorization holds):")
    for (N, K), row in rows.items():
        print(f"        N={N}:  " + "  ".join(f"{v * N * N:+.3e}" for v in row))
    return rows


# ====================================================================== TEST D
# K-CONVERGENCE of the exact flow at fixed N: separates Fock-TRUNCATION error from the
# genuine finite-N closure gap. We track τ(P̃²)(s_end) for Test A across K; it must
# converge (in K) to the closed-form ODE value as K→∞ AT FIXED N (the ODE value is
# K-independent), pinning the residual as a pure finite-N effect.
def test_D_convergence(N, Klist, s_end=0.15):
    print(f"      K-convergence of EXACT τ(P̃²)(s={s_end}) at N={N} vs the (K-independent) ODE value:")
    for K in Klist:
        Xt, Pt, g, D = build(N, K)
        A = gen_TrX3(Xt, Pt)
        fP2 = exact_flow(Xt, Pt, g, A, (1, 1), [s_end])[0].real
        tP2 = expval(g, tau_op(Xt, Pt, (1, 1)))
        tPX2 = expval(g, tau_op(Xt, Pt, (1, 0, 0))); tX2P = expval(g, tau_op(Xt, Pt, (0, 0, 1)))
        tX4 = expval(g, tau_op(Xt, Pt, (0, 0, 0, 0)))
        ode = (tP2 + 3 * s_end * (tPX2 + tX2P) + 9 * s_end ** 2 * tX4).real
        print(f"        K={K:>2} (D={D:>4}):  exact={fP2:.6f}   ODE={ode:.6f}   |diff|={abs(fP2-ode):.2e}")


# ====================================================================== TEST E
# GENERATOR NORMALIZATION: with the DERIVED generator Â=N·Tr(â) (=N²·τ(â)), the
# initial flow derivative dτ(w)/ds|₀ = i⟨[τ(w),Â]⟩ = i⟨[Tr w, Tr â]⟩ must be O(1) AND
# converge to an N-INDEPENDENT planar value. (With Â=Tr(â), i.e. no explicit N, it would
# instead be O(1/N) → 0.) This pins the audit's missing N power.
def test_E_normalization(N_K_list):
    cases = [((0,), (1,)), ((1,), (0, 0, 0)), ((0, 0), (0, 1)),
             ((1, 1), (0, 1)), ((0, 1), (0, 0))]
    print("      dτ(w)/ds|₀ = i⟨[Tr w, Tr â]⟩ with Â=N·Tr(â)  (must be O(1), N-independent):")
    print("        " + "  ".join(f"N={N}" for N, _ in N_K_list) + "   (w; â)")
    caches = {N: build(N, K) for N, K in N_K_list}
    for w, a in cases:
        vals = []
        for N, _ in N_K_list:
            Xt, Pt, g, _ = caches[N]
            v = 1j * expval(g, (Tr_op(Xt, Pt, w) @ Tr_op(Xt, Pt, a)
                                - Tr_op(Xt, Pt, a) @ Tr_op(Xt, Pt, w)))
            vals.append(v.real if abs(v.real) > abs(v.imag) else v.imag)
        flat = max(abs(v - vals[-1]) for v in vals)
        tag = f"planar={vals[-1]:+.4f} (N-indep, Δ={flat:.0e})" if flat < 5e-2 else f"N-DEP Δ={flat:.1e}"
        print(f"        " + "  ".join(f"{v:+.4f}" for v in vals) + f"   ({w}; {a})  {tag}")


# ====================================================================== TEST F
# DERIVED matrix EOMs verified directly against the exact operator flow of the matrix
# ENTRIES (signs + structure, end-to-end):
#   Test A: Â=N Tr(X̃³)            ⟹ dP̃/ds = +3 X̃²,  X̃ frozen
#   Test B: Â=(N/2)Tr(X̃²P̃+P̃X̃²)  ⟹ dX̃/ds = −X̃²
def test_F_eom(N=2, K=7, eps=1e-4):
    from scipy.sparse.linalg import expm_multiply
    Xt, Pt, g, D = build(N, K)
    X2 = word_mat(Xt, Pt, (0, 0))

    def dval(A, Op):  # d/ds ⟨G|e^{-isA} Op e^{isA}|G⟩ at s=0 (central difference)
        A = A.tocsc(); Op = Op.tocsc()

        def v(s):
            psi = expm_multiply((1j * s) * A, g)
            return complex(psi.conj() @ (Op @ psi))

        return (v(eps) - v(-eps)) / (2 * eps)

    A_TA = gen_TrX3(Xt, Pt)
    eP = max(abs(dval(A_TA, Pt[a][b]) - 3 * expval(g, X2[a][b])) for a, b in [(0, 0), (0, 1), (1, 1)])
    eX = max(abs(dval(A_TA, Xt[a][b])) for a, b in [(0, 0), (0, 1)])
    A_TB = gen_X2P_sym(Xt, Pt)
    eB = max(abs(dval(A_TB, Xt[a][b]) + expval(g, X2[a][b])) for a, b in [(0, 0), (0, 1), (1, 1)])
    print(f"  N={N} K={K}:  [A] max|d⟨P̃⟩/ds − 3⟨X̃²⟩|={eP:.1e}  X̃-frozen max|d⟨X̃⟩/ds|={eX:.1e}")
    print(f"            [B] max|d⟨X̃⟩/ds + ⟨X̃²⟩|={eB:.1e}   (DERIVED EOM signs confirmed)")


if __name__ == "__main__":
    print("=" * 78)
    print("TEST A — Â=N Tr(X̃³): single-trace closure + generator normalization (N¹) + sign")
    print("=" * 78)
    test_A(2, 6); test_A(2, 8); test_A(3, 4)
    print()
    print("=" * 78)
    print("TEST D — K-convergence at fixed N (isolates truncation error from finite-N gap)")
    print("=" * 78)
    test_D_convergence(2, [4, 6, 8, 10])
    test_D_convergence(3, [2, 3, 4])
    print()
    print("=" * 78)
    print("TEST B — Â=(N/2)Tr(X̃²P̃+P̃X̃²): nonlinear flow dX̃/ds=−X̃², planar single-trace ODE")
    print("=" * 78)
    test_B(2, 6); test_B(2, 8); test_B(3, 4)
    print()
    print("=" * 78)
    print("TEST C — connected-double-trace factorization (the closure mechanism, ∝1/N²)")
    print("=" * 78)
    test_C([(2, 6), (3, 4), (4, 3)])
    print()
    print("=" * 78)
    print("TEST E — generator normalization Â=N·Tr(â): dτ(w)/ds|₀ is O(1) and N-independent")
    print("=" * 78)
    test_E_normalization([(2, 5), (3, 4), (4, 3), (5, 2)])
    print()
    print("=" * 78)
    print("TEST F — derived matrix EOMs (signs) vs exact operator flow of matrix entries")
    print("=" * 78)
    test_F_eom(2, 7)
