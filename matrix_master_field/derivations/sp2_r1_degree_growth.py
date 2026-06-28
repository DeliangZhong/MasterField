"""R1 feasibility probe — degree growth of (ad_Â)^n H for QUADRATIC vs CUBIC Â.

Companion to derivations/sp2-r1-feasibility.md. Settles whether the BCH/Heisenberg
resummation e^{-iÂ}He^{iÂ}=Σ_n (1/n!)(ad_{-iÂ})^n H closes on a FINITE operator space.

Model (HHK arXiv:2004.10212 Eq 17), single Hermitian-matrix sector (X̃,P̃) — sufficient
to exercise the matrix CCR [X̃_ab,P̃_cd]=(i/N) δ_ad δ_bc and the degree-growth mechanism.
Reference |G⟩ = Gaussian g=0 ground state of Tr(P̃²+X̃²).

We measure degree growth in TWO independent, mutually-checking ways:

  (1) SYMBOLIC: track the polynomial-degree support of (ad_Â)^n H by Leibniz bookkeeping
      on the abstract word algebra (one canonical contraction lowers degree by 2). This is
      N-independent and exact — it is the actual obstruction.

  (2) FOCK (numerical cross-check, reusing exact_diag/qm_fock construction): build the
      nested commutators as sparse operators at finite N,K and read off their occupation
      support / operator-norm growth, confirming the symbolic count and that the QUADRATIC
      case's Heisenberg-evolved X̃(1),P̃(1) close LINEARLY (Bogoliubov), while the cubic
      case's operator norms blow up super-exponentially (no resummation).

Run:
  PYTHONPATH=<repo root> uv run --no-project --with numpy --with scipy python \
      matrix_master_field/derivations/sp2_r1_degree_growth.py
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, norm as spnorm

# Reuse the EXACT Fock construction used by sp2_flow_test.py / exact_diag.py / qm_fock.py.
import importlib.util as _ilu
import os as _os

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_spec = _ilu.spec_from_file_location("sp2_flow_test", _os.path.join(_HERE, "sp2_flow_test.py"))
_fl = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fl)
build = _fl.build
word_mat = _fl.word_mat
Tr_op = _fl.Tr_op
tau_op = _fl.tau_op
expval = _fl.expval


# ======================================================================
# PART (1) — SYMBOLIC degree growth (N-independent; the real obstruction)
# ======================================================================
# We do NOT need the full noncommutative word expansion to get the DEGREE
# support; degree is additive and one canonical contraction subtracts 2.
# Represent an operator by the SET of total-degrees present in it. For a
# product the degrees add; for [A,·] the leading term has degree
# deg(A)+deg(O)-2 (one contraction) and sub-leading terms have lower even
# offsets (more simultaneous contractions). The MAXIMAL degree is the one
# that matters for closure, and it is deg(A)+deg(O)-2 whenever A and O each
# contain at least one X̃ and one P̃ to contract (true for H here).

def max_degree_after_ad(deg_A, deg_H, n):
    """Max total polynomial degree of (ad_Â)^n H, from the contraction rule
    deg([A,O])_max = deg_A + deg_O - 2.  Closed form: deg_H + n*(deg_A - 2)."""
    return deg_H + n * (deg_A - 2)


def symbolic_degree_table():
    deg_H = 2  # the kinetic/mass density Tr(P̃²+X̃²) (the interaction −λτ([X̃,Ỹ]²) is deg 4;
    # using deg 4 only shifts the constant, not the GROWTH rate). Use 2 for the X-sector probe.
    print("  Max polynomial degree of (ad_Â)^n H   (rule: deg[A,O]_max = deg_A+deg_O-2)")
    print("  n :      Â quadratic (d=2)      Â cubic (d=3)       Â quartic (d=4)")
    for n in range(0, 9):
        q = max_degree_after_ad(2, deg_H, n)
        c = max_degree_after_ad(3, deg_H, n)
        f4 = max_degree_after_ad(4, deg_H, n)
        print(f"  {n:>2}:           {q:>4}                  {c:>4}                {f4:>4}")
    print("  ⟹ d=2: degree CONSTANT  → finite operator space → series RESUMS (Bogoliubov).")
    print("    d≥3: degree GROWS by (d-2)·n without bound → NO finite closure.")


# ======================================================================
# PART (2a) — FOCK cross-check of degree growth via occupation bandwidth
# ======================================================================
# In the Hermitian-mode bosonic Fock space, a degree-D polynomial in
# {X̃,P̃} = {(a+a†),(a−a†)} connects occupation number n to n±D (it is a
# sum of products of ≤ D ladder operators). So the OCCUPATION BANDWIDTH of
# the sparse operator (max |Δ total-quanta| with nonzero matrix element)
# equals the max polynomial degree. We read this bandwidth off the nested
# commutators directly — a numerical, N-dependent confirmation of Part (1).

def total_quanta_vector(N, K):
    """For build(N,K): the total-occupation number of each Fock basis state."""
    occ, a, idx, D = _fl.fock_ladders(N * N, K)
    return occ.sum(axis=1), D


def operator_bandwidth(Op, totals, tol=1e-9):
    """Max |total(row) − total(col)| over nonzero entries of sparse Op."""
    Op = Op.tocoo()
    if Op.nnz == 0:
        return 0
    mask = np.abs(Op.data) > tol
    if not mask.any():
        return 0
    dr = totals[Op.row[mask]] - totals[Op.col[mask]]
    return int(np.abs(dr).max())


def comm(A, B):
    return (A @ B - B @ A)


def fock_bandwidth_table(N=2, K=10, nmax=5):
    """Build H_X = Tr(P̃²+X̃²) and the nested commutators (ad_Â)^n H_X as sparse
    operators; report their occupation bandwidth = polynomial degree. K must be
    large enough that the bandwidth is not clipped by the truncation (we flag it)."""
    Xt, Pt, g, D = build(N, K)
    totals, _ = total_quanta_vector(N, K)
    # H in the X-sector: Tr(P̃² + X̃²)  (degree 2)
    H = Tr_op(Xt, Pt, (1, 1)) + Tr_op(Xt, Pt, (0, 0))
    # Generators:
    A_quad = N * Tr_op(Xt, Pt, (0, 0))                     # Â = N Tr(X̃²)   (quadratic, d=2)
    A_cub = N * Tr_op(Xt, Pt, (0, 0, 0))                   # Â = N Tr(X̃³)   (cubic, d=3)
    A_cub2 = (N / 2.0) * (Tr_op(Xt, Pt, (0, 0, 1))
                          + Tr_op(Xt, Pt, (1, 0, 0)))       # Â=(N/2)Tr(X̃²P̃+P̃X̃²) (cubic, d=3)

    def run(A, label):
        cur = H.copy()
        bws, nrm = [], []
        for n in range(nmax + 1):
            bw = operator_bandwidth(cur, totals)
            bws.append(bw)
            nrm.append(spnorm(cur))
            cur = comm(A, cur)
        clip = " (K-CLIPPED: raise K)" if max(bws) >= K else ""
        print(f"    {label}")
        print(f"      bandwidth(=poly degree) per n: {bws}{clip}")
        print(f"      ||(ad_Â)^n H||_F per n:        " + "  ".join(f"{v:.2e}" for v in nrm))
        return bws, nrm

    print(f"  Fock cross-check (N={N}, K={K}, D={D}); bandwidth = max |Δ total-quanta| = poly degree")
    bq, nq = run(A_quad, "Â = N·Tr(X̃²)            [QUADRATIC, d=2]")
    bc, nc = run(A_cub, "Â = N·Tr(X̃³)            [CUBIC,     d=3]")
    bc2, nc2 = run(A_cub2, "Â = (N/2)Tr(X̃²P̃+P̃X̃²)   [CUBIC,     d=3]")
    return {"quad": (bq, nq), "cub": (bc, nc), "cub2": (bc2, nc2)}


# ======================================================================
# PART (2b) — QUADRATIC Â closes: Heisenberg X̃(1),P̃(1) are LINEAR (Bogoliubov)
# ======================================================================
# For Â=(N/2)Tr(X̃P̃+P̃X̃) (a Hermitian quadratic generator = the dilatation),
# the Heisenberg map O(1)=e^{-iÂ}O e^{iÂ} must send X̃,P̃ to LINEAR combinations
# of X̃,P̃ (symplectic). We verify the matrix entries X̃_ab(1),P̃_ab(1) equal
# the Bogoliubov prediction to machine precision — i.e. the series literally
# resums, and the resulting state e^{iÂ}|G⟩ is Gaussian (a squeezed state).

def quadratic_closes(N=2, K=12, s=0.15):
    """Â=(N/2)Tr(X̃P̃+P̃X̃). DERIVED EOM (verified numerically): dX̃/ds=−X̃, dP̃/ds=+P̃
    (the dilatation/squeeze), so X̃(s)=e^{−s}X̃₀, P̃(s)=e^{+s}P̃₀ — EXACTLY linear at any N.
    The residual here is PURE Fock-K truncation (the e^{+s} growth pushes quanta past K);
    it shrinks at smaller s and larger K. The point: the operator series RESUMS to a linear
    (symplectic) map, so e^{iÂ}|G⟩ is a squeezed Gaussian — NOT a non-Gaussian improvement."""
    Xt, Pt, g, D = build(N, K)
    A = (N / 2.0) * (Tr_op(Xt, Pt, (0, 1)) + Tr_op(Xt, Pt, (1, 0)))
    A = A.tocsc()
    errX, errP = 0.0, 0.0
    for (a, b) in [(0, 0), (0, 1), (1, 1)]:
        X0 = Xt[a][b]
        P0 = Pt[a][b]
        # exact Heisenberg-evolved entry acting on |G⟩:  O(s)|G⟩ = e^{-isA} O e^{isA} |G⟩
        psiX = expm_multiply((-1j * s) * A, X0 @ expm_multiply((1j * s) * A, g))
        psiP = expm_multiply((-1j * s) * A, P0 @ expm_multiply((1j * s) * A, g))
        predX = np.exp(-s) * (X0 @ g)
        predP = np.exp(+s) * (P0 @ g)
        errX = max(errX, np.abs(psiX - predX).max())
        errP = max(errP, np.abs(psiP - predP).max())
    print(f"  QUADRATIC closure (N={N},K={K},s={s}): Heisenberg X̃,P̃ vs Bogoliubov e^{{∓s}}:")
    print(f"     max|X̃_ab(s)|G⟩ − e^{{−s}}X̃_ab|G⟩| = {errX:.2e}  (pure Fock-K truncation)")
    print(f"     max|P̃_ab(s)|G⟩ − e^{{+s}}P̃_ab|G⟩| = {errP:.2e}  (shrinks at smaller s / larger K)")
    print(f"     ⟹ e^{{iÂ}}|G⟩ is a SQUEEZED (Gaussian) state — series resums; no improvement.")
    return errX, errP


def quadratic_closes_Kconv(N=2, s=0.1, Klist=(8, 12, 16, 20)):
    """The residual in quadratic_closes is PURE Fock-K truncation: at fixed s it →0 as K→∞.
    (X̃₀₀ acting on |G⟩ excites quanta; the squeeze e^{±s} needs headroom above the support.)
    This certifies that the operator series genuinely resums to the linear Bogoliubov map."""
    print(f"  K-convergence of the QUADRATIC residual at fixed s={s}, N={N} (→0 ⟹ exact resummation):")
    for K in Klist:
        Xt, Pt, g, D = build(N, K)
        A = ((N / 2.0) * (Tr_op(Xt, Pt, (0, 1)) + Tr_op(Xt, Pt, (1, 0)))).tocsc()
        err = 0.0
        for (a, b) in [(0, 0), (0, 1)]:
            psiX = expm_multiply((-1j * s) * A, Xt[a][b] @ expm_multiply((1j * s) * A, g))
            err = max(err, np.abs(psiX - np.exp(-s) * (Xt[a][b] @ g)).max())
        print(f"     K={K:>2} (D={D:>5}):  max|X̃(s)|G⟩ − e^{{−s}}X̃|G⟩| = {err:.2e}")


# ======================================================================
# PART (2c) — CUBIC term coefficients of the BCH series GROW factorially
#             (the unbounded-generator non-convergence of the operator series)
# ======================================================================
# ⟨G|(ad_Â)^n H|G⟩ for Â=N Tr(X̃³): the Gaussian (Wick) expectations of the
# nested commutators. We tabulate |c_n|=|⟨G|(ad)^n H|G⟩|/n! and show the
# operator series Σ c_n is NOT obviously term-bounded (the per-order Wick
# expectation grows because the polynomial degree — hence the number/size of
# Wick pairings — grows with n). This documents the convergence concern for
# any naive operator-series truncation of the UNBOUNDED generator Â(P̃).

def bch_term_growth(N=2, K=18, nmax=10):
    """Two cubic generators, contrasted — the crux of Route 1(b).

    (i)  Â=N Tr(X̃³): the flow is AFFINE in P̃ (P̃→P̃+3sX̃²), so the *expectation*
         ⟨G|H(s)|G⟩ is a finite polynomial in s and the BCH expectation series happens to
         TERMINATE (a misleading special case — the OPERATOR series is still infinite,
         Part 2a). [residuals beyond n=2 are pure K-clipping, growing only as bandwidth>K.]

    (ii) Â=(N/2)Tr(X̃²P̃+P̃X̃²): the GENERIC cubic — nonlinear Riccati flow dX̃/ds=−X̃²,
         X̃(s)=X̃₀(1+sX̃₀)⁻¹. The BCH expectation series does NOT terminate and its terms
         GROW super-exponentially: the series in s has finite radius |s|<1/‖X̃₀‖, and X̃₀ is
         UNBOUNDED in |G⟩ (Gaussian spectrum), so the s=1 series is DIVERGENT. This is the
         convergence obstruction that defeats any naive operator-series truncation+remainder."""
    Xt, Pt, g, D = build(N, K)
    H = Tr_op(Xt, Pt, (1, 1)) + Tr_op(Xt, Pt, (0, 0))
    totals, _ = total_quanta_vector(N, K)
    gens = [
        ("Â = N Tr(X̃³)            [AFFINE — expectation series TERMINATES (special)]",
         N * Tr_op(Xt, Pt, (0, 0, 0))),
        ("Â = (N/2)Tr(X̃²P̃+P̃X̃²)   [GENERIC cubic — Riccati, series DIVERGES]",
         (N / 2.0) * (Tr_op(Xt, Pt, (0, 0, 1)) + Tr_op(Xt, Pt, (1, 0, 0)))),
    ]
    out = {}
    for label, A in gens:
        cur = H.copy()
        vals, bws = [], []
        for n in range(nmax + 1):
            vals.append(expval(g, cur).real)
            bws.append(operator_bandwidth(cur, totals))
            cur = comm(A, cur)
        print(f"  {label}")
        print(f"     ⟨(ad)^n H⟩ n=0..{nmax}:  " + "  ".join(f"{v:+.3e}" for v in vals))
        # ratio of successive NONZERO terms (odd ones vanish by parity) — growth diagnostic
        nz = [(n, v) for n, v in enumerate(vals) if abs(v) > 1e-6 and bws[n] < K]
        ratios = [abs(nz[k + 1][1] / nz[k][1]) for k in range(len(nz) - 1)]
        print(f"     |term ratio| (consecutive nonzero, pre-K-clip): "
              + ("  ".join(f"{r:.1f}" for r in ratios) if ratios else "n/a"))
        out[label] = vals
    return out


if __name__ == "__main__":
    print("=" * 78)
    print("PART 1 — SYMBOLIC degree growth (N-independent; the actual obstruction)")
    print("=" * 78)
    symbolic_degree_table()
    print()
    print("=" * 78)
    print("PART 2a — FOCK cross-check: occupation bandwidth = polynomial degree")
    print("=" * 78)
    fock_bandwidth_table(N=2, K=10, nmax=5)
    print()
    fock_bandwidth_table(N=3, K=8, nmax=4)
    print()
    print("=" * 78)
    print("PART 2b — QUADRATIC Â closes: Heisenberg map is LINEAR (Bogoliubov/squeezed)")
    print("=" * 78)
    quadratic_closes(N=2, K=12, s=0.15)
    print()
    quadratic_closes_Kconv(N=2, s=0.1, Klist=(8, 12, 16, 20))
    print()
    print("=" * 78)
    print("PART 2c — CUBIC BCH per-order growth (unbounded-generator convergence concern)")
    print("=" * 78)
    bch_term_growth(N=2, K=16, nmax=8)
