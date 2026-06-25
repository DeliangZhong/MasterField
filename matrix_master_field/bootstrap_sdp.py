"""
bootstrap_sdp.py — Semidefinite programming bootstrap for moment bounds.

Uses cvxpy to solve:
    min/max  m_k
    subject to:
        - Schwinger-Dyson equations (linear in moments)
        - Moment matrix Ω ≽ 0 (semidefinite constraint)
        - m_0 = 1 (normalisation)

This provides RIGOROUS bounds on the moments, against which the ML
solution can be validated.
"""

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("cvxpy not installed — SDP bootstrap unavailable")


def _mosek_usable():
    """True iff MOSEK is installed AND licensed (a trivial solve actually succeeds).

    cvxpy lists MOSEK in `installed_solvers()` whenever the package merely imports —
    even with NO license — but it then fails at solve time, and the fail-closed gate
    would silently fall back to the untrusted SCS bound. We probe a real solve so an
    installed-but-unlicensed MOSEK is never selected as a *trusted* certifier.
    """
    try:
        x = cp.Variable()
        cp.Problem(cp.Minimize(cp.sum_squares(x - 1))).solve(solver=cp.MOSEK)
        return True
    except Exception:
        return False


def _select_solver():
    """Highest-accuracy *usable* conic solver: MOSEK (if licensed) > CLARABEL > SCS.

    CLARABEL/MOSEK are interior-point (≈1e-8 tolerances) and return certified
    'optimal' status; SCS is a first-order fallback that flags 'optimal_inaccurate'
    on the larger (L≥8) two-matrix moment relaxations. Install `clarabel` (no
    license; also bundled with cvxpy) or `mosek` (academic license) to certify the
    tight islands. An installed-but-unlicensed MOSEK is skipped (see `_mosek_usable`).
    """
    if not HAS_CVXPY:
        return None
    installed = set(cp.installed_solvers())
    for s in ("MOSEK", "CLARABEL", "SCS"):
        if s not in installed:
            continue
        if s == "MOSEK" and not _mosek_usable():
            continue
        return s
    return None


SOLVER = _select_solver()
TRUSTED_SOLVERS = ("MOSEK", "CLARABEL")  # interior-point; their 'optimal' status certifies a bound
_LAST_SOLVE = {"solver": None, "status": None}  # which solver/status produced the last bound


def has_trusted_solver():
    """True iff a certifying interior-point solver (CLARABEL/MOSEK) is installed.

    A bound is only treated as a *certified* island edge when it comes from a
    trusted solver with status 'optimal' (not 'optimal_inaccurate', and not the SCS
    fallback). Without one, `solve_two_matrix(_sparse)` cannot set validated=True.
    """
    return SOLVER in TRUSTED_SOLVERS

# The two-matrix moment relaxations are degenerate (no strictly-interior point), so
# default-CLARABEL fails on several instances (e.g. g=1, L=10). A small static
# regularization stabilizes the KKT factorization and recovers a certified
# 'optimal' there (verified: g=1 L=10 lb 0.69307); the perturbation is ≤1e-3.
_CLARABEL_KW = {"max_iter": 5000, "static_regularization_constant": 1e-7}


def _solve(problem):
    """Solve `problem` with the best available solver, falling back to SCS.

    Tries the high-accuracy solver first (CLARABEL/MOSEK); on a solver error or a
    non-optimal status it falls back to SCS (always installed, robust on the larger
    moment relaxations). `_LAST_SOLVE` records the solver and status that produced
    the returned value, for transparency about which bounds are interior-point
    certified vs SCS estimates.
    """
    order = ([SOLVER] if SOLVER and SOLVER != "SCS" else []) + ["SCS"]
    for s in order:
        try:
            if s == "SCS":
                problem.solve(solver=cp.SCS, max_iters=20000)
            elif s == "CLARABEL":
                problem.solve(solver=cp.CLARABEL, **_CLARABEL_KW)
            else:  # MOSEK or other interior-point solver — robust defaults
                problem.solve(solver=getattr(cp, s))
        except Exception:
            _LAST_SOLVE.update(solver=s, status="error")
            continue
        _LAST_SOLVE.update(solver=s, status=problem.status)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return problem
    return problem


def bootstrap_one_matrix(
    v_prime_coeffs: list[float], max_moment: int = 10, target_moment: int = 2, maximize: bool = True
) -> float | None:
    """Bootstrap bound on m_{target_moment} for a one-matrix model.

    Args:
        v_prime_coeffs: [v_0, v_1, ...] so V'(M) = Σ v_k M^k
        max_moment: highest moment in the truncation
        target_moment: which moment to bound
        maximize: if True, find upper bound; if False, lower bound

    Returns:
        The bound (upper or lower) on m_{target_moment}
    """
    if not HAS_CVXPY:
        return None

    K = max_moment
    n_v = len(v_prime_coeffs)

    # Variables: m_1, m_2, ..., m_K (m_0 = 1 fixed)
    m = cp.Variable(K + 1, name="moments")

    constraints = []

    # m_0 = 1
    constraints.append(m[0] == 1.0)

    # Z₂ symmetry: m_{odd} = 0 (for symmetric potentials with v_{even}=0)
    is_symmetric = all(v_prime_coeffs[k] == 0 for k in range(0, n_v, 2))
    if is_symmetric:
        for k in range(1, K + 1, 2):
            constraints.append(m[k] == 0.0)

    # Hankel (moment) matrix H_{ij} = m_{i+j} ⪰ 0 — positivity of the eigenvalue
    # measure (the classical Hamburger moment condition).
    half_K = max(K - max(n_v, 1), K // 2)
    H = cp.Variable((half_K + 1, half_K + 1), symmetric=True, name="Hankel")
    for i in range(half_K + 1):
        for j in range(i, half_K + 1):
            if i + j <= K:
                constraints.append(H[i, j] == m[i + j])
    constraints.append(H >> 0)

    # Product matrix Q_{jk} relaxing the bilinear m_j·m_k (Lasserre rank-1
    # relaxation). The large-N SD RHS is Σ_j m_j m_{n-1-j} — a PRODUCT of moments,
    # NOT the Hankel entry m_{j+(n-1-j)}. Encode it linearly: Q symmetric,
    # Q[0,k] = m_k (since m_0 = 1), Q ⪰ 0 ⇒ Q ⪰ m mᵀ. The exact solution
    # Q = m mᵀ is feasible, so optimizing a moment over this set brackets it.
    qd = max(K, 1)  # indices 0..K-1 cover every loop-equation split
    Q = cp.Variable((qd, qd), symmetric=True, name="Products")
    for k in range(qd):
        constraints.append(Q[0, k] == m[k])
    constraints.append(Q >> 0)

    # SD equations: Σ_k v_k m_{n+k} = Σ_{j=0}^{n-1} Q[j, n-1-j].
    for n in range(0, max(1, K - n_v + 1)):
        lhs = 0
        for k in range(n_v):
            if 0 <= n + k <= K:
                lhs += v_prime_coeffs[k] * m[n + k]
        if all(j < qd and (n - 1 - j) < qd for j in range(n)):
            rhs = sum(Q[j, n - 1 - j] for j in range(n))
            constraints.append(lhs == rhs)

    # Objective
    if maximize:
        objective = cp.Maximize(m[target_moment])
    else:
        objective = cp.Minimize(m[target_moment])

    problem = cp.Problem(objective, constraints)

    try:
        _solve(problem)
        if problem.status in ["optimal", "optimal_inaccurate"]:
            return problem.value
        else:
            print(f"SDP status: {problem.status}")
            return None
    except Exception as e:
        print(f"SDP solver error: {e}")
        return None


def bootstrap_moment_bounds(v_prime_coeffs: list[float], max_moment: int = 10) -> dict:
    """Compute upper and lower bounds on all even moments."""
    bounds = {}
    for k in range(2, max_moment + 1, 2):
        ub = bootstrap_one_matrix(v_prime_coeffs, max_moment, k, maximize=True)
        lb = bootstrap_one_matrix(v_prime_coeffs, max_moment, k, maximize=False)
        bounds[k] = (lb, ub)
        if lb is not None and ub is not None:
            print(f"  m_{k}: [{lb:.8f}, {ub:.8f}], width = {ub - lb:.2e}")
    return bounds


def _two_matrix_canon(w):
    """Canonical word under cyclicity + M0↔M1 exchange; None if Z2×Z2 parity
    forces the moment to vanish (odd count of any generator)."""
    if any(w.count(c) % 2 == 1 for c in set(w)):
        return None
    if not w:
        return ()
    cands = []
    for v in (w, tuple(1 - x for x in w)):  # exchange
        for i in range(len(v)):  # cyclic
            cands.append(v[i:] + v[:i])
    return min(cands)


def _bootstrap_two_matrix(max_word_len, target_word, maximize, with_status, *,
                          mass, quartic, comm):
    """SDP island edge (min/max) of a single-trace moment for the general two-matrix
    force V'_a = mass·M_a + quartic·M_a³ + comm·(M_a M_b²+M_b² M_a − 2 M_b M_a M_b).

    Relaxation bootstrap: moment matrix Ω⪰0 (state positivity), product matrix G
    relaxing the factorized SD RHS (G[0,k]=m_k, G⪰0 ⇒ G⪰m mᵀ), loop equations, with
    cyclicity/exchange/Z2×Z2 baked into the canonicalization. With `with_status=True`
    returns (value, solver, status); parity-forbidden moments are exactly 0 (solver
    'exact'). Use the `bootstrap_two_matrix` / `bootstrap_two_matrix_kz` wrappers.
    """
    if not HAS_CVXPY:
        return (None, None, None) if with_status else None
    from itertools import product as iproduct

    canon = _two_matrix_canon
    words = [()]
    for L in range(1, max_word_len + 1):
        words += [tuple(c) for c in iproduct((0, 1), repeat=L)]
    canon_list = sorted({canon(w) for w in words} - {None}, key=lambda t: (len(t), t))
    cidx = {c: i for i, c in enumerate(canon_list)}
    nvar = len(canon_list)

    M = cp.Variable(nvar, name="moments")
    cons = [M[cidx[()]] == 1.0]

    def me(w):  # cvxpy moment expression (0.0 if parity-forbidden)
        c = canon(w)
        return 0.0 if c is None else M[cidx[c]]

    # Moment matrix Ω_{ij} = m(reverse(b_i)+b_j) ⪰ 0  (word† = reversed word).
    half = max_word_len // 2
    basis = [()]
    for L in range(1, half + 1):
        basis += [tuple(c) for c in iproduct((0, 1), repeat=L)]
    nb = len(basis)
    Omega = cp.Variable((nb, nb), symmetric=True, name="Omega")
    for i in range(nb):
        for j in range(i, nb):
            cons.append(Omega[i, j] == me(tuple(reversed(basis[i])) + basis[j]))
    cons.append(Omega >> 0)

    # Product matrix G_{kl} relaxing m_k·m_l (G[0,:]=m, G⪰0 ⇒ G⪰m mᵀ).
    G = cp.Variable((nvar, nvar), symmetric=True, name="Products")
    for k in range(nvar):
        cons.append(G[cidx[()], k] == M[k])
    cons.append(G >> 0)

    # Loop equations for V'_a = mass·M_a + quartic·M_a³
    #   + comm·(M_a M_b² + M_b² M_a − 2 M_b M_a M_b). The quartic and commutator terms
    #   each insert 3 letters, so test words run to max_word_len−3 (longest evaluated
    #   moment = max_word_len, which stays inside the moment-variable set).
    test_words = [()]
    for L in range(1, max(1, max_word_len - 3) + 1):
        test_words += [tuple(c) for c in iproduct((0, 1), repeat=L)]
    for w in test_words:
        for a in (0, 1):
            b = 1 - a
            lhs = mass * me((a,) + w)
            if quartic != 0.0:
                lhs = lhs + quartic * me((a, a, a) + w)
            if comm != 0.0:
                lhs = lhs + comm * (
                    me((a, b, b) + w) + me((b, b, a) + w) - 2.0 * me((b, a, b) + w)
                )
            rhs = 0.0
            for j in range(len(w)):
                if w[j] == a:
                    cl, cr = canon(w[:j]), canon(w[j + 1:])
                    if cl is None or cr is None:
                        continue  # product of a parity-forbidden (zero) moment
                    rhs = rhs + G[cidx[cl], cidx[cr]]
            cons.append(lhs == rhs)

    tc = canon(target_word)
    if tc is None:
        # parity-forbidden moment is exactly 0 by Z2×Z2 (certified by symmetry).
        return (0.0, "exact", "optimal") if with_status else 0.0
    obj = cp.Maximize(M[cidx[tc]]) if maximize else cp.Minimize(M[cidx[tc]])
    problem = cp.Problem(obj, cons)
    try:
        _solve(problem)
        val = float(problem.value) if problem.status in ("optimal", "optimal_inaccurate") else None
    except Exception as e:  # pragma: no cover
        print(f"two-matrix SDP error: {e}")
        val = None
    if with_status:
        return val, _LAST_SOLVE["solver"], _LAST_SOLVE["status"]
    return val


def bootstrap_two_matrix(g, max_word_len=4, target_word=(0, 0), maximize=True,
                         with_status=False):
    """SDP island edge for the commutator+mass model
    S = N·tr[½(M0²+M1²) − (g²/4)[M0,M1]²]: V'_a force coeffs (mass 1, no quartic,
    comm g²/2). See `_bootstrap_two_matrix`."""
    return _bootstrap_two_matrix(max_word_len, target_word, maximize, with_status,
                                 mass=1.0, quartic=0.0, comm=float(g) * float(g) / 2.0)


def bootstrap_two_matrix_kz(g, h, max_word_len=4, target_word=(0, 0), maximize=True,
                            with_status=False):
    """SDP island edge for the Kazakov–Zheng model (arXiv:2108.04830 eq.6)
    S = N·tr[½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]²]: V'_a force coeffs
    (mass 1, quartic g, comm h). See `_bootstrap_two_matrix`."""
    return _bootstrap_two_matrix(max_word_len, target_word, maximize, with_status,
                                 mass=1.0, quartic=float(g), comm=float(h))


# ─── M5a: single-particle anharmonic-oscillator QM bootstrap ──────────────────

def _qm_anharmonic_sdp_constraints(m, E, g, K, margin=None):
    """cvxpy constraints for H=p²+x²+g x⁴ at fixed energy E: m_0=1, odd moments 0, the
    stationarity recursion D3 (linear in m for fixed E), and Hankel(m) ⪰ margin·I.

    `m` is a cvxpy Variable of length 2K+1 (m[k]=⟨x^k⟩). With m_0=1 and the K-1
    recursion equalities, the only free moment is m[2]=⟨x²⟩. `margin=None` ⇒ Hankel ⪰ 0;
    a cvxpy `margin` variable enables the max-min-eigenvalue formulation (see
    `qm_anharmonic_margin`).
    """
    import numpy as _np
    cons = [m[0] == 1.0]
    for k in range(1, 2 * K + 1, 2):
        cons.append(m[k] == 0.0)  # parity: odd moments vanish
    for t in range(1, 2 * K - 1, 2):  # t = 1,3,...,2K-3 ; uses m up to index 2K
        expr = 4.0 * t * E * m[t - 1] - 4.0 * (t + 1) * m[t + 1] - 4.0 * g * (t + 2) * m[t + 3]
        if t >= 3:
            expr = expr + t * (t - 1) * (t - 2) * m[t - 3]
        cons.append(expr == 0.0)
    H = cp.bmat([[m[i + j] for j in range(K + 1)] for i in range(K + 1)])  # Hankel
    cons.append(H >> 0 if margin is None else H - margin * _np.eye(K + 1) >> 0)
    return cons


def qm_anharmonic_margin(g, K, E, with_status=False):
    """Max-min-eigenvalue margin of the Hankel matrix over the recursion-constrained
    moments at fixed energy E: maximize t s.t. Hankel(m) ⪰ t·I. Always a BOUNDED
    optimization (λ_min of an affine family is concave; Hankel[0,0]=m_0=1 caps it), so a
    trusted solver returns 'optimal' reliably — unlike the bare feasibility status, which
    flakes to SCS at the island edge. E admits a valid state iff t* ≥ 0; t* < 0 is an
    (optimal) certificate that energy E is below the spectrum.
    """
    if not HAS_CVXPY:
        return (None, None) if with_status else None
    m = cp.Variable(2 * K + 1)
    t = cp.Variable()
    cons = _qm_anharmonic_sdp_constraints(m, E, g, K, margin=t)
    prob = cp.Problem(cp.Maximize(t), cons)
    _solve(prob)
    ok = prob.status in ("optimal", "optimal_inaccurate")
    tstar = float(prob.value) if ok else None
    if with_status:
        return tstar, (_LAST_SOLVE["solver"], _LAST_SOLVE["status"])
    return tstar


def qm_anharmonic_feasibility(g, K, E, with_status=False):
    """At fixed energy E, the [min, max] of m2=⟨x²⟩ over {recursion(E,g) + Hankel ⪰ 0}.

    Returns (m2_lo, m2_hi); an edge is None if that solve is not optimal. E is feasible
    iff both are not None. With with_status, also returns ((lo_solver, lo_status),
    (hi_solver, hi_status)) for certification.
    """
    if not HAS_CVXPY:
        return (None, None, (None, None), (None, None)) if with_status else (None, None)

    def _edge(maximize):
        m = cp.Variable(2 * K + 1)
        cons = _qm_anharmonic_sdp_constraints(m, E, g, K)
        obj = cp.Maximize(m[2]) if maximize else cp.Minimize(m[2])
        prob = cp.Problem(obj, cons)
        _solve(prob)
        ok = prob.status in ("optimal", "optimal_inaccurate")
        val = float(prob.value) if ok else None
        return val, (_LAST_SOLVE["solver"], _LAST_SOLVE["status"])

    lo, lo_st = _edge(False)
    hi, hi_st = _edge(True)
    if with_status:
        return lo, hi, lo_st, hi_st
    return lo, hi


def bootstrap_qm_anharmonic(g, K, e_anchor, e_low=0.0, tol=1e-5, with_status=False):
    """Certified lower bound on the ground-state energy E0(g): the left edge of the
    lowest feasibility island.

    The feasible energies form narrow islands around the eigenvalues (they shrink to the
    exact spectrum as K grows), so a fixed-step scan misses them. We instead ANCHOR at a
    known-feasible energy `e_anchor` near E0 — e.g. the exact-diag E0, whose moments are
    a genuine feasible point at every K — and bisect DOWN to the infeasible boundary,
    which is the certified lower bound E_lo <= E0.

    With with_status, returns (E_lo, solver, status) from the feasible-edge solve so the
    caller can require a trusted, 'optimal' certificate.
    """
    if not HAS_CVXPY:
        return (None, None, None) if with_status else None
    margin_tol = 1e-7

    def feasible(E):  # E admits a valid state iff the margin t* >= 0
        t = qm_anharmonic_margin(g, K, E)
        return t is not None and t >= -margin_tol

    if not feasible(e_anchor):
        return (None, None, None) if with_status else None  # anchor not inside an island
    if feasible(e_low):
        E_lo = e_low  # island extends below e_low; the bound is only <= e_low (loose)
    else:
        lo_E, hi_E = e_low, e_anchor  # margin<0 (infeasible), margin>=0 (feasible)
        while hi_E - lo_E > tol:
            mid = 0.5 * (lo_E + hi_E)
            if feasible(mid):
                hi_E = mid
            else:
                lo_E = mid
        E_lo = hi_E  # left edge of the lowest island = lower bound on E0
    if with_status:
        _, st = qm_anharmonic_margin(g, K, E_lo, with_status=True)  # always 'optimal'
        return E_lo, st[0], st[1]
    return E_lo


# ─── M5b: single-matrix QM bootstrap (large N) ────────────────────────────────

def bootstrap_single_matrix_qm(g, L=4, *, maximize=False, with_status=False):
    """Certified bound on E/N² for single-matrix QM H=Tr P²+Tr X²+(g/N)Tr X⁴ (HHK Eq 8).

    The correct large-N bootstrap (see docs/.../2026-06-25-m5b-single-matrix-qm.md):
    ORDERED single-trace moments m[w]=⟨(1/N)Tr w⟩ as independent variables (NOT reduced by
    any c-number commutator — the matrix [X,P] is NOT iN·𝟙), constrained by
      • m[()]=1; hermiticity m[w]*=m[rev w]; time-reversal reality (m[w] real if #P̃ even,
        imaginary if odd);
      • stationarity ⟨[H,Tr w]⟩=0 (EOM [H,X̃]=−2iP̃, [H,P̃]=i(2X̃+4gX̃³); ordered);
      • SU(N) Gauss law ⟨Tr([X,P]O)⟩=iN⟨Tr O⟩ ⟹ m[(0,1)+O]−m[(1,0)+O]=i·m[O];
      • Gram positivity (complex Hermitian, real-embedded).
    `min` E/N² is the certified LOWER bound (the collective field gives the upper bound).
    `E/N² = m[(1,1)] + m[(0,0)] + g·m[(0,0,0,0)]`. Returns the bound (or (val,solver,status)).
    """
    if not HAS_CVXPY:
        return (None, None, None) if with_status else None
    from itertools import product as _ip

    def words_upto(n):
        o = [()]
        for k in range(1, n + 1):
            o += [tuple(c) for c in _ip((0, 1), repeat=k)]
        return o

    allw = [w for w in words_upto(L) if len(w) % 2 == 0]
    var = {w: cp.Variable(complex=True) for w in allw if w != ()}

    def m(w):
        if len(w) % 2 == 1:
            return 0
        if w == ():
            return 1.0 + 0j
        return var.get(w, None)

    cons = []
    for w in allw:
        if w == ():
            continue
        mw = m(w)
        cons.append(cp.imag(mw) == 0 if sum(w) % 2 == 0 else cp.real(mw) == 0)  # reality
        mr = m(tuple(reversed(w)))
        if mr is not None:
            cons.append(mw == cp.conj(mr))  # hermiticity
    # stationarity ⟨[H, Tr w]⟩=0
    for w in words_upto(max(1, L - 2)):
        expr, ok = 0, True
        for k, letter in enumerate(w):
            if letter == 0:
                t = m(w[:k] + (1,) + w[k + 1:])
                if t is None:
                    ok = False; break
                expr = expr + (-2j) * t
            else:
                t1 = m(w[:k] + (0,) + w[k + 1:])
                t3 = m(w[:k] + (0, 0, 0) + w[k + 1:])
                if t1 is None or t3 is None:
                    ok = False; break
                expr = expr + (2j) * t1 + (4j * g) * t3
        if ok and not isinstance(expr, (int, float, complex)):
            cons += [cp.real(expr) == 0, cp.imag(expr) == 0]
    # SU(N) Gauss law: m[(0,1)+O] - m[(1,0)+O] = i m[O]
    for O in words_upto(L - 2):
        a, b, c = m((0, 1) + O), m((1, 0) + O), m(O)
        if a is None or b is None or c is None:
            continue
        cons += [cp.real(a - b - 1j * c) == 0, cp.imag(a - b - 1j * c) == 0]
    # Gram PSD (complex Hermitian via real embedding)
    basis = words_upto(L // 2)
    nb = len(basis)
    A = [[None] * nb for _ in range(nb)]
    B = [[None] * nb for _ in range(nb)]
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            e = m(tuple(reversed(u)) + v)
            if e is None:
                return (None, None, None) if with_status else None
            A[i][j] = cp.real(e)
            B[i][j] = cp.imag(e)
    embed = cp.bmat([[cp.bmat(A), -cp.bmat(B)], [cp.bmat(B), cp.bmat(A)]])
    cons.append(embed >> 0)
    energy = cp.real(m((1, 1)) + m((0, 0)) + g * m((0, 0, 0, 0)))
    prob = cp.Problem(cp.Maximize(energy) if maximize else cp.Minimize(energy), cons)
    _solve(prob)
    ok = prob.status in ("optimal", "optimal_inaccurate")
    val = float(prob.value) if ok else None
    if with_status:
        return val, _LAST_SOLVE["solver"], _LAST_SOLVE["status"]
    return val


def _tm_qm_words_upto(L):
    out, cur = [()], [()]
    for _ in range(L):
        cur = [w + (c,) for w in cur for c in (0, 1, 2, 3)]  # X̃,Ỹ,P̃_X,P̃_Y
        out += cur
    return out


def bootstrap_two_matrix_qm(m, lam, L=4, *, maximize=False, with_status=False):
    """Certified bound on E/N² for HHK Eq 17 H=Tr(P_X²+P_Y²+m²(X²+Y²)−g²[X,Y]²).

    Ordered single-trace moments in {X̃,Ỹ,P̃_X,P̃_Y}; stationarity (T5) + SU(N) Gauss law
    (T2) + Gram PSD. Minimizing the relaxation gives a certified lower bound on E/N².
    Reuses the M5b single-matrix-QM machinery (real-embedded Hermitian Gram, _solve).
    """
    from matrix_master_field.tm_qm_relations import stationarity_terms
    if L < 4:
        raise ValueError("L>=4 required: the E/N² objective reads length-4 commutator moments "
                         "m[[X̃,Ỹ]²]; L=4 has only length-0,2 words.")
    if not HAS_CVXPY:
        return (None, None, None) if with_status else None

    allw = [w for w in _tm_qm_words_upto(L) if len(w) % 2 == 0]
    var = {w: cp.Variable(complex=True) for w in allw if w != ()}

    def mm(w):
        w = tuple(w)
        if len(w) % 2 == 1:
            return 0.0 + 0j
        if w == ():
            return 1.0 + 0j
        return var.get(w, None)

    cons = []
    # stationarity ⟨[H,Tr w]⟩=0
    for w in _tm_qm_words_upto(max(1, L - 2)):
        expr, ok = 0, True
        for coeff, ww in stationarity_terms(w):
            t = mm(ww)
            if t is None:
                ok = False
                break
            expr = expr + coeff(m, lam) * t
        if ok and not isinstance(expr, (int, float, complex)):
            cons += [cp.real(expr) == 0, cp.imag(expr) == 0]
    # SU(N) Gauss law, both canonical pairs
    for O in _tm_qm_words_upto(L - 2):
        for pair in ((0, 2), (1, 3)):  # position-first; rel = m[XP+O]−m[PX+O]−i·m[O]=0
            rel = [(1.0, pair + O), (-1.0, (pair[1], pair[0]) + O), (-1j, O)]
            terms = [(c, mm(ww)) for c, ww in rel]
            if any(t is None for _, t in terms):
                continue
            e = sum(c * t for c, t in terms)
            cons += [cp.real(e) == 0, cp.imag(e) == 0]
    # Gram PSD (complex Hermitian via real embedding), basis = words up to L//2
    basis = _tm_qm_words_upto(L // 2)
    A = [[None] * len(basis) for _ in basis]
    B = [[None] * len(basis) for _ in basis]
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            e = mm(tuple(reversed(u)) + v)
            if e is None:
                return (None, None, None) if with_status else None
            A[i][j], B[i][j] = cp.real(e), cp.imag(e)
    embed = cp.bmat([[cp.bmat(A), -cp.bmat(B)], [cp.bmat(B), cp.bmat(A)]])
    cons.append(embed >> 0)

    # E/N² = m[P̃_X²]+m[P̃_Y²]+m²(m[X̃²]+m[Ỹ²]) − λ·m[[X̃,Ỹ]²]
    comm2 = (mm((0, 1, 0, 1)) - mm((0, 1, 1, 0)) - mm((1, 0, 0, 1)) + mm((1, 0, 1, 0)))
    energy = (mm((2, 2)) + mm((3, 3)) + m**2 * (mm((0, 0)) + mm((1, 1))) - lam * comm2)
    obj = cp.Maximize(cp.real(energy)) if maximize else cp.Minimize(cp.real(energy))
    prob = _solve(cp.Problem(obj, cons))
    val = None if prob.status not in ("optimal", "optimal_inaccurate") else float(prob.value)
    if with_status:
        return val, _LAST_SOLVE["solver"], _LAST_SOLVE["status"]
    return val


if __name__ == "__main__":
    if not HAS_CVXPY:
        print("Install cvxpy to run bootstrap validation")
        exit()

    print("=" * 60)
    print("SDP Bootstrap Validation")
    print("=" * 60)

    # Gaussian: V'(M) = M
    print("\nGaussian model (V' = M):")
    bounds_g = bootstrap_moment_bounds([0.0, 1.0], max_moment=8)

    from matrix_master_field.one_matrix import gaussian_moments

    m_exact = gaussian_moments(10)
    print("\nExact values:")
    for k in range(2, 9, 2):
        print(f"  m_{k} = {m_exact[k]:.8f}")

    # Quartic: V'(M) = M + 0.5 M³
    print("\nQuartic model (V' = M + 0.5 M³):")
    bounds_q = bootstrap_moment_bounds([0.0, 1.0, 0.0, 0.5], max_moment=8)
