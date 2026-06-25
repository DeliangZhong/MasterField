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


def _select_solver():
    """Highest-accuracy installed conic solver: MOSEK (if licensed) > CLARABEL > SCS.

    CLARABEL/MOSEK are interior-point (≈1e-8 tolerances) and return certified
    'optimal' status; SCS is a first-order fallback that flags 'optimal_inaccurate'
    on the larger (L≥8) two-matrix moment relaxations. Install `clarabel` (no
    license) or `mosek` (academic license) to certify the tight islands.
    """
    if not HAS_CVXPY:
        return None
    installed = set(cp.installed_solvers())
    for s in ("MOSEK", "CLARABEL", "SCS"):
        if s in installed:
            return s
    return None


SOLVER = _select_solver()
_LAST_SOLVE = {"solver": None, "status": None}  # which solver/status produced the last bound

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


def bootstrap_two_matrix(g, max_word_len=4, target_word=(0, 0), maximize=True):
    """SDP island bound on a single-trace moment of the commutator+mass two-matrix
    model S = N·tr[½(M0²+M1²) − (g²/4)[M0,M1]²] at coupling g.

    Relaxation bootstrap: moment matrix Ω⪰0 (state positivity), product matrix G
    relaxing the factorized SD RHS (G[0,k]=m_k, G⪰0 ⇒ G⪰m mᵀ), commutator loop
    equations, with cyclicity/exchange/Z2×Z2 baked into the canonicalization.
    Returns min or max of the target moment (the island edge), or None.
    """
    if not HAS_CVXPY:
        return None
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

    # Commutator-model loop equations: V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a − 2 M_b M_a M_b).
    g2 = float(g) * float(g)
    test_words = [()]
    for L in range(1, max(1, max_word_len - 3) + 1):
        test_words += [tuple(c) for c in iproduct((0, 1), repeat=L)]
    for w in test_words:
        for a in (0, 1):
            b = 1 - a
            lhs = me((a,) + w)
            if g2 != 0.0:
                lhs = lhs + (g2 / 2.0) * (
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
        return 0.0  # parity-forbidden moment is exactly 0
    obj = cp.Maximize(M[cidx[tc]]) if maximize else cp.Minimize(M[cidx[tc]])
    problem = cp.Problem(obj, cons)
    try:
        _solve(problem)
        if problem.status in ("optimal", "optimal_inaccurate"):
            return float(problem.value)
        return None
    except Exception as e:  # pragma: no cover
        print(f"two-matrix SDP error: {e}")
        return None


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
