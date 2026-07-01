"""Non-relaxation constraint handling & master-field feasibility on the massive
D=2 Lin-Zheng two-matrix QM (arXiv:2507.21007).

Testbed (complex App-E letters Z,z=Zb,P,q=Pb; P=Pi=-iP_phys so correlators real):

    H = Tr[ 1/2 (Pi_I^2 + M^2 X_I^2) - 1/4 [X_1,X_2]^2 ],   I=1,2,   M^2 = 1.

Conventions (pinned, matching bfss/lz_port.py + lz_pos2.py, verified last session):
  - single-trace moment variables x[w] = <(1/N) tr w>, w a charge-neutral word.
  - E := E_0/N^2 = -1.5 <tr P q> + 0.5 M^2 <tr Z z>            (objE below).
  - <tr Z z> = <tr Z Zbar> = (1/2)<tr X_I X_I> (per-flavour <tr X^2>).
  - Rigorous level-14 island (Lin-Zheng Table II), the REFEREE ONLY (never a constraint):
        E              in [1.172098376, 1.172098408]
        <tr X_I X_I>   in [0.77800898, 0.77800934]  => <tr Z z> in [0.38900449, 0.38900467]

This module adds the Part-A (flatness/relaxation-cost) and Part-C (selection) diagnostics
ON TOP of the existing engine; it does not re-derive the bootstrap. The factorized loop
variety and the ground-state positivity blocks both come straight from lz_pos2.build:
  rels = relZ + commH_rels(1.0),  each compiled as (const, [(i,c)], [(i,j,c)]) meaning
      const + sum c*x[i] + sum c*x[i]*x[j] = 0,
  where the quad terms ARE the large-N factorization products (double-trace = product of
  single-traces). Mblocks -> M(x)>=0 (state positivity), Nblocks(g) -> N(x)>=0 (ground state).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lz_pos2  # noqa: E402  (build: rels + PSD blocks)
from lz_gauss_moments import moment, twopt_table  # noqa: E402  (true Gaussian anchor)

# Rigorous referee island (Lin-Zheng Table II). NEVER enters a solver.
ISLAND_E = (1.172098376, 1.172098408)
ISLAND_ZZ = (0.38900449, 0.38900467)  # <tr Z z> = half of <tr X_I X_I> in [0.778...]


# --------------------------------------------------------------------------- #
#  System assembly (reuse lz_pos2.build)                                       #
# --------------------------------------------------------------------------- #
class System:
    """Compiled loop system + positivity blocks at fixed (level, M2, g=1)."""

    def __init__(self, level: int, M2: float = 1.0):
        B = lz_pos2.build(level, M2)
        self.level = level
        self.M2 = M2
        self.Wneut = B["Wneut"]
        self.vidx = B["vidx"]
        self.nV = B["nV"]
        # exact factorized loop equations at g=1: (const, lin, quad)
        self.rels = list(B["relZ"]) + list(B["commH_rels"](1.0))
        self.Mblocks = B["Mblocks"]          # entries (idx, sign)
        self.Nblocks = B["Nblocks"](1.0)     # entries list[(idx, coef)]
        # key observables

        def vio(t):
            t = tuple(t)
            return self.vidx.get(t, self.vidx.get(t[::-1]))

        self.iZz = vio(("Z", "z"))
        self.iPq = vio(("P", "q"))
        # true Gaussian (g=0) anchor for scale / dimension base point
        tp = twopt_table(M2)
        self.x_gauss = np.array([moment(w, tp).real for w in self.Wneut])

    def objE(self, x: np.ndarray) -> float:
        return float(-1.5 * x[self.iPq] + 0.5 * self.M2 * x[self.iZz])


# --------------------------------------------------------------------------- #
#  Residual, Jacobian of the factorized (quadratic) loop system                #
# --------------------------------------------------------------------------- #
def residual(x: np.ndarray, rels) -> np.ndarray:
    out = np.empty(len(rels))
    for r, (cst, lin, quad) in enumerate(rels):
        v = cst
        for i, c in lin:
            v += c * x[i]
        for i, j, c in quad:
            v += c * x[i] * x[j]
        out[r] = v
    return out


def jacobian(x: np.ndarray, rels, nV: int) -> np.ndarray:
    J = np.zeros((len(rels), nV))
    for r, (cst, lin, quad) in enumerate(rels):
        for i, c in lin:
            J[r, i] += c
        for i, j, c in quad:
            J[r, i] += c * x[j]
            J[r, j] += c * x[i]
    return J


def newton_project(x0, rels, nV, iters=200, tol=1e-11, damp0=1.0):
    """Gauss-Newton (min-norm pseudo-inverse) drive F(x)->0 from x0. Returns (x, |F|)."""
    x = np.array(x0, float)
    fn = np.linalg.norm(residual(x, rels))
    for _ in range(iters):
        F = residual(x, rels)
        fn = np.linalg.norm(F)
        if fn < tol:
            break
        J = jacobian(x, rels, nV)
        step = np.linalg.lstsq(J, -F, rcond=None)[0]
        # backtracking line search
        d = damp0
        while d > 1e-6:
            xn = x + d * step
            if np.linalg.norm(residual(xn, rels)) < fn:
                x = xn
                break
            d *= 0.5
        else:
            break
    return x, float(np.linalg.norm(residual(x, rels)))


# --------------------------------------------------------------------------- #
#  Part C.3 — variety dimension = nV - rank(Jacobian) at smooth points         #
# --------------------------------------------------------------------------- #
def variety_dimension(sys: System, n_probe=12, seed=0, rank_tol=1e-7, verbose=True):
    """Local dimension of the exact (factorized) loop variety.

    Newton-projects random seeds onto the variety and reports rank(J) there.
    dim = nV - rank(J). Uses several projected points to get the GENERIC rank
    (a positive-dimensional variety has constant rank on its smooth locus).
    """
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(sys.x_gauss), 0.3)
    ranks, dims, resids = [], [], []
    for _ in range(n_probe):
        x0 = sys.x_gauss + rng.normal(0, 1.0, sys.nV) * scale
        x, fn = newton_project(x0, sys.rels, sys.nV)
        if fn > 1e-6:            # did not reach the variety
            continue
        J = jacobian(x, sys.rels, sys.nV)
        s = np.linalg.svd(J, compute_uv=False)
        tol = rank_tol * max(J.shape) * (s[0] if s.size else 1.0)
        rank = int((s > tol).sum())
        ranks.append(rank)
        dims.append(sys.nV - rank)
        resids.append(fn)
    if not ranks:
        return {"nV": sys.nV, "rank": None, "dim": None, "n_ok": 0}
    rank = int(np.median(ranks))
    dim = sys.nV - rank
    if verbose:
        print(f"  [dim] level={sys.level}: nV={sys.nV}  #eqs={len(sys.rels)}  "
              f"rank(J)={rank} (spread {min(ranks)}-{max(ranks)})  => dim={dim}  "
              f"(projected {len(ranks)}/{n_probe} seeds, |F|<1e-6)")
    return {"nV": sys.nV, "n_eqs": len(sys.rels), "rank": rank, "dim": dim,
            "n_ok": len(ranks), "rank_min": min(ranks), "rank_max": max(ranks)}


# --------------------------------------------------------------------------- #
#  Positivity test on the variety                                              #
# --------------------------------------------------------------------------- #
def _block_min_eig_M(x, Mblocks):
    mn = np.inf
    for ent in Mblocks:
        n = len(ent)
        A = np.array([[ent[i][j][1] * x[ent[i][j][0]] for j in range(n)] for i in range(n)])
        A = 0.5 * (A + A.T)
        mn = min(mn, float(np.linalg.eigvalsh(A)[0]))
    return mn


def _block_min_eig_N(x, Nblocks):
    mn = np.inf
    for ent in Nblocks:
        n = len(ent)
        A = np.array([[sum(c * x[idx] for idx, c in ent[i][j]) if ent[i][j] else 0.0
                       for j in range(n)] for i in range(n)])
        A = 0.5 * (A + A.T)
        mn = min(mn, float(np.linalg.eigvalsh(A)[0]))
    return mn


def positivity(x, sys: System):
    mM = _block_min_eig_M(x, sys.Mblocks) if sys.Mblocks else np.inf
    mN = _block_min_eig_N(x, sys.Nblocks) if sys.Nblocks else np.inf
    return mM, mN


# --------------------------------------------------------------------------- #
#  Part C.4/5 — sample the variety, map the positive region                    #
# --------------------------------------------------------------------------- #
def sample_variety(sys: System, n_seeds=400, seed=1, spread=1.0, ftol=1e-9):
    """Multistart Gauss-Newton; collect distinct real points on the variety with
    their observables and positivity margins."""
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(sys.x_gauss), 0.3)
    pts = []
    for _ in range(n_seeds):
        x0 = sys.x_gauss + rng.normal(0, spread, sys.nV) * scale
        x, fn = newton_project(x0, sys.rels, sys.nV)
        if fn > ftol:
            continue
        mM, mN = positivity(x, sys)
        pts.append({"x": x, "res": fn, "E": sys.objE(x), "ZZ": float(x[sys.iZz]),
                    "minM": mM, "minN": mN})
    return pts


def _sqp_opt(sys: System, sense: str, x0, sqp_iters=8, tr=0.5, eps=1e-7):
    """One min/max of E over {exact factorization (SQP-linearised) AND M>=0 AND N>=0},
    from start x0. The SQP linearises the quad (factorization) terms around the current
    iterate; at the fixed point the linearisation is exact, so the returned point lies on
    the exact variety (residual reported). Reuses lz_pos2's inner-loop construction."""
    import cvxpy as cp
    nV = sys.nV
    x0 = np.array(x0, float)
    for _ in range(sqp_iters):
        x = cp.Variable(nV)
        cons = []
        for (cst, lin, quad) in sys.rels:
            ex = cst
            for i, c in lin:
                ex = ex + c * x[i]
            for i, j, c in quad:
                ex = ex + c * (x0[i] * x[j] + x[i] * x0[j] - x0[i] * x0[j])
            cons.append(ex == 0)
        for ent in sys.Mblocks:
            n = len(ent)
            Mm = cp.bmat([[ent[i][j][1] * x[ent[i][j][0]] for j in range(n)] for i in range(n)])
            cons.append(0.5 * (Mm + Mm.T) >> 0)
        for ent in sys.Nblocks:
            n = len(ent)
            Nm = cp.bmat([[sum(c * x[idx] for idx, c in ent[i][j]) if ent[i][j] else 0
                           for j in range(n)] for i in range(n)])
            cons.append(0.5 * (Nm + Nm.T) >> 0)
        cons.append(cp.norm(x - x0, "inf") <= tr)
        obj = -1.5 * x[sys.iPq] + 0.5 * sys.M2 * x[sys.iZz]
        pr = cp.Problem(cp.Minimize(obj) if sense == "min" else cp.Maximize(obj), cons)
        try:
            pr.solve(solver=cp.SCS, eps=eps, max_iters=20000, verbose=False)
        except Exception:
            break
        if x.value is None:
            break
        x0 = np.array(x.value).flatten()
    return x0, float(np.linalg.norm(residual(x0, sys.rels)))


def positive_extent(sys: System, n_starts=4, seed=7, verbose=True):
    """Authoritative point-vs-set measure: [E_min, E_max] over the POSITIVE variety
    (exact factorization + M>=0 + N>=0), by min/max SQP from several starts. Width ->
    island width means positivity collapses to a POINT; wide width means a SET."""
    rng = np.random.default_rng(seed)
    scale = np.maximum(np.abs(sys.x_gauss), 0.3)
    Emin, Emax = np.inf, -np.inf
    xmin = xmax = None
    for s in range(n_starts):
        x0 = sys.x_gauss if s == 0 else sys.x_gauss + rng.normal(0, 0.25, sys.nV) * scale
        xa, ra = _sqp_opt(sys, "min", x0)
        xb, rb = _sqp_opt(sys, "max", x0)
        if ra < 1e-4 and sys.objE(xa) < Emin:
            Emin, xmin = sys.objE(xa), xa
        if rb < 1e-4 and sys.objE(xb) > Emax:
            Emax, xmax = sys.objE(xb), xb
    width = Emax - Emin if np.isfinite(Emin) and np.isfinite(Emax) else float("nan")
    is_point = np.isfinite(width) and width < 5e-3
    if verbose:
        print(f"  [positive extent, min/max SQP] E in [{Emin:.5f}, {Emax:.5f}]  "
              f"width={width:.3e}  -> {'POINT' if is_point else 'SET'}")
        print(f"     island E={ISLAND_E} (width {ISLAND_E[1]-ISLAND_E[0]:.1e}); "
              f"island {'INSIDE' if (np.isfinite(Emin) and Emin <= ISLAND_E[0] and Emax >= ISLAND_E[1]) else 'not bracketed'}")
    return {"E_min_pos": float(Emin), "E_max_pos": float(Emax), "pos_width": float(width),
            "pos_is_point": bool(is_point)}


def _island_member(E, ZZ, tolE=1e-4, tolZ=1e-4):
    return (ISLAND_E[0] - tolE <= E <= ISLAND_E[1] + tolE and
            ISLAND_ZZ[0] - tolZ <= ZZ <= ISLAND_ZZ[1] + tolZ)


def analyse_level(sys: System, n_seeds=400, pos_eps=1e-6, verbose=True):
    """Full Part-C analysis at one level: dim, sampling, positive-region map."""
    dim = variety_dimension(sys, verbose=verbose)
    ext = positive_extent(sys, verbose=verbose)
    pts = sample_variety(sys, n_seeds=n_seeds)
    real = pts
    positive = [p for p in pts if p["minM"] >= -pos_eps and p["minN"] >= -pos_eps]
    # characterise the positive region spread in the two physical observables
    if positive:
        Es = np.array([p["E"] for p in positive])
        ZZs = np.array([p["ZZ"] for p in positive])
        e_lo, e_hi = float(Es.min()), float(Es.max())
        z_lo, z_hi = float(ZZs.min()), float(ZZs.max())
        e_width = e_hi - e_lo
        is_point = e_width < 5e-3           # a "point" vs a "set" at this resolution
        matches = any(_island_member(p["E"], p["ZZ"]) for p in positive)
    else:
        e_lo = e_hi = z_lo = z_hi = e_width = float("nan")
        is_point = False
        matches = False
    if verbose:
        print(f"  [sample] level={sys.level}: variety pts (|F|<1e-9)={len(real)}  "
              f"positive (minM,minN>=-{pos_eps:.0e})={len(positive)}")
        if positive:
            print(f"  [positive region] E in [{e_lo:.5f},{e_hi:.5f}] (width {e_width:.2e})  "
                  f"<trZz> in [{z_lo:.5f},{z_hi:.5f}]")
            print(f"     -> positive set is a {'POINT' if is_point else 'SET'};  "
                  f"island match={matches}   (island E {ISLAND_E}, <trZz> {ISLAND_ZZ})")
        else:
            print("  [positive region] EMPTY at this sampling/eps")
    island_in_set = (np.isfinite(ext["E_min_pos"]) and np.isfinite(ext["E_max_pos"]) and
                     ext["E_min_pos"] <= ISLAND_E[0] and ext["E_max_pos"] >= ISLAND_E[1])
    return {"level": sys.level, "nV": dim["nV"], "n_eqs": dim["n_eqs"],
            "dim": dim["dim"], "n_positive_sampled": len(positive),
            "E_min_pos": ext["E_min_pos"], "E_max_pos": ext["E_max_pos"],
            "pos_width": ext["pos_width"], "positive_is_point": ext["pos_is_point"],
            "island_in_positive_set": bool(island_in_set),
            "sampled_hit_island": bool(matches)}


def part_c(levels=(4, 6, 8), M2=1.0, n_seeds=400, out_csv=None):
    print("=" * 78)
    print("PART C — SELECTION PROBE (positivity stripped, then tested pointwise)")
    print("massive D=2 Lin-Zheng QM, M^2 =", M2, "  [island is REFEREE ONLY]")
    print("=" * 78)
    rows = []
    for L in levels:
        print(f"\n--- level {L} ---")
        sys_L = System(L, M2)
        rows.append(analyse_level(sys_L, n_seeds=n_seeds))
    print("\n=== selection summary: does positivity collapse the variety to a point? ===")
    for r in rows:
        print(f"  L={r['level']}: variety dim={r['dim']}  positive E-extent="
              f"[{r['E_min_pos']:.4f},{r['E_max_pos']:.4f}] width={r['pos_width']:.2e}  "
              f"-> {'POINT' if r['positive_is_point'] else 'SET'}   "
              f"(island inside positive set={r['island_in_positive_set']})")
    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")
    return rows


# --------------------------------------------------------------------------- #
#  Part A — what the Shor relaxation of Q = x x^T costs                         #
# --------------------------------------------------------------------------- #
#  In the COMPLEX (Z,z,P,q) formulation the cyclicity+CCR rule produces genuine
#  double-trace products (the `quad` terms). The Shor relaxation replaces each
#  product x_i x_j by an INDEPENDENT variable Q_ij and adds the convex coupling
#  [[1, x_S^T],[x_S, Q]] >= 0  (Schur complement => Q >= x_S x_S^T). At N=inf the
#  true state factorizes (Q = x_S x_S^T exactly, rank 1); the relaxation drops that.
#  S_l = ||Q* - x_S* x_S*^T||_F at the optimum is the DIRECT factorization violation.
#  (Basis caveat: in the REAL X,Y basis the QM loop equations are already linear in
#  single-trace moments, so there Q=xx^T is not the operative relaxation — the
#  positivity-hierarchy truncation is. Slack is a complex-basis diagnostic.)

def _product_indices(sys: System):
    S = set()
    for (_, _, quad) in sys.rels:
        for (i, j, _) in quad:
            S.add(i)
            S.add(j)
    return sorted(S)


def shor_relax(sys: System, sense: str, eps=1e-7, max_iters=100000):
    """Solve the Shor relaxation (explicit Q, Schur PSD, state positivity), min/max E.
    Returns dict with x, Q, S index list, E, and factorization slack."""
    import cvxpy as cp
    nV = sys.nV
    S = _product_indices(sys)
    pos = {s: k for k, s in enumerate(S)}
    nS = len(S)
    x = cp.Variable(nV)
    Q = cp.Variable((nS, nS), symmetric=True) if nS else None
    cons = []
    for (cst, lin, quad) in sys.rels:
        ex = cst
        for i, c in lin:
            ex = ex + c * x[i]
        for i, j, c in quad:
            ex = ex + c * Q[pos[i], pos[j]]
        cons.append(ex == 0)
    if nS:
        # Shor coupling [[1, x_S^T],[x_S, Q]] >= 0  (Schur => Q >= x_S x_S^T)
        xS = cp.reshape(cp.hstack([x[s] for s in S]), (1, nS), order="C")
        cons.append(cp.bmat([[np.array([[1.0]]), xS], [xS.T, Q]]) >> 0)
    # state positivity (same blocks as the exact solve)
    for ent in sys.Mblocks:
        n = len(ent)
        Mm = cp.bmat([[ent[i][j][1] * x[ent[i][j][0]] for j in range(n)] for i in range(n)])
        cons.append(0.5 * (Mm + Mm.T) >> 0)
    for ent in sys.Nblocks:
        n = len(ent)
        Nm = cp.bmat([[sum(c * x[idx] for idx, c in ent[i][j]) if ent[i][j] else 0
                       for j in range(n)] for i in range(n)])
        cons.append(0.5 * (Nm + Nm.T) >> 0)
    obj = -1.5 * x[sys.iPq] + 0.5 * sys.M2 * x[sys.iZz]
    pr = cp.Problem(cp.Minimize(obj) if sense == "min" else cp.Maximize(obj), cons)
    pr.solve(solver=cp.SCS, eps=eps, max_iters=max_iters, verbose=False)
    if x.value is None:
        return None
    xv = np.array(x.value).flatten()
    if nS:
        Qv = np.array(Q.value)
        xSv = xv[S]
        slack = Qv - np.outer(xSv, xSv)
        slack_fro = float(np.linalg.norm(slack))
        slack_lmax = float(np.linalg.eigvalsh(0.5 * (slack + slack.T))[-1])
    else:
        # no factorization products at this level -> nothing to relax (system linear)
        slack_fro = 0.0
        slack_lmax = 0.0
    return {"x": xv, "S": S, "nQ": nS, "E": sys.objE(xv),
            "slack_fro": slack_fro, "slack_lmax": slack_lmax, "status": pr.status}


def moment_matrix_rank(sys: System, x, tau=1e-6):
    """Rank + singular spectrum of the (block-diagonal) state moment matrix M(x)."""
    sv = []
    for ent in sys.Mblocks:
        n = len(ent)
        A = np.array([[ent[i][j][1] * x[ent[i][j][0]] for j in range(n)] for i in range(n)])
        A = 0.5 * (A + A.T)
        sv.extend(sorted(np.abs(np.linalg.eigvalsh(A)), reverse=True))
    sv = np.array(sorted(sv, reverse=True))
    thr = tau * (sv[0] if sv.size else 1.0)
    rank = int((sv > thr).sum())
    return rank, sv


def part_a(levels=(4, 6, 8), M2=1.0, out_csv=None):
    print("=" * 78)
    print("PART A — WHAT THE SHOR RELAXATION OF Q = x x^T COSTS (complex basis)")
    print("massive D=2 Lin-Zheng QM, M^2 =", M2, "  [double precision — trend, not 8-digit]")
    print("=" * 78)
    rows = []
    prev_rank = None
    for L in levels:
        sys_L = System(L, M2)
        ext = positive_extent(sys_L, verbose=False)   # robust (trust-region) min/max width
        rmin = shor_relax(sys_L, "min")                # slack + moment rank at min-E vertex
        if rmin is None:
            print(f"  L={L}: min-E SDP failed")
            continue
        w = ext["pos_width"]
        rank, sv = moment_matrix_rank(sys_L, rmin["x"])
        flat = "flat (=prev)" if prev_rank == rank else (f"prev={prev_rank}" if prev_rank else "")
        print(f"\n--- level {L}  (nV={sys_L.nV}, #factorization products |Q|={rmin['nQ']}) ---")
        print(f"  positive-set width w_l = {w:.4e}   E in [{ext['E_min_pos']:.5f}, {ext['E_max_pos']:.5f}]")
        print(f"  factorization slack S_l = ||Q-xx^T||_F = {rmin['slack_fro']:.4e}  "
              f"(lambda_max = {rmin['slack_lmax']:+.3e})   [at min-E vertex; 0 => no products to relax]")
        print(f"  moment-matrix rank r_l(1e-6) = {rank}  {flat}")
        print(f"     top |eig|: {np.array2string(sv[:6], precision=3)}")
        rows.append({"level": L, "nV": sys_L.nV, "nQ": rmin["nQ"],
                     "rank": rank, "slack_fro": rmin["slack_fro"],
                     "slack_lmax": rmin["slack_lmax"], "width": w,
                     "E_min": ext["E_min_pos"], "E_max": ext["E_max_pos"]})
        prev_rank = rank
    print("\n=== flatness/relaxation-cost summary ===")
    for r in rows:
        print(f"  L={r['level']}: rank={r['rank']}  slack S_l={r['slack_fro']:.3e}  "
              f"width w_l={r['width']:.3e}")
    print("  Interpretation: S_l->0 & w_l collapsing & rank stabilising => gap is")
    print("  flatness-controlled (relaxation asymptotically free). S_l plateauing")
    print("  nonzero with finite w_l => relaxation genuinely lossy here.")
    if out_csv and rows:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")
    return rows


if __name__ == "__main__":
    import argparse
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["a", "c", "both"], default="both")
    args = ap.parse_args()
    if args.part in ("c", "both"):
        part_c(levels=(4, 6, 8), M2=1.0, n_seeds=150,
               out_csv=os.path.join(repo, "results", "selection_branches.csv"))
    if args.part in ("a", "both"):
        part_a(levels=(4, 6, 8), M2=1.0,
               out_csv=os.path.join(repo, "results", "relaxation_flatness.csv"))
