"""Sequential-SDP (SQP) with continuation on the KZ two-matrix model (moment space).

Non-linear method #3.  At each iterate m_k, linearize the quadratic loop-equation RHS
  m[j]*m[l]  ~=  m_prev[j]*m[l] + m[j]*m_prev[l] - m_prev[j]*m_prev[l]
and solve a CONVEX SDP with EXACT positivity Omega(m) >= 0 (cvxpy), minimizing the
change ||m - m_k|| (trust-region-ish).  Iterate to convergence (Newton-SDP), then step
the homotopy (g,h) = t*(g_t,h_t).  Exact PSD removes the soft-penalty drift that made the
ALM continuation land slightly off the manifold.

GT: g=1,h=1 -> [0.4204,0.4224];  g=0.5,h=1 -> [0.4803,0.4842].
"""
import sys, time
import numpy as np
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))
import cvxpy as cp
from matrix_master_field.twomatrix_alm import build_structure, free_moment
from matrix_master_field.bootstrap_sdp import _two_matrix_canon as canon


def sqp_solve_at(struct, m_full0, n_newton=12, tr=0.5):
    """Newton-SDP: converge to a feasible point of the exact loop eqs + Omega>=0,
    starting from m_full0 (length nvar; m_full0[0]=1). Returns m_full."""
    nvar = struct["nvar"]
    omega_idx = struct["omega_idx"]; SENT = nvar
    eqs = struct["loop_eqs"]
    nb = omega_idx.shape[0]
    m_prev = m_full0.copy()
    for _ in range(n_newton):
        M = cp.Variable(nvar)
        def me(i):
            return 0.0 if i == SENT else M[i]
        cons = [M[0] == 1.0]
        # Omega(m) >= 0  (exact)
        Om = cp.bmat([[me(int(omega_idx[i, j])) for j in range(nb)] for i in range(nb)])
        cons.append(Om >> 0)
        # linearized loop equations
        for (lc, li, rcl, rcr) in eqs:
            lhs = sum(float(lc[k]) * me(int(li[k])) for k in range(len(lc)))
            rhs = 0.0
            for a in range(len(rcl)):
                j, l = int(rcl[a]), int(rcr[a])
                mj, ml = (1.0 if j == 0 else m_prev[j]) if j != SENT else 0.0, \
                         (1.0 if l == 0 else m_prev[l]) if l != SENT else 0.0
                rhs = rhs + mj * me(l) + me(j) * ml - mj * ml
            cons.append(lhs == rhs)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(M - m_prev)), cons)
        try:
            prob.solve(solver=cp.SCS, max_iters=20000)
        except Exception as e:
            print("   SDP err", e); break
        if M.value is None:
            break
        step = M.value - m_prev
        n = np.linalg.norm(step)
        if n > tr:
            step = step * (tr / n)
        m_prev = m_prev + step
        if n < 1e-9:
            break
    return m_prev


def continuation_sqp(g, h, L, n_steps=12, target=(0, 0)):
    s_t = build_structure(L, 1.0, g, h)
    cl, cidx = s_t["canon_list"], s_t["cidx"]
    tidx = cidx[canon(target)]
    m_full = np.concatenate([[1.0], [float(free_moment(c)) for c in cl[1:]]])
    for t in np.linspace(0.0, 1.0, n_steps + 1):
        s = build_structure(L, 1.0, t * g, t * h)
        m_full = sqp_solve_at(s, m_full)
    return float(m_full[tidx])


if __name__ == "__main__":
    print("SEQUENTIAL-SDP (SQP) + continuation, exact PSD, KZ model")
    print("GT: g=1,h=1 -> [0.4204,0.4224];  g=0.5,h=1 -> [0.4803,0.4842]\n")
    for (g, h, isl) in [(1.0, 1.0, "[0.4204,0.4224]"), (0.5, 1.0, "[0.4803,0.4842]")]:
        print(f"g={g} h={h}  island {isl}:")
        for L in (6, 8):
            t0 = time.time()
            m = continuation_sqp(g, h, L, n_steps=12)
            print(f"  L={L}: m[A^2]={m:.5f}   {time.time()-t0:.0f}s", flush=True)
        print()
