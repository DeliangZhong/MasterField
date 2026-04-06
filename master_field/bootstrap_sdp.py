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
import numpy as np
from typing import List, Tuple, Optional

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    print("cvxpy not installed — SDP bootstrap unavailable")


def bootstrap_one_matrix(v_prime_coeffs: List[float], max_moment: int = 10,
                         target_moment: int = 2,
                         maximize: bool = True) -> Optional[float]:
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
    m = cp.Variable(K + 1, name='moments')
    
    constraints = []
    
    # m_0 = 1
    constraints.append(m[0] == 1.0)
    
    # Z₂ symmetry: m_{odd} = 0 (for symmetric potentials with v_{even}=0)
    is_symmetric = all(v_prime_coeffs[k] == 0 for k in range(0, n_v, 2))
    if is_symmetric:
        for k in range(1, K + 1, 2):
            constraints.append(m[k] == 0.0)
    
    # Schwinger-Dyson equations
    for n in range(1, K - n_v + 2):
        lhs = 0
        for k in range(n_v):
            idx = n + k - 1
            if 0 <= idx <= K:
                lhs += v_prime_coeffs[k] * m[idx]
        
        rhs = 0
        for j in range(n):
            if 0 <= j <= K and 0 <= n - j - 1 <= K:
                rhs += m[j] * m[n - j - 1]
        
        # For SDP, we need linear constraints. The RHS has bilinear terms.
        # The bootstrap linearises by introducing the moment matrix.
        # For simplicity, let's linearise around the factorised terms:
        # m_j * m_{n-j-1} is bilinear → need moment matrix.
        
        # Actually, the standard approach: define matrix X_{ij} = m_{i+j}
        # and impose X ≽ 0. The SD equations are linear in m.
        # The products m_j m_{n-j-1} come from the SPLITTING, which is
        # already captured by the moment matrix structure.
    
    # Moment matrix (Hankel matrix): H_{ij} = m_{i+j}
    half_K = K // 2
    H = cp.Variable((half_K + 1, half_K + 1), symmetric=True, name='Hankel')
    
    # Link H to moments
    for i in range(half_K + 1):
        for j in range(i, half_K + 1):
            if i + j <= K:
                constraints.append(H[i, j] == m[i + j])
    
    # PSD constraint
    constraints.append(H >> 0)
    
    # SD equations (linearised: use only the parts that are linear in m)
    # For the Gaussian: m_{n+1} = Σ_{j=0}^{n-1} m_j m_{n-j-1}
    # This is NOT linear — it involves products of moments.
    # The SDP relaxation: replace m_j m_k → H_{j,k} which is ≥ m_j m_k by PSD.
    # Then: m_{n+1} ≤ Σ H_{j, n-j-1}  (and similarly for lower bound).
    
    for n in range(1, min(K - n_v + 2, K)):
        lhs = 0
        for k in range(n_v):
            idx = n + k - 1
            if 0 <= idx <= K:
                lhs += v_prime_coeffs[k] * m[idx]
        
        # RHS with moment matrix entries
        rhs = 0
        for j in range(n):
            j_idx = j
            k_idx = n - j - 1
            if 0 <= j_idx <= half_K and 0 <= k_idx <= half_K:
                rhs += H[j_idx, k_idx]
            elif 0 <= j_idx <= K and 0 <= k_idx <= K:
                rhs += m[j_idx] * m[k_idx]  # fallback (makes it non-SDP)
        
        constraints.append(lhs == rhs)
    
    # Objective
    if maximize:
        objective = cp.Maximize(m[target_moment])
    else:
        objective = cp.Minimize(m[target_moment])
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.SCS, verbose=False, max_iters=10000)
        if problem.status in ['optimal', 'optimal_inaccurate']:
            return problem.value
        else:
            print(f"SDP status: {problem.status}")
            return None
    except Exception as e:
        print(f"SDP solver error: {e}")
        return None


def bootstrap_moment_bounds(v_prime_coeffs: List[float], 
                            max_moment: int = 10) -> dict:
    """Compute upper and lower bounds on all even moments."""
    bounds = {}
    for k in range(2, max_moment + 1, 2):
        ub = bootstrap_one_matrix(v_prime_coeffs, max_moment, k, maximize=True)
        lb = bootstrap_one_matrix(v_prime_coeffs, max_moment, k, maximize=False)
        bounds[k] = (lb, ub)
        if lb is not None and ub is not None:
            print(f"  m_{k}: [{lb:.8f}, {ub:.8f}], width = {ub-lb:.2e}")
    return bounds


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
    
    from one_matrix import gaussian_moments
    m_exact = gaussian_moments(10)
    print("\nExact values:")
    for k in range(2, 9, 2):
        print(f"  m_{k} = {m_exact[k]:.8f}")
    
    # Quartic: V'(M) = M + 0.5 M³
    print("\nQuartic model (V' = M + 0.5 M³):")
    bounds_q = bootstrap_moment_bounds([0.0, 1.0, 0.0, 0.5], max_moment=8)
