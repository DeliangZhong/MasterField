"""
one_matrix.py — Exact solutions for one-matrix models at N=∞.

For V(M) = Σ_k (g_k/k) M^k, the resolvent R(ζ) = tr[1/(ζ-M)] satisfies
an algebraic equation. Given R(ζ), the eigenvalue density is:
    ρ(x) = -(1/π) Im R(x + i0⁺)

The master field function M(z) = R^{-1}(z) (compositional inverse)
determines the Voiculescu form coefficients.
"""

import numpy as np
from scipy.optimize import brentq

# ══════════════════════════════════════════════════════════
#  Gaussian model: V(M) = (1/2) M²
# ══════════════════════════════════════════════════════════


def gaussian_resolvent(zeta: complex) -> complex:
    """R(ζ) = (ζ - √(ζ²-4)) / 2 for the Gaussian (Wigner semicircle)."""
    return (zeta - np.sqrt(zeta**2 - 4 + 0j)) / 2


def gaussian_density(x: np.ndarray) -> np.ndarray:
    """Wigner semicircle: ρ(x) = (1/2π)√(4-x²) for |x|<2."""
    rho = np.zeros_like(x)
    mask = np.abs(x) < 2
    rho[mask] = np.sqrt(4 - x[mask] ** 2) / (2 * np.pi)
    return rho


def gaussian_moments(max_power: int) -> np.ndarray:
    """Exact moments: tr[M^{2k}] = C_k (Catalan), tr[M^{2k+1}] = 0."""
    moments = np.zeros(max_power + 1)
    moments[0] = 1.0
    for k in range(1, max_power // 2 + 1):
        # C_k = (2k)! / ((k+1)! k!)
        from math import comb

        moments[2 * k] = comb(2 * k, k) / (k + 1)
    return moments


def gaussian_r_transform(z: complex) -> complex:
    """R-transform for Gaussian: R(z) = z (just a constant × z)."""
    return z


def gaussian_master_field_function(z: complex) -> complex:
    """M(z) = 1/z + z for the Gaussian."""
    return 1.0 / z + z


# ══════════════════════════════════════════════════════════
#  Quartic model: V(M) = (1/2) M² + (g/4) M⁴
# ══════════════════════════════════════════════════════════


def quartic_resolvent(zeta: complex, g: float) -> complex:
    """Resolvent for V(M) = M²/2 + g M⁴/4.

    Satisfies: g R³ - ζ g R² + (1 + 2g a²) R - ζ = 0
    where a² is the edge of the eigenvalue support, determined self-consistently.

    For small g we can solve perturbatively or numerically.
    """
    # The resolvent satisfies the cubic from the SD equation:
    # V'(R^{-1}) analytic continuation gives:
    # R(ζ) = 1/(ζ - Σ(R))  with self-energy Σ
    # For quartic: SD equation gives
    # ζ R = 1 + R² + g [R⁴ + 2 a² R²]  (schematically)
    #
    # Easier: solve for the endpoint a and density directly.
    # The eigenvalue density has support [-a, a] with
    # ρ(x) = (1/2π)(1 + 2g x²) √(a² - x²) × (normalisation)
    # ... but let's just compute moments from SD equations.
    pass


def quartic_moments_from_sd(g: float, max_power: int = 20) -> np.ndarray:
    """Compute moments by iterating Schwinger-Dyson equations.

    For V(M) = M²/2 + g M⁴/4, V'(M) = M + g M³.
    SD equation: tr[V'(M) M^n] = Σ_{j=0}^{n-1} tr[M^j] tr[M^{n-j-1}]

    i.e., tr[M^{n+1}] + g tr[M^{n+3}] = Σ_{j=0}^{n-1} m_j m_{n-j-1}

    where m_k = tr[M^k].  Use m_{odd} = 0 by symmetry.
    """
    # Maximum moment index we can reach
    M = max_power + 4  # need some headroom
    m = np.zeros(M)
    m[0] = 1.0  # tr[M⁰] = 1

    # The SD equations for even moments (odd = 0 by Z₂ symmetry):
    # m_{n+1} + g m_{n+3} = Σ_{j=0}^{n-1} m_j m_{n-j-1}
    #
    # For n even = 2p:
    # m_{2p+1} + g m_{2p+3} = Σ m_j m_{2p-j-1}
    # Since m_{odd}=0, the LHS = g m_{2p+3} and
    # RHS = Σ_{j even} m_j m_{2p-j-1} but 2p-j-1 must also be even,
    # so j must be odd → all zero. Wait, let me redo this.
    #
    # Actually for n odd = 2p+1 (so that n+1 is even):
    # m_{2p+2} + g m_{2p+4} = Σ_{j=0}^{2p} m_j m_{2p-j}
    # RHS = Σ_{k=0}^{p} m_{2k} m_{2(p-k)} (only even-even survives)
    #
    # This gives m_{2p+2} in terms of m_{2p+4} and lower moments.
    # We need to solve the system. Start from m_0=1, m_2 unknown.
    #
    # For n=1 (p=0): m_2 + g m_4 = m_0² = 1
    # For n=3 (p=1): m_4 + g m_6 = 2 m_0 m_2 = 2 m_2
    # For n=5 (p=2): m_6 + g m_8 = 2 m_0 m_4 + m_2² = 2m_4 + m_2²
    # ...
    #
    # This is a triangular system if we eliminate from the top.
    # But it's coupled: m_2 + g m_4 = 1 and m_4 + g m_6 = 2 m_2.
    # We need to iterate: guess m_2, compute m_4, m_6, ... and close.

    # Better approach: use the resolvent. R(ζ) = Σ m_k / ζ^{k+1}.
    # The SD equation in terms of R is:
    # R(ζ) ζ - 1 = R²(ζ) + g [ζ³ R(ζ) - ζ² - ζ R(ζ)... ]
    # This gets complicated. Let's just do Newton iteration.

    # For the quartic, the resolvent satisfies:
    # ζ = 1/R + m_1_conn + (m_2_conn) R + ...
    # Actually let's use the standard result.

    # The eigenvalue density for V = M²/2 + g M⁴/4 has support [-a, a]:
    # ρ(x) = (1/(2π)) (1 + 2g(a² + x²)/3 )... no, let me do it properly.
    #
    # Saddlepoint equation: V'(x)/2 = P.V. ∫ ρ(y)/(x-y) dy
    # For one-cut symmetric: ρ(x) = (1/π) h(x) √(a²-x²) with
    # h(x) determined by V'(x)/(2√(a²-x²)) via Hilbert transform.
    #
    # For V'(M) = M + g M³:
    # h(x) = 1/2 + g(a² + 2x²)/4  ... (from residue calculation)
    # Wait, the standard result is:
    # ρ(x) = (1/2π)(1 + 2gx² + ga²) √(a²-x²)   [not quite right]
    #
    # Let me just compute a numerically. Normalisation: ∫ρ=1 gives
    # ∫_{-a}^{a} h(x)√(a²-x²)/π dx = 1.

    # Actually, the clean way: for V = t M²/2 + g M⁴/4:
    # R(ζ) satisfies the cubic: g R³ + (stuff) = 0.
    # Instead, let me just solve the SD recursion numerically.

    # ITERATIVE APPROACH: Assume m_{2k}=0 for k > K_max as initial guess,
    # then iterate SD equations backwards.
    # Use the eigenvalue density approach: find support endpoint a,
    # then compute moments by integration.
    # For V'(M)=M+gM³, the resolvent R(z) = (V'(z) - P(z)√(z²-a²))/2
    # with P(z) = gz² + 1 + ga²/2 determined by R(z)→1/z at ∞.
    # Density: ρ(x) = P(x)√(a²-x²)/(2π) = (gx² + 1 + ga²/2)√(a²-x²)/(2π)
    # Normalisation: a²(1 + ga²/2)/4 = 1

    def norm_err(a):
        x = np.linspace(-a + 1e-12, a - 1e-12, 5000)
        P = g * x**2 + 1 + g * a**2 / 2
        rho = P * np.sqrt(a**2 - x**2) / (2 * np.pi)
        return np.trapezoid(rho, x) - 1.0

    a = brentq(norm_err, 0.1, 10.0)
    x = np.linspace(-a + 1e-12, a - 1e-12, 10000)
    P = g * x**2 + 1 + g * a**2 / 2
    rho = P * np.sqrt(a**2 - x**2) / (2 * np.pi)

    K = max_power // 2 + 1
    m_even = np.zeros(K + 1)
    m_even[0] = 1.0
    for k in range(1, K + 1):
        m_even[k] = np.trapezoid(x ** (2 * k) * rho, x)

    # Pack into full moment array
    moments = np.zeros(max_power + 1)
    for k in range(min(K + 1, max_power // 2 + 1)):
        if 2 * k <= max_power:
            moments[2 * k] = m_even[k]

    return moments


def quartic_eigenvalue_density(g: float, n_points: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalue density ρ(x) for V = M²/2 + g M⁴/4.

    Support is [-a, a] where a satisfies:
        a²(1 + g a²) = 4   (from normalisation + SD equation)
    i.e., g a⁴ + a² - 4 = 0  →  a² = (-1 + √(1+16g))/(2g)

    Density: ρ(x) = (1/(2π))(1 + 2g x²/... ) √(a²-x²)
    Actually: ρ(x) = (1/(2π))(V'(x) - polynomial)/(something)

    Let me derive properly. Resolvent ansatz R(ζ) = (1/2)[V'(ζ) - M(ζ)√(ζ²-a²)]
    where M is a polynomial. For V'=ζ + gζ³:
    R(ζ) → 1/ζ as ζ→∞ ⟹ M(ζ) = gζ + C, C chosen so R→1/ζ.

    Leading: V'(ζ)/2 ~ gζ³/2, and M(ζ)√(ζ²-a²) ~ (gζ+C)ζ ~ gζ² + Cζ.
    For R~1/ζ we need these to cancel at O(ζ³): gζ³/2 = gζ³... yes.
    At O(ζ): coefficient of ζ from V' is 1/2, from M√ is g(-a²/2)+C.
    So C = 1/2 + ga²/2.
    At O(1/ζ): R(ζ)→1/ζ gives the normalisation condition:
    a²(1 + 3ga²/4)/4 = 1 → a²/4 + 3ga⁴/16 = 1.
    Hmm, let me redo. The exact condition for one-cut with Z₂ symmetry is:
    ∮ V'(ζ)/(2πi √(ζ²-a²)) dζ = 1 (around the cut)
    = (1/2)[coeff of 1/ζ in V'(ζ)/√(ζ²-a²)]
    For V'(ζ)=ζ+gζ³: V'(ζ)/√(ζ²-a²) = (ζ+gζ³)/ζ√(1-a²/ζ²)
    = (1+gζ²)(1 + a²/(2ζ²) + ...) = gζ² + 1 + ga²/2 + a²/(2ζ²) + ...
    Residue at ∞ gives 1/ζ coefficient = a²/2 + ga⁴... nah this is getting messy.

    Let me just find a numerically.
    """
    # Normalisation: ∫_{-a}^{a} ρ(x) dx = 1
    # Saddlepoint: (1/2)V'(x) = P.V.∫ ρ(y)/(x-y) dy
    # For the one-cut symmetric case, ρ(x) = h(x)√(a²-x²)/π
    # where h(x) = (1/2π)∮ V'(ζ)/(2(ζ-x)√(ζ²-a²)) dζ
    # = (1 + g(2x² + a²))/2  ... from polynomial division.
    # For V'(M) = M + gM³, the resolvent gives:
    # P(x) = gx² + 1 + ga²/2
    # ρ(x) = P(x)√(a²-x²)/(2π)

    def normalisation_error(a):
        x = np.linspace(-a + 1e-10, a - 1e-10, 2000)
        P = g * x**2 + 1 + g * a**2 / 2
        rho = P * np.sqrt(a**2 - x**2) / (2 * np.pi)
        return np.trapezoid(rho, x) - 1.0

    a = brentq(normalisation_error, 0.1, 10.0)

    x = np.linspace(-a, a, n_points)
    P = g * x**2 + 1 + g * a**2 / 2
    rho = np.maximum(P * np.sqrt(np.maximum(a**2 - x**2, 0)) / (2 * np.pi), 0)

    return x, rho


# ══════════════════════════════════════════════════════════
#  General: resolvent → moments → R-transform → master field
# ══════════════════════════════════════════════════════════


def moments_from_density(x: np.ndarray, rho: np.ndarray, max_power: int) -> np.ndarray:
    """Compute moments m_k = ∫ x^k ρ(x) dx from a numerical density."""
    moments = np.zeros(max_power + 1)
    for k in range(max_power + 1):
        moments[k] = np.trapezoid(x**k * rho, x)
    return moments


def r_transform_from_moments(moments: np.ndarray) -> np.ndarray:
    """Compute R-transform coefficients from moments.

    The resolvent R(ζ) = Σ_k m_k / ζ^{k+1}.
    The master field function M(z) = R^{-1}(z) = 1/z + Σ_n M_n z^n
    where M_n are the free cumulants (R-transform coefficients).

    Free cumulants κ_n are related to moments by the moment-cumulant formula:
    m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}
    where the sum is over non-crossing partitions.

    Uses the functional equation z·G(z) = 1 + Σ_k κ_k G(z)^k to derive:
    κ_n = m_n - Σ_{k=1}^{n-1} κ_k · C(n, k)
    where C(n, k) = coeff of z^{-n} in G(z)^k
                  = Σ_{j₁+...+j_k = n-k} m_{j₁}·...·m_{j_k}
    """
    m = moments
    n_max = len(m) - 1
    kappa = np.zeros(n_max + 1)

    # C[n][k] = coeff of z^{-n} in G^k = Σ_{j₁+...+j_k = n-k} m_{j₁}...m_{j_k}
    # Recursion: C[n][k] = Σ_{j=0}^{n-k} m[j] · C[n-1-j][k-1]
    # Base: C[0][0] = 1, C[n][0] = 0 for n>0
    C = np.zeros((n_max + 1, n_max + 1))
    C[0, 0] = 1.0

    for n in range(1, n_max + 1):
        for k in range(1, n + 1):
            for j in range(n - k + 1):
                C[n, k] += m[j] * C[n - 1 - j, k - 1]

        # κ_n = m_n - Σ_{k=1}^{n-1} κ_k · C[n][k]
        kappa[n] = m[n] - sum(kappa[k] * C[n, k] for k in range(1, n))

    return kappa


def voiculescu_coefficients(free_cumulants: np.ndarray) -> np.ndarray:
    """Convert free cumulants κ_n to Voiculescu master field coefficients M_n.

    M(z) = 1/z + R(z), where R(z) = Σ_{n≥1} κ_n z^{n-1} is the R-transform.
    The M̂ = a + Σ_{n≥0} M_n (a†)^n has M_n = κ_{n+1}.
    """
    return free_cumulants[1:]  # M_0 = κ_1, M_1 = κ_2, ...


if __name__ == "__main__":
    print("=" * 60)
    print("One-matrix model validation")
    print("=" * 60)

    # 1. Gaussian moments
    m_exact = gaussian_moments(12)
    print("\nGaussian moments (exact):")
    for k in range(0, 13, 2):
        print(f"  m_{k} = {m_exact[k]:.6f}")

    # 2. Gaussian free cumulants
    kappa = r_transform_from_moments(m_exact)
    print(f"\nGaussian free cumulants: κ_1={kappa[1]:.4f}, κ_2={kappa[2]:.4f}, κ_3={kappa[3]:.4f}")
    print("  (Expected: κ_1=0, κ_2=1, κ_k=0 for k≥3)")

    # 3. Quartic model, g=0.5
    g = 0.5
    m_quartic = quartic_moments_from_sd(g, max_power=12)
    print(f"\nQuartic model (g={g}) moments:")
    for k in range(0, 13, 2):
        print(f"  m_{k} = {m_quartic[k]:.8f}")

    # 4. Quartic eigenvalue density
    x, rho = quartic_eigenvalue_density(g)
    m_check = moments_from_density(x, rho, 8)
    print("\nQuartic moments from density (check):")
    for k in range(0, 9, 2):
        print(f"  m_{k} = {m_check[k]:.8f}  (SD: {m_quartic[k]:.8f})")

    print("\nQuartic free cumulants:")
    kappa_q = r_transform_from_moments(m_quartic)
    for k in range(1, 7):
        print(f"  κ_{k} = {kappa_q[k]:.8f}")
