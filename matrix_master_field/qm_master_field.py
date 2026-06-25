# matrix_master_field/qm_master_field.py
"""M5c — master fields for two-matrix QM (HHK Eq 17).

C2 gaussian_master_field: explicit Gaussian trial state |ψ_G(Ω)⟩, ⟨H⟩ by Wick →
rigorous upper bound (NOT via Φ*). C3 free-Fisher operator field is added in a later
task. See docs/superpowers/specs/2026-06-25-m5c-two-matrix-qm-design.md (C2/C3).
"""
import numpy as np
from scipy import optimize


def gaussian_comm_moment(omega, N):
    """Wick ⟨Tr[X,Y]²⟩ in the Gaussian ground state of Tr(P²+Ω²X²) (X,Y independent).

    ⟨X_ij X_kl⟩=a δ_il δ_jk, a=1/(2Ω): ⟨Tr XYXY⟩=a²N, ⟨Tr XYYX⟩=a²N³ ⇒
    ⟨Tr[X,Y]²⟩ = 2a²N − 2a²N³ = 2a²N(1−N²)  (leading large-N: −N³/(2Ω²)).
    """
    a = 1.0 / (2.0 * omega)
    return 2.0 * a**2 * N * (1.0 - N**2)


def gaussian_master_field(m, lam):
    """Variational E/N² over the Gaussian family: min_Ω [Ω + m²/Ω + λ/(2Ω²)].

    A rigorous upper bound (variational principle on the explicit state |ψ_G(Ω)⟩).
    Returns dict(energy, omega, m2=1/(2Ω)).

    The stationarity condition f'(Ω)=0 is Ω³ − m²Ω − λ = 0, solved via brentq on
    the derivative 1 − m²/Ω² − λ/Ω³. At λ=0 the root is Ω=m exactly.
    """
    def f(omega):
        return omega + m**2 / omega + lam / (2.0 * omega**2)

    if lam == 0.0:
        omega = float(m)
    else:
        # f'(Ω) = 1 - m²/Ω² - λ/Ω³; unique positive root for λ>0.
        def df(omega):
            return 1.0 - m**2 / omega**2 - lam / omega**3

        # Upper bracket: for large Ω, df→1>0; lower bracket: df→-∞ for Ω→0+.
        upper = 10.0 + 10.0 * (m + lam)
        omega = float(optimize.brentq(df, 1e-8, upper))

    return {"energy": float(f(omega)), "omega": omega, "m2": 1.0 / (2.0 * omega)}
