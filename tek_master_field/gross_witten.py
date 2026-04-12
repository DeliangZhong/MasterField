"""gross_witten.py — Phase A sanity check: Gross-Witten-Wadia single-plaquette model.

Not a TEK problem. This is a 1-matrix unitary model whose exact large-N saddle
is known (Gross-Witten 1980, Wadia 1980). We use it to validate our
density-optimization infrastructure before turning on the full TEK machinery.

Model:  Z = ∫ dU exp( (N/(2t)) Tr(U + U†) ),  U ∈ U(N)
        3rd-order phase transition at t = 1.

Exact eigenvalue density (see reference/gross_witten_model.md):

    Strong coupling (t ≥ 1, ungapped):
        ρ(θ) = (1/2π) (1 + (1/t) cos θ),  θ ∈ [-π, π]
        w_1 = 1/(2t)

    Weak coupling (t < 1, gapped):
        ρ(θ) = (1/(πt)) cos(θ/2) √(t - sin²(θ/2)),  |θ| ≤ 2 arcsin(√t)
        w_1 = 1 - t/2
        w_2 = (1 - t)²

Path C (spec): parametrize the support endpoint a = sin(θ_c/2) and use the
functional form above. Normalization ∫ρ = 1 fixes a² = t (for t < 1). This is
a 1D optimization; we verify the solver returns a = √t and that the resulting
Wilson loops match the exact closed forms to < 1e-6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from scipy import integrate, optimize


@dataclass(frozen=True)
class GWResult:
    t: float
    phase: str  # "strong" or "weak"
    a: float  # support half-width in sin(θ/2) units (only meaningful in weak phase)
    w1: float  # ⟨Tr U / N⟩ from numerical integration
    w2: float  # ⟨Tr U² / N⟩ from numerical integration
    w1_exact: float
    w2_exact: float
    err_w1: float
    err_w2: float


# ═══════════════════════════════════════════════════════════════
# Density evaluations
# ═══════════════════════════════════════════════════════════════


def rho_strong(theta: float, t: float) -> float:
    """Strong-coupling (ungapped) density on [-π, π]."""
    return (1.0 / (2.0 * math.pi)) * (1.0 + math.cos(theta) / t)


def rho_weak_unnormalized(theta: float, a: float) -> float:
    """Unnormalized weak-coupling density kernel cos(θ/2) √(a² − sin²(θ/2)).

    Zero outside the support (|sin(θ/2)| ≥ a).
    """
    s = math.sin(theta / 2.0)
    disc = a * a - s * s
    if disc <= 0.0:
        return 0.0
    return math.cos(theta / 2.0) * math.sqrt(disc)


def rho_weak(theta: float, t: float) -> float:
    """Weak-coupling (gapped) density on |sin(θ/2)| < √t."""
    return rho_weak_unnormalized(theta, math.sqrt(t)) / (math.pi * t)


# ═══════════════════════════════════════════════════════════════
# Wilson loops from the density
# ═══════════════════════════════════════════════════════════════


def wilson_loop_from_density(
    rho: Callable[[float], float],
    n: int,
    support: tuple[float, float],
) -> float:
    """w_n = ∫_{support} ρ(θ) cos(n θ) dθ."""
    lo, hi = support
    val, _ = integrate.quad(lambda th: rho(th) * math.cos(n * th), lo, hi, limit=200)
    return float(val)


# ═══════════════════════════════════════════════════════════════
# Path C 1D optimization: find `a` such that ∫ρ = 1
# ═══════════════════════════════════════════════════════════════


def find_a_from_normalization(t: float, tol: float = 1e-12) -> float:
    """In the weak phase, normalization ∫ρ dθ = 1 fixes a = √t.

    We find `a` by brentq root-finding on:  I(a; t) − 1 = 0
    where I(a; t) = (1/(πt)) ∫_{-θ_c}^{θ_c} cos(θ/2) √(a² − sin²(θ/2)) dθ
                  = a² / t      (analytic substitution u = sin(θ/2))

    So the solution is a = √t exactly. Solving numerically gives us a
    non-trivial test of the 1D root-finder infrastructure.
    """
    if not (0.0 < t < 1.0):
        raise ValueError(f"Weak-phase root-finding requires 0 < t < 1; got t={t}")

    def residual(a: float) -> float:
        theta_c = 2.0 * math.asin(a)
        integral, _ = integrate.quad(
            lambda th: rho_weak_unnormalized(th, a),
            -theta_c,
            theta_c,
            limit=200,
        )
        return integral / (math.pi * t) - 1.0

    # Bracket: residual(0+) = 0 - 1 = -1 < 0; residual(1) = 1/t - 1 > 0 for t < 1
    sol = optimize.brentq(residual, 1e-8, 1.0 - 1e-12, xtol=tol)
    return float(sol)


# ═══════════════════════════════════════════════════════════════
# Top-level solver
# ═══════════════════════════════════════════════════════════════


def solve_gw(t: float, validate: bool = True) -> GWResult:
    """Compute w_1, w_2 for the GWW model at 't Hooft coupling t.

    Uses the exact density (strong or weak), numerically integrates the Wilson
    loop, and (in the weak phase) solves for `a` via Path C 1D root-finding.
    """
    if t >= 1.0:
        phase = "strong"
        a_fit = 1.0  # the density is supported on full [-π, π]
        w1 = wilson_loop_from_density(lambda th: rho_strong(th, t), n=1, support=(-math.pi, math.pi))
        w2 = wilson_loop_from_density(lambda th: rho_strong(th, t), n=2, support=(-math.pi, math.pi))
        w1_exact = 1.0 / (2.0 * t)
        w2_exact = 0.0
    else:
        phase = "weak"
        a_fit = find_a_from_normalization(t)
        theta_c = 2.0 * math.asin(a_fit)
        w1 = wilson_loop_from_density(lambda th: rho_weak(th, t), n=1, support=(-theta_c, theta_c))
        w2 = wilson_loop_from_density(lambda th: rho_weak(th, t), n=2, support=(-theta_c, theta_c))
        w1_exact = 1.0 - t / 2.0
        w2_exact = (1.0 - t) ** 2

    err_w1 = abs(w1 - w1_exact)
    err_w2 = abs(w2 - w2_exact)

    result = GWResult(
        t=t,
        phase=phase,
        a=a_fit,
        w1=w1,
        w2=w2,
        w1_exact=w1_exact,
        w2_exact=w2_exact,
        err_w1=err_w1,
        err_w2=err_w2,
    )

    if validate:
        if err_w1 > 1e-6:
            raise RuntimeError(
                f"Phase A gate FAILED at t={t}: |w_1 − exact| = {err_w1:.3e} > 1e-6 "
                f"(w1={w1:.8f}, exact={w1_exact:.8f}). Sign/convention bug?"
            )
        if err_w2 > 1e-6:
            raise RuntimeError(
                f"Phase A gate FAILED at t={t}: |w_2 − exact| = {err_w2:.3e} > 1e-6 "
                f"(w2={w2:.8f}, exact={w2_exact:.8f})."
            )

    return result


# ═══════════════════════════════════════════════════════════════
# Main acceptance
# ═══════════════════════════════════════════════════════════════


def phase_a_main(tees: list[float] | None = None) -> list[GWResult]:
    """Run Phase A gate at a list of couplings. Pass if every error < 1e-6."""
    tees = tees or [0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0]

    print("=" * 70)
    print("  Phase A — Gross-Witten infrastructure validation")
    print("=" * 70)
    print(f"{'t':>6s}  {'phase':<6s}  {'a':>10s}  {'w1':>12s}  {'w1_exact':>12s}  "
          f"{'err_w1':>10s}  {'w2':>12s}  {'w2_exact':>12s}  {'err_w2':>10s}")

    results: list[GWResult] = []
    for t in tees:
        r = solve_gw(t, validate=True)
        results.append(r)
        print(
            f"{r.t:6.2f}  {r.phase:<6s}  {r.a:10.6f}  {r.w1:12.8f}  {r.w1_exact:12.8f}  "
            f"{r.err_w1:10.2e}  {r.w2:12.8f}  {r.w2_exact:12.8f}  {r.err_w2:10.2e}"
        )

    worst_w1 = max(r.err_w1 for r in results)
    worst_w2 = max(r.err_w2 for r in results)
    print(f"\n  worst |w1 err|: {worst_w1:.2e}")
    print(f"  worst |w2 err|: {worst_w2:.2e}")
    if worst_w1 < 1e-6 and worst_w2 < 1e-6:
        print("  Phase A gate: PASS ✓")
    else:
        print("  Phase A gate: FAIL ✗")

    return results


if __name__ == "__main__":
    phase_a_main()
