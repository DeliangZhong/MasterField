"""
qcd2.py — 2D lattice Yang-Mills master field solver (Phase 0 validation).

In 2D, the exact large-N Wilson loop average for a non-self-intersecting loop is:
    W[C] = exp(-λ · Area(C) / 2)
where λ is the 't Hooft coupling. In axial gauge, the master field is Gaussian:
    Û_μ = exp(iα (â_μ + â_μ†))
with a single parameter α(λ) fixed by matching the plaquette.

This module:
1. Solves for α(λ) by matching W[□] = exp(-λ/2) at a given Fock truncation L.
2. Validates the result against the exact area law for larger loops.
3. Validates the Makeenko-Migdal equation residuals.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from cuntz_fock import CuntzFockSpace
from lattice import (
    abs_area_2d,
    cyclic_canonical,
    enumerate_nonself_intersecting_2d,
    plaquette_insertions,
    reduce_backtracks,
    self_intersection_splits,
)

PLAQUETTE: tuple[int, ...] = (1, 2, -1, -2)


def _plaquette_vev(fock: CuntzFockSpace, alpha: float) -> complex:
    """W[□]_ML at given α: <Ω| Û_1 Û_2 Û_1† Û_2† |Ω>."""
    U1 = fock.build_unitary_gaussian(alpha, matrix_idx=0)
    U2 = fock.build_unitary_gaussian(alpha, matrix_idx=1)
    ops = {1: U1, 2: U2}
    return fock.wilson_loop_vev(ops, PLAQUETTE)


def solve_alpha_for_plaquette(
    lam: float,
    fock: CuntzFockSpace,
    bracket: tuple[float, float] = (1e-6, math.pi / 2),
    tol: float = 1e-12,
) -> float:
    """Find α such that W[□]_ML(α) = exp(-λ/2)."""
    target = math.exp(-lam / 2)

    def residual(a: float) -> float:
        val = _plaquette_vev(fock, a).real
        return val - target

    lo, hi = bracket
    r_lo = residual(lo)
    r_hi = residual(hi)
    if r_lo * r_hi > 0:
        # Widen the search: at α=0, W[□]=1 > target; at α=π/2 it should be below target.
        raise RuntimeError(
            f"No sign change in bracket {bracket}: residual(lo)={r_lo:.4e}, residual(hi)={r_hi:.4e}"
        )
    return brentq(residual, lo, hi, xtol=tol, rtol=tol)


def _build_operators(fock: CuntzFockSpace, alpha: float) -> dict[int, np.ndarray]:
    return {
        1: fock.build_unitary_gaussian(alpha, matrix_idx=0),
        2: fock.build_unitary_gaussian(alpha, matrix_idx=1),
    }


def validate_wilson_loops(
    lam: float,
    fock: CuntzFockSpace,
    max_loop_length: int = 8,
) -> dict:
    """Compute Wilson loops for all non-self-intersecting 2D loops up to given length
    and compare against the exact area law.

    Returns a dict with 'alpha', 'loops' (list of per-loop dicts), and summary errors.
    """
    alpha = solve_alpha_for_plaquette(lam, fock)
    ops = _build_operators(fock, alpha)

    loops_data = []
    for word in enumerate_nonself_intersecting_2d(max_loop_length):
        area = abs_area_2d(word)
        w_ml = fock.wilson_loop_vev(ops, word).real
        w_exact = math.exp(-lam * area / 2)
        loops_data.append(
            {
                "word": word,
                "length": len(word),
                "area": area,
                "w_ml": w_ml,
                "w_exact": w_exact,
                "error": abs(w_ml - w_exact),
            }
        )

    errors = [d["error"] for d in loops_data]
    return {
        "alpha": alpha,
        "lambda": lam,
        "L": fock.L,
        "n_loops": len(loops_data),
        "max_error": max(errors) if errors else 0.0,
        "mean_error": (sum(errors) / len(errors)) if errors else 0.0,
        "loops": loops_data,
    }


def mm_residual(
    word: tuple[int, ...],
    e_idx: int,
    lam: float,
    w_of: callable,
    D: int = 2,
) -> float:
    """λ W[C] − ∑_{P ∋ e} W[P_e ∘ C] + ∑_{splits} W[C1] W[C2]

    w_of: callable (tuple) → complex. Returns Wilson loop value for a canonical word.
    """
    lhs = lam * w_of(word).real
    plaq_sum = sum(w_of(p).real for p in plaquette_insertions(word, e_idx, D))
    split_sum = sum((w_of(c1) * w_of(c2)).real for c1, c2 in self_intersection_splits(word))
    return lhs - plaq_sum + split_sum


def validate_mm_equation_exact(
    lam: float,
    max_loop_length: int = 6,
) -> dict:
    """Check the MM equation residual against the EXACT area-law Wilson loop.

    This tests whether our MM implementation is correct, independent of the Fock
    space truncation. Strictly speaking the area law is exact only for non-self-
    intersecting loops, so we test only those and avoid splits.
    """

    def w_exact(word: tuple[int, ...]) -> complex:
        word_r = cyclic_canonical(reduce_backtracks(word))
        if not word_r:
            return 1.0 + 0j
        try:
            area = abs_area_2d(word_r)
        except ValueError:
            return 0.0 + 0j
        return math.exp(-lam * area / 2) + 0j

    # For each non-self-intersecting loop, evaluate MM residual at link e_idx=0
    residuals = []
    for word in enumerate_nonself_intersecting_2d(max_loop_length):
        r = mm_residual(word, 0, lam, w_exact, D=2)
        residuals.append({"word": word, "residual": r})

    abs_r = [abs(d["residual"]) for d in residuals]
    return {
        "lambda": lam,
        "n_tests": len(residuals),
        "max_residual": max(abs_r) if abs_r else 0.0,
        "mean_residual": (sum(abs_r) / len(abs_r)) if abs_r else 0.0,
        "residuals": residuals,
    }


def qcd2_main(Ls: list[int] | None = None, lams: list[float] | None = None) -> None:
    """Phase 0 acceptance test: sweep λ and L, report errors."""
    Ls = Ls or [4, 6]
    lams = lams or [0.5, 1.0, 2.0, 5.0]

    print("=" * 70)
    print("  QCD₂ Master Field — Phase 0 Validation")
    print("=" * 70)

    for L in Ls:
        fock = CuntzFockSpace(n_matrices=2, max_length=L)
        print(f"\n--- Fock space: n=2, L={L}, dim={fock.dim} ---")
        for lam in lams:
            try:
                res = validate_wilson_loops(lam, fock, max_loop_length=min(8, L + 2))
                # Also check unitarity
                U1 = fock.build_unitary_gaussian(res["alpha"], 0)
                ok, uerr = fock.check_unitarity(U1, tol=1e-4)
                print(
                    f"  λ={lam:4.1f}  α={res['alpha']:.6f}  n_loops={res['n_loops']:4d}  "
                    f"max_err={res['max_error']:.2e}  mean_err={res['mean_error']:.2e}  "
                    f"‖UU†-I‖={uerr:.2e}"
                )
            except RuntimeError as e:
                print(f"  λ={lam:4.1f}  FAILED: {e}")

    print("\n--- MM equation check (against exact area law) ---")
    for lam in lams:
        mm = validate_mm_equation_exact(lam, max_loop_length=6)
        print(
            f"  λ={lam:4.1f}  n_tests={mm['n_tests']:3d}  "
            f"max_res={mm['max_residual']:.2e}  mean_res={mm['mean_residual']:.2e}"
        )


if __name__ == "__main__":
    qcd2_main()
