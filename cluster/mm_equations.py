"""
mm_equations.py — Lattice Makeenko-Migdal equation machinery.

The lattice MM equation at large N, for Wilson action S = -(N/(2λ)) Σ_P (Tr U_P + Tr U_P†),
relates Wilson loops at different lengths. For each edge e_j in a closed loop C:

    (1/λ) Σ_{P ∋ e_j} W[staple_P(C, e_j)] = W[C] · (# plaquettes) - (splits)    (candidate form)

or other closely related forms. The exact coefficient structure is convention-dependent.

THIS MODULE IMPLEMENTS AND TESTS MULTIPLE CANDIDATE FORMS against the exactly known D=2
large-N lattice answer (the Gross-Witten single-plaquette model: w_+ = 1/(2λ) for λ > 1,
and W[m×n] = w_+^{mn} for simple rectangles on an infinite 2D lattice).

The WINNING candidate is the one whose residuals vanish on every simple (non-self-
intersecting) loop for all couplings. Until then, this module is experimental.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from lattice import (
    abs_area_2d,
    cyclic_canonical,
    enumerate_nonself_intersecting_2d,
    plaquette_insertions,
    reduce_backtracks,
    self_intersection_splits,
)

# ═══════════════════════════════════════════════════════════
# Candidate MM equation forms
# ═══════════════════════════════════════════════════════════


def _area_law_2d(C: tuple[int, ...], lam: float) -> float:
    """Exact area-law Wilson loop for a non-self-intersecting 2D loop.

    On the continuum limit / infinite lattice of 2D YM, W[C] = exp(-λ Area / 2).
    On the finite lattice with Wilson action, W[C] = w_+^{Area} where w_+ is the
    single-plaquette GW-strong-coupling value 1/(2λ) for λ ≥ 1.
    """
    C_r = cyclic_canonical(reduce_backtracks(C))
    if not C_r:
        return 1.0
    return math.exp(-lam * abs_area_2d(C_r) / 2)


def _single_plaquette_lattice_2d(C: tuple[int, ...], w_plus: float) -> float:
    """Lattice-exact Wilson loop for simple 2D loops: W[C] = w_+^{Area}."""
    C_r = cyclic_canonical(reduce_backtracks(C))
    if not C_r:
        return 1.0
    return w_plus ** abs_area_2d(C_r)


def mm_residual_staple(
    loop: tuple[int, ...],
    edge_idx: int,
    D: int,
    lam: float,
    W: Callable[[tuple[int, ...]], float],
    include_self_closure: bool = True,
) -> float:
    """Candidate MM residual with STAPLE-replacement convention.

    LHS = (1/λ) Σ_{P ∋ e_j} W[staple(C, e_j, P)]
    RHS = W[C] (if include_self_closure else 0) + Σ_{non-trivial splits} W[C1] W[C2]

    Returns LHS - RHS.
    """
    insertions = plaquette_insertions(loop, edge_idx, D)
    lhs = sum(W(insert) for insert in insertions) / lam

    rhs = W(loop) if include_self_closure else 0.0
    for c1, c2 in self_intersection_splits(loop):
        rhs += W(c1) * W(c2)

    return lhs - rhs


# ═══════════════════════════════════════════════════════════
# Scan candidate equations to find the one that matches the area law
# ═══════════════════════════════════════════════════════════


def gw_w_plus(lam: float) -> float:
    """Gross-Witten single-plaquette expectation value at large N.

    On an infinite 2D lattice with Wilson action, each plaquette is GW-distributed
    (plaquettes are independent at N=∞ due to 2D area-exactness).
    - Strong coupling (λ ≥ 1): w_+ = 1/(2λ)  [exact; higher Wilson loops in GW are zero]
    - Weak coupling (λ < 1):   w_+ = 1 - λ/2
    """
    if lam >= 1.0:
        return 1.0 / (2.0 * lam)
    return 1.0 - lam / 2.0


def scan_candidates_2d(
    max_length: int = 6,
    lams: list[float] | None = None,
    target: str = "lattice",
) -> dict:
    """For each candidate MM form, report max residual over all simple 2D loops.

    target = "lattice": use W[C] = w_+^Area with w_+ = GW value (this is the
                        EXACT answer on the infinite 2D lattice at N=∞).
    target = "continuum": use W[C] = exp(-λ Area / 2) (continuum limit only).
    """
    lams = lams or [1.5, 2.0, 5.0, 10.0]  # stay in strong-coupling phase
    loops = enumerate_nonself_intersecting_2d(max_length)

    def make_W(lam, target):
        if target == "lattice":
            w_plus = gw_w_plus(lam)

            def W(C, wp=w_plus):
                return _single_plaquette_lattice_2d(C, wp)
        else:

            def W(C, lam=lam):
                return _area_law_2d(C, lam)

        return W

    results: dict = {}
    for label, eq in _candidate_catalog().items():
        per_lam: dict = {}
        for lam in lams:
            W = make_W(lam, target)
            residuals = []
            for C in loops:
                for j in range(len(C)):
                    r = eq(C, j, 2, lam, W)
                    residuals.append(abs(r))
            per_lam[lam] = {"max": max(residuals), "mean": sum(residuals) / len(residuals)}
        results[label] = per_lam
    return results


def _candidate_catalog() -> dict[str, Callable]:
    """Collection of candidate MM equation forms to test."""

    def A_staple_with_closure(C, j, D, lam, W):
        return mm_residual_staple(C, j, D, lam, W, include_self_closure=True)

    def B_staple_no_closure(C, j, D, lam, W):
        return mm_residual_staple(C, j, D, lam, W, include_self_closure=False)

    def C_staple_lhs_times_2(C, j, D, lam, W):
        lhs = 2.0 * sum(W(i) for i in plaquette_insertions(C, j, D)) / lam
        rhs = W(C) + sum(W(c1) * W(c2) for c1, c2 in self_intersection_splits(C))
        return lhs - rhs

    def D_staple_rhs_times_2(C, j, D, lam, W):
        lhs = sum(W(i) for i in plaquette_insertions(C, j, D)) / lam
        rhs = 2.0 * W(C) + sum(W(c1) * W(c2) for c1, c2 in self_intersection_splits(C))
        return lhs - rhs

    def E_signed_staples(C, j, D, lam, W):
        """Try with ALTERNATING signs for the two perpendicular directions.

        In the MM derivation, variation with respect to U_e (left) and U_e (right) can
        produce terms with opposite signs from U_P and U_P†. Effectively:
          (1/λ)(W[+ν staple] - W[-ν staple]) = ...
        """
        insertions = plaquette_insertions(C, j, D)
        # Alternate sign by ν orientation: first is +ν, second is -ν (per our construction)
        lhs = sum((+1 if k % 2 == 0 else -1) * W(ins) for k, ins in enumerate(insertions)) / lam
        rhs = W(C) + sum(W(c1) * W(c2) for c1, c2 in self_intersection_splits(C))
        return lhs - rhs

    def F_staple_trivial_only(C, j, D, lam, W):
        """Only keep staples that genuinely EXTEND the loop (filter out ones that
        reduce to empty or to the original loop)."""
        C_canon = cyclic_canonical(reduce_backtracks(C))
        insertions = plaquette_insertions(C, j, D)
        non_trivial = [
            ins
            for ins in insertions
            if cyclic_canonical(reduce_backtracks(ins)) not in ((), C_canon)
        ]
        lhs = sum(W(ins) for ins in non_trivial) / lam
        rhs = W(C) + sum(W(c1) * W(c2) for c1, c2 in self_intersection_splits(C))
        return lhs - rhs

    def G_staple_2D_no_closure(C, j, D, lam, W):
        """Candidate D without the self-closure term (for non-self-intersecting C,
        RHS = 0 instead of 2 W[C])."""
        lhs = sum(W(i) for i in plaquette_insertions(C, j, D)) / lam
        rhs = sum(W(c1) * W(c2) for c1, c2 in self_intersection_splits(C))
        return lhs - rhs

    return {
        "A (staple, self-closure)": A_staple_with_closure,
        "B (staple, no closure)": B_staple_no_closure,
        "C (staple, LHS ×2)": C_staple_lhs_times_2,
        "D (staple, RHS ×2)": D_staple_rhs_times_2,
        "E (staple, signed)": E_signed_staples,
        "F (staple, non-trivial only)": F_staple_trivial_only,
        "G (LHS=Σ/λ, RHS=splits)": G_staple_2D_no_closure,
    }


# ═══════════════════════════════════════════════════════════
# Main: run a scan if called directly
# ═══════════════════════════════════════════════════════════


def scan_report(results: dict) -> None:
    """Pretty-print the scan results."""
    print("\nCandidate | λ=0.5 | λ=1.0 | λ=2.0 | λ=5.0")
    print("-" * 70)
    for label, per_lam in results.items():
        row = f"{label:50s}"
        for lam, stats in per_lam.items():
            row += f"  {stats['max']:.2e}"
        print(row)


if __name__ == "__main__":
    print("=" * 70)
    print("  MM Equation Candidate Scan — D=2")
    print("=" * 70)
    print("\n>>> Target: LATTICE (W[C] = w_+^Area with GW strong-coupling w_+)")
    results_lattice = scan_candidates_2d(max_length=6, target="lattice")
    scan_report(results_lattice)
    print("\n>>> Target: CONTINUUM (W[C] = exp(-λ Area / 2))")
    results_cont = scan_candidates_2d(max_length=6, target="continuum")
    scan_report(results_cont)
