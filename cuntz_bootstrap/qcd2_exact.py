"""Exact QCD_2 Wilson loops at large N via window decomposition.

At N = infinity on an infinite 2D lattice with Wilson action at coupling
lambda, the master field gives:

    W[C] = w_+^|Area(C)|              for simple (non-self-intersecting) loops
    W[C] = prod_i w_+^|Area(C_i)|     for C self-intersecting, C decomposed
                                         into simple windows C_i

with w_+ = 1/(2 lambda) at strong coupling (lambda >= 1)
     w_+ = 1 - lambda/2 at weak coupling (lambda < 1).

The Gopakumar-Gross window process (hep-th/9411021 §6): a self-intersecting
lattice loop can be repeatedly split at any self-intersection vertex into
two closed sub-loops, until only simple loops remain.

This module reuses `master_field/lattice.py` helpers for backtrack reduction
and signed area.
"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import reduce_backtracks, signed_area_2d  # noqa: E402


def gw_w_plus(lam: float) -> float:
    """w_+ = 1/(2 lambda) for lambda >= 1, 1 - lambda/2 for lambda < 1."""
    if lam >= 1.0:
        return 1.0 / (2.0 * lam)
    return 1.0 - lam / 2.0


def _step_to_delta(mu: int) -> tuple[int, int]:
    """Map signed direction mu in {+-1, +-2} to (dx, dy) 2D displacement."""
    if mu == 1:
        return (1, 0)
    if mu == -1:
        return (-1, 0)
    if mu == 2:
        return (0, 1)
    if mu == -2:
        return (0, -1)
    raise ValueError(f"Unsupported direction mu={mu}; qcd2_exact handles D=2 only")


def loop_to_vertices(loop: tuple[int, ...]) -> list[tuple[int, int]]:
    """Compute the vertex sequence v_0, v_1, ..., v_k with v_0 = (0,0)."""
    vs: list[tuple[int, int]] = [(0, 0)]
    for mu in loop:
        dx, dy = _step_to_delta(mu)
        x, y = vs[-1]
        vs.append((x + dx, y + dy))
    return vs


def signed_area(loop: tuple[int, ...]) -> int:
    """Shoelace signed area of the 2D lattice loop (must be closed)."""
    return signed_area_2d(loop)


def detect_self_intersection(
    loop: tuple[int, ...],
) -> tuple[int, int] | None:
    """Return the first (a, b) with 0 <= a < b <= len(loop), v_a = v_b, and
    (a, b) != (0, len(loop)). Vertices v_0 ... v_k where k = len(loop).

    Returns None if the loop is simple (only the endpoint self-intersection).
    """
    vs = loop_to_vertices(loop)
    k = len(loop)  # number of edges; vs has k+1 entries
    # Iterate (a, b) with a < b, exclude (0, k)
    for a in range(k + 1):
        for b in range(a + 1, k + 1):
            if (a, b) == (0, k):
                continue
            if vs[a] == vs[b]:
                return (a, b)
    return None


def window_decomposition(loop: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Recursively split a closed 2D lattice loop into simple sub-loops.

    Algorithm:
      1. If loop is empty or simple (only endpoint coincides with start),
         return [loop].
      2. Find any interior self-intersection (a, b).
      3. Split into A = loop[a:b]  and  B = loop[:a] + loop[b:].
         Both are closed because v_a = v_b.
      4. Recurse on each.

    Returns a list of simple loops (each may be empty after backtrack
    reduction).
    """
    if not loop:
        return []
    intersection = detect_self_intersection(loop)
    if intersection is None:
        return [loop]
    a, b = intersection
    A = loop[a:b]
    B = loop[:a] + loop[b:]
    out: list[tuple[int, ...]] = []
    # Recurse with backtrack reduction on each piece (they may reduce to
    # empty or simpler loops)
    for piece in (A, B):
        reduced = reduce_backtracks(tuple(piece))
        if not reduced:
            continue
        out.extend(window_decomposition(reduced))
    return out


def qcd2_wilson_loop(loop: tuple[int, ...], lam: float) -> float:
    """Exact QCD_2 Wilson loop at coupling lam via window decomposition."""
    reduced = reduce_backtracks(tuple(loop))
    if not reduced:
        return 1.0
    windows = window_decomposition(reduced)
    w_plus = gw_w_plus(lam)
    result = 1.0
    for W in windows:
        area = abs(signed_area(W))
        result *= w_plus ** area
    return result
