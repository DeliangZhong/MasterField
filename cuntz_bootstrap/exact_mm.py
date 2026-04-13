"""Direct lattice Makeenko-Migdal equations for the Wilson-action master field.

Derivation (sketch). Starting from Haar-measure left-invariance on link U_e:

    0 = ∫ dU_e (∂/∂U_e) [ (1/N) Tr(U_C) · exp(−S) ]

for Wilson action S = −(N/λ) Σ_{P} (1/N) Re Tr(U_P). At large N, factorisation
reduces the correlator identity to a RELATION ON SINGLE WILSON LOOPS plus
SELF-INTERSECTION FACTORISATIONS.

For loop C and edge e = +μ at position i ∈ {0, ..., |C|−1}:

    (1/λ) Σ_{plaquettes P through e} W[ C with e replaced by staple(P, e) ]
       =  c_self · W[C]  +  Σ_{self-intersection splits at v(e)} W[C_1] · W[C_2]

with c_self = 2 in candidate-D (leading-order in 1/λ — the form we use now;
exact form may introduce subleading corrections from the precise
Haar-derivative coefficients).

The `staple(P, e)` is the three-edge path from v(e) to v(e)+μ going around
plaquette P; there are 2·(D−1) plaquettes through e (one per plane (μ, ν)
with ν ≠ ±μ, each with two sides). For each plaquette, replace the edge
at position i with its staple, then cyclically canonicalise and reduce
backtracks to get the canonical replaced loop.

For D=2, only the (1, 2) plane exists and e = ±1 (or ±2) has two staples:
"above" and "below" the edge. For D>2, 2(D−1) staples per edge.

At N=∞, self-intersection contributions factor into products of smaller
Wilson loops via Gopakumar-Gross window decomposition (implemented in
`qcd2_exact.py`).

This module is pure Python; the W_func callable can be any Wilson-loop
evaluator (exact or Fock-space-computed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import (  # noqa: E402
    cyclic_canonical,
    enumerate_closed_loops,
    reduce_backtracks,
)

from .qcd2_exact import loop_to_vertices


def _axis_of(mu: int) -> int:
    """|μ| for a signed direction."""
    if mu == 0:
        raise ValueError("mu cannot be 0")
    return abs(mu)


def staple_replacement(
    loop: tuple[int, ...], edge_idx: int, D: int
) -> list[tuple[int, ...]]:
    """Return staple-replaced loops.

    For edge loop[edge_idx] = μ (signed), and each direction ν with
    1 ≤ |ν| ≤ D and |ν| ≠ |μ|, build the staple of length 3 connecting
    v(e) → v(e) + μ by going (ν, μ, −ν). Replace the edge with this
    3-edge path.

    Returns 2·(D−1) loops (one per (ν, μ) plane and orientation), each
    cyclically canonicalised + backtrack-reduced.
    """
    if not loop:
        return []
    mu = loop[edge_idx]
    abs_mu = _axis_of(mu)
    prefix = loop[:edge_idx]
    suffix = loop[edge_idx + 1 :]
    out: list[tuple[int, ...]] = []
    for abs_nu in range(1, D + 1):
        if abs_nu == abs_mu:
            continue
        for sign in (+1, -1):
            nu = sign * abs_nu
            staple = (nu, mu, -nu)
            replaced = prefix + staple + suffix
            reduced = reduce_backtracks(tuple(replaced))
            canon = cyclic_canonical(reduced)
            out.append(canon)
    return out


def split_pairs_at_vertex(
    loop: tuple[int, ...], edge_idx: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Find self-intersections of `loop` at v(e_start) = vertex BEFORE
    edge loop[edge_idx]. For each other index j where v_j equals v(e_start),
    the loop splits at that vertex into:

        C_1 = loop[edge_idx:j]   (arc from e_start to v_j)
        C_2 = loop[j:] + loop[:edge_idx]
                                 (the complement, closed at the same vertex)

    Both reduced + cyclically canonicalised. Returns list of (C_1, C_2) pairs.
    """
    vs = loop_to_vertices(loop)
    e_start = edge_idx  # vertex index before edge (= edge_idx)
    target_vertex = vs[e_start]
    out: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    n = len(loop)
    for j in range(n + 1):
        if j == e_start:
            continue
        if (j, e_start) == (n, 0) or (e_start, j) == (0, n):
            # trivial start=end coincidence, not an interior self-intersection
            continue
        if vs[j] == target_vertex:
            # Split: C1 from edge_idx..j, C2 = rest
            if j > e_start:
                c1 = loop[e_start:j]
                c2 = loop[j:] + loop[:e_start]
            else:
                c1 = loop[e_start:] + loop[:j]
                c2 = loop[j:e_start]
            c1_red = cyclic_canonical(reduce_backtracks(tuple(c1)))
            c2_red = cyclic_canonical(reduce_backtracks(tuple(c2)))
            out.append((c1_red, c2_red))
    # De-duplicate (each split counted once)
    uniq: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    seen: set = set()
    for pair in out:
        key = tuple(sorted([pair[0], pair[1]]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(pair)
    return uniq


def mm_direct_residual(
    loop: tuple[int, ...],
    edge_idx: int,
    D: int,
    W_func: Callable[[tuple[int, ...]], float],
    lam: float,
    c_self: float = 2.0,
) -> float:
    """LHS − RHS of the direct MM equation at edge edge_idx of loop.

    LHS = (1/λ) Σ W[staple-replaced]
    RHS = c_self · W[loop] + Σ_{splits} W[C_1] · W[C_2]
    """
    staples = staple_replacement(loop, edge_idx, D)
    lhs = sum(W_func(C) for C in staples) / lam
    rhs = c_self * W_func(loop)
    for (C1, C2) in split_pairs_at_vertex(loop, edge_idx):
        rhs += W_func(C1) * W_func(C2)
    return lhs - rhs


def enumerate_loops(D: int, L_max: int) -> list[tuple[int, ...]]:
    """Non-backtracking closed loops up to length L_max, canonicalised."""
    return enumerate_closed_loops(D, L_max)
