"""
lattice.py — Lattice Wilson loop machinery for the master field program.

Words encode closed loops on a D-dimensional hypercubic lattice. A step μ ∈ {±1,...,±D}
means: forward (μ>0) or backward (μ<0) in direction |μ|. A closed loop has steps
summing to zero.

Equivalences:
- Cyclic rotation (trace cyclicity)
- Backtrack reduction (unitarity: U_μ U_μ^dagger = I means adjacent μ,-μ cancel)
- Lattice symmetry (hyperoctahedral group B_D)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

# ═══════════════════════════════════════════════════════════
# Basic loop operations
# ═══════════════════════════════════════════════════════════


def reduce_backtracks(word: tuple[int, ...]) -> tuple[int, ...]:
    """Iteratively remove adjacent (μ, -μ) pairs.

    Also handles the cyclic wrap: if the first and last elements cancel,
    they are removed.
    """
    w = list(word)
    changed = True
    while changed and w:
        changed = False
        # Linear sweep
        i = 0
        while i < len(w) - 1:
            if w[i] == -w[i + 1]:
                del w[i : i + 2]
                changed = True
                # step back one in case a new backtrack formed
                if i > 0:
                    i -= 1
            else:
                i += 1
        # Cyclic wrap-around cancellation
        if len(w) >= 2 and w[0] == -w[-1]:
            w = w[1:-1]
            changed = True
    return tuple(w)


def cyclic_canonical(word: tuple[int, ...]) -> tuple[int, ...]:
    """Lexicographically smallest cyclic rotation."""
    if not word:
        return word
    n = len(word)
    rotations = [word[i:] + word[:i] for i in range(n)]
    return min(rotations)


def is_closed(word: tuple[int, ...], D: int) -> bool:
    """Check that the loop returns to origin (net displacement = 0)."""
    disp = [0] * D
    for step in word:
        if step > 0:
            disp[step - 1] += 1
        else:
            disp[-step - 1] -= 1
    return all(d == 0 for d in disp)


def signed_area_2d(word: tuple[int, ...]) -> int:
    """Signed enclosed lattice area in 2D (shoelace formula).

    Only meaningful for loops confined to D=2. Sign encodes orientation
    (positive = counterclockwise).
    """
    x, y = 0, 0
    path = [(0, 0)]
    for step in word:
        if step == 1:
            x += 1
        elif step == -1:
            x -= 1
        elif step == 2:
            y += 1
        elif step == -2:
            y -= 1
        else:
            raise ValueError(f"signed_area_2d: step {step} not in {{±1, ±2}}")
        path.append((x, y))
    # Shoelace
    area2 = 0
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        area2 += x1 * y2 - x2 * y1
    return area2 // 2


def abs_area_2d(word: tuple[int, ...]) -> int:
    """|Area| for a 2D lattice loop."""
    return abs(signed_area_2d(word))


# ═══════════════════════════════════════════════════════════
# Lattice symmetry (hyperoctahedral group B_D)
# ═══════════════════════════════════════════════════════════


def _apply_symmetry(
    word: tuple[int, ...], perm: tuple[int, ...], signs: tuple[int, ...]
) -> tuple[int, ...]:
    """Apply direction permutation + sign flips to a word.

    perm[i] = j means direction i+1 → direction j+1.
    signs[i] = ±1 applies to the image of direction i+1.
    """
    out = []
    for step in word:
        idx = abs(step) - 1
        new_dir = perm[idx] + 1
        new_sign = signs[idx] if step > 0 else -signs[idx]
        out.append(new_sign * new_dir)
    return tuple(out)


def lattice_symmetry_orbit(word: tuple[int, ...], D: int) -> set[tuple[int, ...]]:
    """Orbit of `word` under the hyperoctahedral group B_D (permutations + sign flips),
    modulo cyclic rotation and backtrack reduction.
    """
    from itertools import permutations

    word = cyclic_canonical(reduce_backtracks(word))
    orbit: set[tuple[int, ...]] = set()
    for perm in permutations(range(D)):
        for signs in product((1, -1), repeat=D):
            w = _apply_symmetry(word, perm, signs)
            orbit.add(cyclic_canonical(reduce_backtracks(w)))
    return orbit


# ═══════════════════════════════════════════════════════════
# Loop enumeration
# ═══════════════════════════════════════════════════════════


def enumerate_closed_loops(D: int, max_length: int) -> list[tuple[int, ...]]:
    """Enumerate all distinct reduced closed loops in D dimensions up to given length.

    Returns a list of canonical representatives (backtrack-reduced, cyclic-canonical).
    The empty loop () is NOT included.
    """
    directions = list(range(-D, 0)) + list(range(1, D + 1))
    seen: set[tuple[int, ...]] = set()
    loops: list[tuple[int, ...]] = []
    for length in range(2, max_length + 1, 2):  # closed loops have even length on a square lattice
        for candidate in product(directions, repeat=length):
            if not is_closed(candidate, D):
                continue
            reduced = reduce_backtracks(candidate)
            if not reduced:
                continue
            canon = cyclic_canonical(reduced)
            if canon in seen:
                continue
            seen.add(canon)
            loops.append(canon)
    return loops


def enumerate_nonself_intersecting_2d(max_length: int) -> list[tuple[int, ...]]:
    """Subset of 2D closed loops that are non-self-intersecting (visit each site at
    most once, except for the start=end)."""

    def path_sites(steps):
        x, y = 0, 0
        sites = [(0, 0)]
        for s in steps:
            if s == 1:
                x += 1
            elif s == -1:
                x -= 1
            elif s == 2:
                y += 1
            elif s == -2:
                y -= 1
            sites.append((x, y))
        return sites

    out = []
    for w in enumerate_closed_loops(2, max_length):
        sites = path_sites(w)
        interior = sites[:-1]  # start = end, don't count the end
        if len(set(interior)) == len(interior):
            out.append(w)
    return out


# ═══════════════════════════════════════════════════════════
# Plaquette insertions and self-intersection splits for MM equation
# ═══════════════════════════════════════════════════════════


def plaquette_loops(D: int) -> list[tuple[int, ...]]:
    """All elementary plaquette loops in D dimensions (both orientations, all 2-planes)."""
    loops = []
    for i in range(1, D + 1):
        for j in range(i + 1, D + 1):
            # Counterclockwise in (i, j) plane
            loops.append((i, j, -i, -j))
            loops.append((-i, j, i, -j))  # other orientation starting differently
            loops.append((i, -j, -i, j))
            loops.append((-i, -j, i, j))
    return loops


def plaquette_insertions(word: tuple[int, ...], e_idx: int, D: int) -> list[tuple[int, ...]]:
    """Replace edge e_j = ±μ at position e_idx with a 3-edge "staple" around a plaquette.

    STAPLE CONVENTION (Kazakov-Zheng 2203.11360, Anderson-Kruczenski 2017):

    For edge e_j = +μ at position e_idx, and a plaquette P in the (μ, ν) plane:
      replace (+μ) → (+ν, +μ, -ν) for ν ∈ {+ν₀} (ν₀ running over perpendicular dirs)
      replace (+μ) → (-ν, +μ, +ν) for the opposite orientation of the plaquette

    Both staples have NET DISPLACEMENT +μ (same as the replaced edge), so the
    new loop is still closed. They differ by which side of edge e_j the plaquette
    is attached.

    For edge e_j = -μ, the staples are:
      replace (-μ) → (+ν, -μ, -ν) and (-ν, -μ, +ν)

    Returns the 2(D-1) resulting loops (in canonical form, after backtrack reduction).
    Each is the loop with edge e_j detoured around one plaquette containing it.
    """
    if e_idx < 0 or e_idx >= len(word):
        raise IndexError(f"e_idx {e_idx} out of range for word of length {len(word)}")
    mu = word[e_idx]  # signed step; |mu| is direction, sign(mu) is orientation
    inserted: list[tuple[int, ...]] = []
    for nu_dir in range(1, D + 1):
        if nu_dir == abs(mu):
            continue
        for nu_sign in (1, -1):
            nu = nu_sign * nu_dir
            # Staple: replace (mu) with (nu, mu, -nu)
            # Net displacement: nu + mu + (-nu) = mu ✓
            staple = (nu, mu, -nu)
            new_word = word[:e_idx] + staple + word[e_idx + 1 :]
            reduced = reduce_backtracks(new_word)
            if reduced:
                inserted.append(cyclic_canonical(reduced))
            else:
                inserted.append(())  # empty loop has W=1
    return inserted


def self_intersection_splits(
    word: tuple[int, ...],
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Find all self-intersection splits of a 2D loop.

    At a self-intersection (a site visited twice), the loop splits into two
    sub-loops. Returns a list of (C1, C2) pairs, each reduced and cyclically canonical.
    """
    # Compute the path
    x, y = 0, 0
    sites = [(0, 0)]
    for s in word:
        if s == 1:
            x += 1
        elif s == -1:
            x -= 1
        elif s == 2:
            y += 1
        elif s == -2:
            y -= 1
        else:
            raise ValueError(f"self_intersection_splits: unsupported step {s}")
        sites.append((x, y))

    # Find pairs of visit indices to the same site (excluding start==end)
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    site_visits: dict[tuple[int, int], list[int]] = {}
    for idx, site in enumerate(sites[:-1]):  # skip final wraparound
        site_visits.setdefault(site, []).append(idx)

    for _site, visits in site_visits.items():
        if len(visits) < 2:
            continue
        # For each pair of visits i < j, split word[i:j] (C1) and word[j:] + word[:i] (C2).
        for a in range(len(visits)):
            for b in range(a + 1, len(visits)):
                i, j = visits[a], visits[b]
                c1 = tuple(word[i:j])
                c2 = tuple(word[j:]) + tuple(word[:i])
                c1_reduced = cyclic_canonical(reduce_backtracks(c1))
                c2_reduced = cyclic_canonical(reduce_backtracks(c2))
                splits.append((c1_reduced, c2_reduced))
    return splits


# ═══════════════════════════════════════════════════════════
# LoopSystem: precomputed loop enumeration + MM equation tables
# ═══════════════════════════════════════════════════════════


@dataclass(frozen=True)
class MMEquation:
    """Precomputed Makeenko-Migdal equation as index arrays.

    Associates the equation for loop[loop_idx] at edge edge_idx to:
      LHS = (1/λ) Σ_{k in lhs_loop_indices} w[k]
      RHS = rhs_self_coeff · w[loop_idx]
          + Σ_{(i,j) in rhs_split_pairs} w[i] · w[j]

    Candidate D (the working MM form from Phase 1 Step 1):
      rhs_self_coeff = 2.0
      lhs_loop_indices = staple-replaced-loop indices
      rhs_split_pairs = non-trivial self-intersection splits (usually empty for simple loops)
    """

    loop_idx: int
    edge_idx: int
    lhs_loop_indices: tuple[int, ...]
    rhs_self_coeff: float
    rhs_split_pairs: tuple[tuple[int, int], ...]


@dataclass
class LoopSystem:
    """Precomputed lattice loop enumeration + MM-equation tables.

    Build once per (D, L_max, form). The neural network trains against
    `mm_equations` using the precomputed index arrays.

    The `loops` list extends to L_max + 2 so that every MM equation's LHS
    staple-insertion is still in the table. MM equations are only generated
    for loops of length ≤ L_max (the loops of length L_max + 2 are "output
    only" — they feed into MM equations for shorter loops but have no
    MM constraints of their own).
    """

    D: int
    L_max: int
    loops: list[tuple[int, ...]]
    loop_to_idx: dict[tuple[int, ...], int] = field(default_factory=dict)
    mm_equations: list[MMEquation] = field(default_factory=list)
    areas: dict[int, int] | None = None

    @property
    def K(self) -> int:
        """Number of distinct loops in the table."""
        return len(self.loops)

    @property
    def empty_idx(self) -> int:
        """Index of the empty loop (W = 1 by normalization)."""
        return self.loop_to_idx[()]


def build_loop_system(D: int, L_max: int, mm_form: str = "D") -> LoopSystem:
    """Enumerate loops up to L_max + 2 and precompute MM equations for loops up to L_max.

    mm_form: key in the candidate catalog (default "D" = the working form from Phase 1 Step 1).
    """
    from mm_equations import _candidate_catalog  # avoid circular import at module load

    # Enumerate loops up to L_max + 2 (plus the empty loop at index 0)
    raw_loops = enumerate_closed_loops(D, L_max + 2)
    loops: list[tuple[int, ...]] = [()]  # empty loop reserved at index 0
    for w in raw_loops:
        loops.append(w)

    loop_to_idx: dict[tuple[int, ...], int] = {}
    for i, w in enumerate(loops):
        loop_to_idx[w] = i

    # For D=2, compute areas (absolute)
    areas: dict[int, int] | None = None
    if D == 2:
        areas = {}
        for i, w in enumerate(loops):
            if not w:
                areas[i] = 0
            else:
                try:
                    areas[i] = abs_area_2d(w)
                except ValueError:
                    areas[i] = 0

    # Build MM equations for loops up to length L_max
    # Currently only candidate D is implemented as an equation (the scan candidates
    # live in mm_equations.py as residual functions; here we materialize the index tables).
    assert mm_form == "D", f"Only MM form 'D' is supported so far, got {mm_form!r}"

    mm_eqs: list[MMEquation] = []
    for loop_idx, word in enumerate(loops):
        if not word or len(word) > L_max:
            continue
        for edge_idx in range(len(word)):
            # LHS: staple insertions
            insertions = plaquette_insertions(word, edge_idx, D)
            lhs_indices: list[int] = []
            for ins in insertions:
                canon = cyclic_canonical(reduce_backtracks(ins))
                if canon in loop_to_idx:
                    lhs_indices.append(loop_to_idx[canon])
                # else: truncation boundary — drop the contribution
            # RHS: self-coefficient + splits
            rhs_coeff = 2.0  # candidate D
            splits = self_intersection_splits(word) if D == 2 else []
            rhs_split_pairs: list[tuple[int, int]] = []
            for c1, c2 in splits:
                if c1 in loop_to_idx and c2 in loop_to_idx:
                    rhs_split_pairs.append((loop_to_idx[c1], loop_to_idx[c2]))

            mm_eqs.append(
                MMEquation(
                    loop_idx=loop_idx,
                    edge_idx=edge_idx,
                    lhs_loop_indices=tuple(lhs_indices),
                    rhs_self_coeff=rhs_coeff,
                    rhs_split_pairs=tuple(rhs_split_pairs),
                )
            )

    # Suppress unused import warning from IDE: catalog may be used in future mm_forms
    _ = _candidate_catalog

    return LoopSystem(
        D=D,
        L_max=L_max,
        loops=loops,
        loop_to_idx=loop_to_idx,
        mm_equations=mm_eqs,
        areas=areas,
    )
