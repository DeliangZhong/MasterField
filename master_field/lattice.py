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
    """Insert each plaquette containing link e at position e_idx in `word`.

    The MM equation: ∑_{P ∋ e} W[P_e ∘ C]. For link e = word[e_idx] in direction μ,
    the plaquettes containing e are the 2(D-1) loops that start with step μ.

    P_e ∘ C means: replace the single step μ at position e_idx with the 4 steps
    of the plaquette (which net to μ).
    """
    if e_idx < 0 or e_idx >= len(word):
        raise IndexError(f"e_idx {e_idx} out of range for word of length {len(word)}")
    mu = word[e_idx]
    inserted: list[tuple[int, ...]] = []
    # A plaquette containing link μ in direction |μ| is: (μ, ν, -μ, -ν) with ν ≠ ±|μ|.
    # Its net displacement is μ (the first step).
    # Sub-sequence replaces 1 step by 4.
    for nu in range(1, D + 1):
        if nu == abs(mu):
            continue
        for sign in (1, -1):
            plaq = (mu, sign * nu, -mu, -sign * nu)
            # Replacing step μ at e_idx with the plaquette preserves net displacement.
            new_word = word[:e_idx] + plaq + word[e_idx + 1 :]
            reduced = reduce_backtracks(new_word)
            if reduced:
                inserted.append(cyclic_canonical(reduced))
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
