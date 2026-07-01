"""Exact large-N lattice loop (Makeenko-Migdal) equation, dimension-general.

This module supersedes the "candidate D" form baked into
``cluster/lattice.py::build_loop_system``, which is only the LEADING
strong-coupling term: on the exact D=2 area-law master field
(W[C] = w_+^{Area}, w_+ = 1/(2 lam) for lam >= 1) candidate D leaves a
per-edge residual of exactly ``(1/lam) * w_+^{Area+1}`` -- it drops the
quadratic *contact* term. See ``docs/superpowers/results/2026-07-01-lattice-loop-equation.md``.

The exact equation, for a Wilson loop C written with base link mu = C[0]
and remainder Chat = C[1:], at large N and in the strong-coupling phase
(lam >= 1), is

    (1/lam)  Sum_nu  W(B_nu)              # link-detour staples
  -  2       W(C)                          # self / "trivial" term
  - (1/2lam) Sum_nu  W(A_nu)              # plaquette-prepended CONTACT term
  -          Sum_splits  sigma_s W(C1) W(C2)   # self-intersection splits of C
  =  0

where nu runs over the 2(D-1) signed lattice directions perpendicular to
|mu|, and

    B_nu = (nu, mu, -nu) + Chat            # detour link mu around a plaquette
    A_nu = (mu, nu, -mu, -nu) + C          # prepend a plaquette at the base x

At N = infinity A_nu is a figure-eight touching C only at the base point x,
so it FACTORIZES: W(A_nu) = W(plaquette_{mu,nu}) * W(C). That factorization
is the contact term candidate D omitted.

Equivalence check: for the unit plaquette C = (1,2,-1,-2) at D=2 this reduces
to the exact plaquette equation used by the *working* QCD_2 unsupervised
solver (``cuntz_bootstrap/qcd2_q2.py::plaquette_mm_residual``):

    (1/lam)[W_empty + W_{1x2}] - 2 W_plaq - (1/lam) W_plaq^2 = 0 .

The structure is DIMENSION-GENERAL: only the perpendicular set {nu} grows
with D. At D=3 each in-plane plaquette edge additionally belongs to two
perpendicular plaquettes (the nu = +-3 staples), which inject the
out-of-plane loops responsible for the O(1/lam^4) corrections to the naive
plane-by-plane area law -- i.e. the genuine D=3 dynamics.

Run ``python -m cuntz_bootstrap.lattice_loop_eq`` (or execute this file) for
the self-contained machine-zero verification at D=2.
"""
from __future__ import annotations

# --- minimal, self-contained lattice geometry (no fragile cross-package deps) ---


def reduce_backtracks(word: tuple[int, ...]) -> tuple[int, ...]:
    """Remove adjacent (mu, -mu) pairs, including the cyclic wrap."""
    w = list(word)
    changed = True
    while changed and w:
        changed = False
        i = 0
        while i < len(w) - 1:
            if w[i] == -w[i + 1]:
                del w[i:i + 2]
                changed = True
                if i > 0:
                    i -= 1
            else:
                i += 1
        if len(w) >= 2 and w[0] == -w[-1]:
            w = w[1:-1]
            changed = True
    return tuple(w)


def cyclic_canonical(word: tuple[int, ...]) -> tuple[int, ...]:
    """Lexicographically smallest cyclic rotation."""
    if not word:
        return word
    n = len(word)
    return min(word[i:] + word[:i] for i in range(n))


def perpendicular_dirs(mu: int, D: int) -> list[int]:
    """Signed lattice directions perpendicular to |mu| (2(D-1) of them)."""
    out = []
    for d in range(1, D + 1):
        if d == abs(mu):
            continue
        out.extend((d, -d))
    return out


def detour_staple(C: tuple[int, ...], nu: int) -> tuple[int, ...]:
    """B_nu = (nu, mu, -nu) + Chat  -- detour the base link mu around plaquette."""
    mu = C[0]
    return (nu, mu, -nu) + C[1:]


def prepended_plaquette(C: tuple[int, ...], nu: int) -> tuple[int, ...]:
    """A_nu = (mu, nu, -mu, -nu) + C  -- prepend a plaquette at the base point."""
    mu = C[0]
    return (mu, nu, -mu, -nu) + C


def plaquette(mu: int, nu: int) -> tuple[int, ...]:
    return (mu, nu, -mu, -nu)


def loop_equation_terms(C: tuple[int, ...], D: int):
    """Return the symbolic term lists of the exact loop equation at base edge 0.

    Returns a dict with:
      'detours'   : list of B_nu words          (coefficient +1/lam each)
      'contacts'  : list of A_nu words          (coefficient -1/2lam each)
      'self'      : the word C                   (coefficient -2)
    Self-intersection split terms of C are NOT included here (handle via the
    factorization identities, as the D=2 Q2 solver does).
    """
    nus = perpendicular_dirs(C[0], D)
    return {
        "detours": [detour_staple(C, nu) for nu in nus],
        "contacts": [prepended_plaquette(C, nu) for nu in nus],
        "self": C,
    }


# --------------------------------------------------------------------------
# Self-contained verification against the D=2 factorized area-law master field
# --------------------------------------------------------------------------


def _signed_area_2d(word: tuple[int, ...]) -> int:
    x = y = 0
    path = [(0, 0)]
    for s in word:
        x += (s == 1) - (s == -1)
        y += (s == 2) - (s == -2)
        path.append((x, y))
    a2 = sum(path[i][0] * path[i + 1][1] - path[i + 1][0] * path[i][1]
             for i in range(len(path) - 1))
    return a2 // 2


def _sites(word):
    x = y = 0
    s = [(0, 0)]
    for st in word:
        x += (st == 1) - (st == -1)
        y += (st == 2) - (st == -2)
        s.append((x, y))
    return s


def area_law_W(word: tuple[int, ...], w_plus: float) -> float:
    """N=inf 2D master field: factorize at self-touch points, simple -> w_+^|area|.

    Valid ONLY in D=2 and the strong-coupling phase (where the area law is exact).
    Used purely as the verification oracle.
    """
    w = cyclic_canonical(reduce_backtracks(word))
    if not w:
        return 1.0
    pts = _sites(w)[:-1]
    seen: dict = {}
    for idx, p in enumerate(pts):
        if p in seen:
            i, j = seen[p], idx
            return area_law_W(w[i:j], w_plus) * area_law_W(w[j:] + w[:i], w_plus)
        seen[p] = idx
    return w_plus ** abs(_signed_area_2d(w))


def residual_d2(C: tuple[int, ...], lam: float, W=area_law_W) -> float:
    """Exact loop-equation residual at base edge 0, evaluated with oracle W."""
    wp = 1.0 / (2.0 * lam)
    t = loop_equation_terms(C, D=2)
    return (
        (1.0 / lam) * sum(W(b, wp) for b in t["detours"])
        - 2.0 * W(t["self"], wp)
        - (1.0 / (2.0 * lam)) * sum(W(a, wp) for a in t["contacts"])
    )


def _verify_d2(max_len: int = 12, lams=(1.0, 2.0, 5.0, 10.0)) -> float:
    """Enumerate simple D=2 loops, apply the equation at every edge, report max|res|."""
    from itertools import product

    # enumerate reduced closed non-self-intersecting 2D loops up to max_len
    dirs = [-2, -1, 1, 2]
    seen: set = set()
    simple: list = []
    for L in range(4, max_len + 1, 2):
        for cand in product(dirs, repeat=L):
            if sum((s == 1) - (s == -1) for s in cand) != 0:
                continue
            if sum((s == 2) - (s == -2) for s in cand) != 0:
                continue
            r = reduce_backtracks(cand)
            if not r:
                continue
            canon = cyclic_canonical(r)
            if canon in seen:
                continue
            seen.add(canon)
            pts = _sites(canon)[:-1]
            if len(set(pts)) == len(pts):  # non-self-intersecting
                simple.append(canon)

    worst = 0.0
    for lam in lams:
        mx = 0.0
        for C0 in simple:
            for i in range(len(C0)):
                C = C0[i:] + C0[:i]
                mx = max(mx, abs(residual_d2(C, lam)))
        worst = max(worst, mx)
        print(f"  lam={lam:5.2f}: {len(simple)} simple loops, "
              f"max|residual| = {mx:.3e}")
    return worst


if __name__ == "__main__":
    print("Exact lattice loop equation -- D=2 machine-zero verification")
    print("(strong-coupling phase lam >= 1, area-law master field)")
    worst = _verify_d2()
    tol = 1e-12
    print(f"\nworst over all lam = {worst:.3e}  "
          f"[{'PASS' if worst < tol else 'FAIL'}] (tol {tol})")
