"""Vehicle B decisive experiment: LARGE-observable reach of the operator master field
vs the convex bootstrap, on the Kazakov-Zheng two-matrix model (arXiv:2108.04830 eq.6),

    S = N*tr[ 1/2 (A^2+B^2) + (g/4)(A^4+B^4) - (h/2)[A,B]^2 ],   g=h=1.

THESIS UNDER TEST. The operator field is a *construction* (a genuine tracial state on the
free Cuntz-Fock space): once the coefficients are fitted to the planar loop equations, ONE
configuration yields tau(W) for ANY word W -- including long words -- at the cost of a few
matvecs. The convex bootstrap can only bound a moment m[W] if its cutoff max_word_len >= len(W),
and the SDP (product matrix G is nvar x nvar, nvar ~ #canonical words up to the cutoff) blows up
exponentially in the cutoff -- so there is a word length beyond which it gives NO bound.

Only meaningful if the operator large-observable values are TRUSTWORTHY, so it is gated on:
  (A) ANSATZ convergence: m[A^{2k}] stabilises as the polynomial degree / loop-cutoff W grows;
  (B) FOCK convergence: it stabilises as the Fock-space cutoff L grows (A raises the Fock level,
      so m[A^{2k}] needs L >~ 2k -- the dense reach wall);
  (C) AGREEMENT: where the bootstrap bracket is tight, the operator value lies inside it.
The k where (A)/(B) break is the dense operator-field reach; pushing it is what the matrix-free
lift (cuntz_bootstrap/matfree_expm.py, sparse Fock) buys. Coefficients c are Fock-INDEPENDENT, so
we fit cheap (small L) and evaluate at large L.

Ground-truth anchor: m[A^2] in the level-12 convex island [0.4204, 0.4224] at g=h=1.

Run: uv run --no-project --with numpy --with scipy --with jax --with cvxpy \
         python matrix_master_field/kz/kz_large_observables.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))

from opfield_kz_cont import continuation  # noqa: E402  (fit the operator field)
from matrix_master_field.cuntz_fock import CuntzFockSpace  # noqa: E402
from matrix_master_field.bootstrap_sdp import bootstrap_two_matrix_kz  # noqa: E402

ISLAND_A2 = (0.4204, 0.4224)  # level-12 convex island for m[A^2] at g=h=1 (ground-truth anchor)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fitcache")


def _monomials(u, v, degree):
    """Yield the self-adjoint monomials of `opfield_kz_cont.sa_monomials(u,v,degree)` ONE AT A
    TIME (odd in u, even in v), so build_ops never holds the full stack (memory-safe at large L)."""
    def prod(seq):
        P = seq[0]
        for m in seq[1:]:
            P = P @ m
        return P

    def sym(seq):
        return 0.5 * (prod(seq) + prod(seq[::-1]))

    yield u
    if degree >= 3:
        yield prod([u, u, u]); yield sym([u, v, v]); yield prod([v, u, v])
    if degree >= 5:
        yield prod([u, u, u, u, u]); yield sym([u, u, u, v, v]); yield sym([u, v, v, v, v])
        yield prod([v, v, u, v, v]); yield sym([u, u, v, u, v]); yield prod([v, u, u, u, v])


def build_ops(fock, c, degree):
    """A, B self-adjoint operators (dense) from fitted coefficients c, accumulated incrementally."""
    x0 = fock.x(0).astype(np.float64)
    x1 = fock.x(1).astype(np.float64)
    A = B = None
    for ck, ma in zip(c, _monomials(x0, x1, degree)):
        A = ck * ma if A is None else A + ck * ma
    for ck, mb in zip(c, _monomials(x1, x0, degree)):
        B = ck * mb if B is None else B + ck * mb
    return A, B, fock.vacuum_state()


def A_power_moments(A, vac, ks):
    """m[A^{2k}] for k in ks, reusing the running vector (A^2)^k |Omega>."""
    out = {}
    v = vac
    for k in range(1, max(ks) + 1):
        v = A @ (A @ v)          # multiply by A^2
        if k in ks:
            out[k] = float(np.real(v[0]))
    return out


def fit(g, h, W, degree, fock_L, n_steps=8, maxiter=1200):
    """Fit (continuation h:0->1) with on-disk caching of the Fock-independent coefficients c."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"g{g}_h{h}_W{W}_d{degree}_L{fock_L}.npy")
    if os.path.exists(path):
        c = np.load(path)
        return {"c": c, "W": W, "deg": degree, "fit_L": fock_L, "sec": 0.0, "cached": True}
    fock = CuntzFockSpace(2, fock_L)
    t = time.time()
    _, c = continuation(fock, g, h, W, degree, n_steps=n_steps, maxiter=maxiter)
    c = np.asarray(c)
    np.save(path, c)
    return {"c": c, "W": W, "deg": degree, "fit_L": fock_L, "sec": time.time() - t, "cached": False}


def bootstrap_bracket(g, h, target_word, cutoff):
    """[lo, hi] for m[target_word] at moment cutoff (None if unreachable/failed)."""
    if len(target_word) > cutoff:
        return None  # the SDP literally cannot represent this moment at this cutoff
    lo = bootstrap_two_matrix_kz(g, h, max_word_len=cutoff, target_word=target_word, maximize=False)
    hi = bootstrap_two_matrix_kz(g, h, max_word_len=cutoff, target_word=target_word, maximize=True)
    return None if (lo is None or hi is None) else (lo, hi)


def run(g=1.0, h=1.0, Ks=(1, 2, 3, 4, 5, 6, 7, 8), out_csv=None):
    print("=" * 84)
    print("VEHICLE B — LARGE-OBSERVABLE REACH: operator master field vs bootstrap (KZ, g=h=1)")
    print(f"family m[A^(2k)], k in {list(Ks)}  (word lengths {[2 * k for k in Ks]})   "
          f"anchor: m[A^2] island {ISLAND_A2}")
    print("=" * 84)

    # ---- fit two ansatz richnesses (deg 3 vs 5); c is Fock-independent so fit cheap ----
    print("\n[A] operator-field fits (cached; c is Fock-independent):")
    fits = [fit(g, h, 3, 3, 8), fit(g, h, 3, 5, 9)]
    for f in fits:
        tag = "cached" if f["cached"] else f"{f['sec']:.0f}s"
        print(f"  W={f['W']} deg={f['deg']} (fit L={f['fit_L']}, {tag})")

    # ---- (A) ANSATZ convergence: evaluate every fit at a COMMON Fock-L=10 ----
    Lc = 10
    print(f"\n[A] ansatz convergence — m[A^(2k)] at common Fock-L={Lc} (deg 3 vs deg 5):")
    fock = CuntzFockSpace(2, Lc)
    acols = {}
    for f in fits:
        A, _, vac = build_ops(fock, f["c"], f["deg"])
        acols[(f["W"], f["deg"])] = A_power_moments(A, vac, list(Ks))
    print("  " + "obs".rjust(6) + "".join(f"W{W}d{deg}".rjust(13) for (W, deg) in acols))
    for k in Ks:
        print("  " + f"A^{2 * k}".rjust(6)
              + "".join(f"{acols[key][k]:.6f}".rjust(13) for key in acols))

    # ---- (B) FOCK convergence of the deg-5 fit: L = 8, 10, 12 ----
    best = fits[-1]
    Ls = (8, 10, 12)
    print(f"\n[B] Fock convergence of deg-{best['deg']} fit — m[A^(2k)] at L={Ls}:")
    conv = {}
    for L in Ls:
        fk = CuntzFockSpace(2, L)
        A, _, vac = build_ops(fk, best["c"], best["deg"])
        conv[L] = A_power_moments(A, vac, list(Ks))
    print("  " + "obs".rjust(6) + "".join(f"L={L}".rjust(12) for L in Ls) + "|Δ(10,12)|".rjust(13))
    reach = None
    for k in Ks:
        d = abs(conv[12][k] - conv[10][k])
        flag = ""
        if d > 5e-3 and reach is None:
            reach = 2 * (k - 1)
            flag = "  <- Fock wall"
        print("  " + f"A^{2 * k}".rjust(6) + "".join(f"{conv[L][k]:.6f}".rjust(12) for L in Ls)
              + f"{d:.2e}".rjust(13) + flag)
    reach_txt = f"word length ~{reach}" if reach else f">= {2 * max(Ks)}"
    print(f"  => dense operator-field Fock-converged reach (|Δ(10,12)|<5e-3): {reach_txt}")

    # ---- (C) bootstrap brackets at increasing cutoff vs the converged operator value ----
    print("\n[C] bootstrap brackets for m[A^(2k)] vs operator value (converged, L=12):")
    print("  '—' = the SDP cannot represent this moment at that cutoff (len > cutoff)")
    cutoffs = [4, 6, 8, 10]
    op = conv[12]
    print("  " + "obs".rjust(5) + "op".rjust(11) + "".join(f"boot@{c}".rjust(19) for c in cutoffs))
    rows = []
    for k in Ks:
        tw = (0,) * (2 * k)
        line = "  " + f"A^{2 * k}".rjust(5) + f"{op[k]:.5f}".rjust(11)
        rec = {"twok": 2 * k, "op_value": op[k]}
        for c in cutoffs:
            br = bootstrap_bracket(g, h, tw, c)
            if br is None:
                line += "—".rjust(19)
                rec[f"boot{c}_lo"] = rec[f"boot{c}_hi"] = ""
            else:
                inside = br[0] - 2e-3 <= op[k] <= br[1] + 2e-3
                line += f"[{br[0]:.3f},{br[1]:.3f}]{'*' if inside else '!'}".rjust(19)
                rec[f"boot{c}_lo"], rec[f"boot{c}_hi"] = br
        print(line)
        rows.append(rec)

    print("\n  legend: [lo,hi]* operator INSIDE bracket (validated);  ! outside;  — unreachable.")
    print("  Where the bracket is tight the operator value sits inside it (validation); for longer")
    print("  words the bracket widens / becomes unreachable while the operator field returns a")
    print("  single Fock-converged value from ONE configuration — until its own dense Fock wall,")
    print("  which the matrix-free/sparse lift is built to push.")

    if out_csv:
        import csv
        keys = ["twok", "op_value"] + [f"boot{c}_{s}" for c in cutoffs for s in ("lo", "hi")]
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")
    return rows


if __name__ == "__main__":
    repo = str(_RepoP(__file__).resolve().parents[2])
    run(out_csv=os.path.join(repo, "results", "kz_large_observables.csv"))
