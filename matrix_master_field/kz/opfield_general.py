"""Vehicle B construction, finished: general-degree operator master field for the
Kazakov-Zheng two-matrix model, with the COMPLETE self-adjoint operator basis and a
reach-vs-ansatz-richness characterization.

Motivation. The decisive experiment (kz_large_observables.py) found the operator field's
reliable large-observable reach is capped by ANSATZ DEGREE, and the previous ansatz
(opfield_kz_cont.sa_monomials) was hardcoded to degree 3/5 AND incomplete (only 10 of the 14
self-adjoint operators at degree 5). This module supplies:
  1. sa_basis(u,v,degree): the COMPLETE self-adjoint operator basis of the A-sector -- every
     monomial in the two free generators with ODD #u, EVEN #v (the KZ Z2xZ2 selection),
     symmetrized (M+M^dag)/2, one per reversal class, up to `degree`. Counts: deg 3/5/7/9 ->
     4/14/50/186. B-sector = swap (u,v). (Verified: contains the old basis; self-adjoint to 1e-13.)
  2. a general-degree operator-field fit (continuation h:0->1) reusing the verified KZ planar
     loop residual of opfield_kz_cont, but over the complete basis at arbitrary degree.
  3. reach_study(): does the reliable reach (largest word length whose m[A^{2k}] lands INSIDE
     the rigorous bootstrap bracket) grow as the ansatz richens? -> reach(degree).

Coefficients c are Fock-independent: fit at small Fock-L (dense monomial stack feasible), evaluate
observables at larger Fock-L via a memory-safe incremental build. High degree needs high Fock-L to
resolve V'(A)*w during the FIT (products of length ~degree+W), which is the dense wall that the
matrix-free/sparse-Fock lift (cuntz_bootstrap/matfree_expm.py) is built to remove -- the clearly
motivated next construction step once this characterizes the dense reach.

Run: uv run --no-project --with numpy --with scipy --with jax --with cvxpy \
         python matrix_master_field/kz/opfield_general.py
"""
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from matrix_master_field.cuntz_fock import CuntzFockSpace  # noqa: E402
from matrix_master_field.bootstrap_sdp import bootstrap_two_matrix_kz  # noqa: E402

ISLAND_A2 = (0.4204, 0.4224)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".fitcache")


def _basis_keys(degree):
    """Reversal-class representatives: words in {0,1}, odd #0, even #1, odd length <= degree."""
    keys, seen = [], set()
    for L in range(1, degree + 1, 2):
        for w in itertools.product((0, 1), repeat=L):
            if w.count(0) % 2 == 1 and w.count(1) % 2 == 0:
                k = min(w, w[::-1])
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
    return keys


def _word_matrix(key, u, v):
    M = (u if key[0] == 0 else v)
    for c in key[1:]:
        M = M @ (u if c == 0 else v)
    return M


def sa_basis(u, v, degree):
    """Complete self-adjoint A-sector operator basis (list of dense matrices)."""
    out = []
    for key in _basis_keys(degree):
        fwd = _word_matrix(key, u, v)
        out.append(fwd if key == key[::-1] else 0.5 * (fwd + _word_matrix(key[::-1], u, v)))
    return out


def _make_loss(fock, g, W, degree):
    x0 = fock.x(0).astype(np.float64)
    x1 = fock.x(1).astype(np.float64)
    MA = jnp.asarray(np.stack(sa_basis(x0, x1, degree)))
    MB = jnp.asarray(np.stack(sa_basis(x1, x0, degree)))
    nP = MA.shape[0]
    vac = jnp.asarray(fock.vacuum_state())
    words = [()]
    for L in range(1, W + 1):
        words += [tuple(c) for c in itertools.product((0, 1), repeat=L)]

    def build(c, h):
        A = jnp.tensordot(c, MA, axes=(0, 0))
        B = jnp.tensordot(c, MB, axes=(0, 0))
        A2, B2 = A @ A, B @ B
        VpA = A + g * (A2 @ A) + h * (A @ B2 + B2 @ A - 2.0 * (B @ A @ B))
        VpB = B + g * (B2 @ B) + h * (B @ A2 + A2 @ B - 2.0 * (A @ B @ A))
        return A, B, VpA, VpB

    def tau(ops):
        v = vac
        for op in reversed(ops):
            v = op @ v
        return v[0]

    def loss(c, h):
        A, B, VpA, VpB = build(c, h)
        opm = (A, B)
        s = 0.0
        for w in words:
            wo = [opm[k] for k in w]
            for a, Vp in ((0, VpA), (1, VpB)):
                r = tau([Vp] + wo)
                for k in range(len(w)):
                    if w[k] == a:
                        r = r - tau([opm[j] for j in w[:k]]) * tau([opm[j] for j in w[k + 1:]])
                s = s + r * r
        return s

    def mA2(c):
        A, _, _, _ = build(c, 0.0)
        return (A @ (A @ vac))[0]

    return jax.jit(jax.value_and_grad(loss)), jax.jit(loss), jax.jit(mA2), nP


def continuation(fock, g, h_target, W, degree, n_steps=10, maxiter=1500, verbose=False):
    vg, L, M, nP = _make_loss(fock, g, W, degree)
    c = np.zeros(nP)
    c[0] = 1.0
    for h in np.linspace(0.0, h_target, n_steps + 1):
        def fg(x):
            v, gr = vg(jnp.asarray(x), h)
            return float(v), np.asarray(gr, float)
        c = minimize(fg, c, jac=True, method="L-BFGS-B",
                     options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-13}).x
        if verbose:
            print(f"    h={h:.3f} loss={float(L(jnp.asarray(c), h)):.2e} "
                  f"m[A^2]={float(M(jnp.asarray(c))):.5f}", flush=True)
    return c, float(L(jnp.asarray(c), h_target)), float(M(jnp.asarray(c)))


def fit(g, h, W, degree, fit_L, n_steps=10, maxiter=1500):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"gen_g{g}_h{h}_W{W}_d{degree}_L{fit_L}.npy")
    if os.path.exists(path):
        return np.load(path), True, 0.0
    fock = CuntzFockSpace(2, fit_L)
    t = time.time()
    c, loss, mA2 = continuation(fock, g, h, W, degree, n_steps=n_steps, maxiter=maxiter)
    np.save(path, c)
    return c, False, time.time() - t


# ---- memory-safe evaluation at large Fock-L (no monomial stack) ----
def _sa_gen(u, v, degree):
    for key in _basis_keys(degree):
        fwd = _word_matrix(key, u, v)
        yield fwd if key == key[::-1] else 0.5 * (fwd + _word_matrix(key[::-1], u, v))


def A_operator(fock, c, degree):
    x0 = fock.x(0).astype(np.float64)
    x1 = fock.x(1).astype(np.float64)
    A = None
    for ck, m in zip(c, _sa_gen(x0, x1, degree)):
        A = ck * m if A is None else A + ck * m
    return A, fock.vacuum_state()


def A_moments(A, vac, ks):
    out, v = {}, vac
    for k in range(1, max(ks) + 1):
        v = A @ (A @ v)
        if k in ks:
            out[k] = float(np.real(v[0]))
    return out


def reach_study(g=1.0, h=1.0, configs=((5, 3, 8), (5, 5, 10), (7, 3, 10)),
                Ks=(1, 2, 3, 4, 5), eval_L=12, out_csv=None):
    """For each (degree, W, fit_L): fit the complete-basis operator field, evaluate m[A^{2k}],
    and report the reliable reach = largest word length whose value is INSIDE the rigorous
    bootstrap bracket. Sweeping W (the loop-equation cutoff) tests the true reach knob; sweeping
    degree tests basis richness."""
    print("=" * 86)
    print("VEHICLE B (finished) — reliable reach vs loop-cutoff W and basis degree (KZ g=h=1)")
    print("  reach = largest word length whose m[A^{2k}] is INSIDE the rigorous bootstrap bracket")
    print("=" * 86)
    bracket = {}
    for k in Ks:
        tw = (0,) * (2 * k)
        for cutoff in (10, 8, 6, 4):
            if 2 * k > cutoff:
                continue
            lo = bootstrap_two_matrix_kz(g, h, max_word_len=cutoff, target_word=tw, maximize=False)
            hi = bootstrap_two_matrix_kz(g, h, max_word_len=cutoff, target_word=tw, maximize=True)
            if lo is not None and hi is not None:
                bracket[k] = (lo, hi, cutoff)
                break
    print("  rigorous brackets:  " + "   ".join(
        f"A^{2*k}[{bracket[k][0]:.3f},{bracket[k][1]:.3f}]@{bracket[k][2]}" for k in Ks if k in bracket))
    rows = []
    for (deg, W, fit_L) in configs:
        c, cached, sec = fit(g, h, W, deg, fit_L)
        fock = CuntzFockSpace(2, eval_L)
        A, vac = A_operator(fock, c, deg)
        vals = A_moments(A, vac, list(Ks))
        nP = len(_basis_keys(deg))
        reach, line = 0, []
        for k in Ks:
            v = vals[k]
            if k in bracket:
                lo, hi, _ = bracket[k]
                inside = lo - 2e-3 <= v <= hi + 2e-3
                if inside and 2 * k > reach:
                    reach = 2 * k
                line.append(f"A^{2*k}:{v:.4f}{'in ' if inside else 'OUT'}")
            else:
                line.append(f"A^{2*k}:{v:.4f}(?)")
        tag = "cached" if cached else f"{sec:.0f}s"
        print(f"\n  deg={deg} W={W} (|basis|={nP}, fit L={fit_L}, {tag})  m[A^2]={vals[1]:.5f}")
        print("    " + "  ".join(line))
        print(f"    => reliable reach: word length {reach}")
        rows.append({"degree": deg, "W": W, "n_basis": nP, "m_A2": vals[1], "reach_len": reach,
                     **{f"A{2*k}": vals[k] for k in Ks}})
    print("\n  Reading: if reach_len grows with W (at fixed sufficient degree), the loop-equation")
    print("  cutoff is the reach knob (operator analogue of the bootstrap cutoff) and the")
    print("  construction reaches large observables with enough W. If reach saturates even as W")
    print("  grows, the fit is under-determined for long words (zero residual doesn't pin them).")
    if out_csv and rows:
        import csv
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")
    return rows


if __name__ == "__main__":
    repo = str(_RepoP(__file__).resolve().parents[2])
    reach_study(out_csv=os.path.join(repo, "results", "kz_reach_vs_degree.csv"))
