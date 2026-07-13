"""HONEST 2D area-law gate for the operator (Cuntz-Fock) master field.

Question: does an enriched unitary-link master field REPRODUCE the exact 2D
lattice area law  W[R x T] = w_+^(R*T),  w_+ = 1/(2 lam) (lam>=1),  as a
PREDICTION -- i.e. WITHOUT the area law being fed in?

The existing qcd2_q2.py pass is circular: its `factorization_loss` hard-codes
W[2x1]=W[plaq]^2, W[2x2]=W[plaq]^4, W[3x1]=W[plaq]^3 (the area law itself).
Here we replace that with the EXACT contact-term-complete lattice loop equation
(cuntz_bootstrap/lattice_loop_eq.py, verified machine-zero) imposed on every
loop in a set, at every base edge:

    r(C) = (1/lam) Sum_nu W(B_nu)  -  2 W(C)  -  (1/2lam) Sum_nu W(plaq(mu,nu)) W(C)

  * detours B_nu = (nu,mu,-nu)+Chat  -> evaluated DIRECTLY on the master field
    (this is the dynamical content that connects area A to area A+-1),
  * contacts A_nu = (mu,nu,-mu,-nu)+C -> base-point factorization W(plaq)*W(C)
    (a GENERIC N=inf identity: two loops joined at one point factorize; NOT the
    area law).

For C = unit plaquette this reduces EXACTLY to
    (1/lam)[W_empty + W_1x2] - 2 W_plaq - (1/lam) W_plaq^2 = 0
(= the working plaquette_mm_residual). The area law is a held-out PREDICTION we
then CHECK against qcd2_exact.qcd2_wilson_loop.

Plus generic (non-area-law) regularizers: cyclicity (trace property),
reflection positivity (genuine state), lattice B_D symmetry.

Convention (pinned, lattice Wilson action, verified regime): SU(N) N=inf,
lam = 't Hooft coupling, w_+ = 1/(2 lam) for lam >= 1.

Run:
  uv run --no-project --with numpy --with scipy --with jax --with optax \
      python qcd2_area_gate.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = "/Users/deliangzhong/Documents/Working/Master Field"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cluster"))  # qcd2_exact.py does bare `import lattice`

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from cuntz_bootstrap.fock import CuntzFockJAX  # noqa: E402
from cuntz_bootstrap.hermitian_operator import init_hermitian_params  # noqa: E402
from cuntz_bootstrap.matfree_expm import (  # noqa: E402
    build_forward_link_ops_matfree,
    build_word_pairs,
)
from cuntz_bootstrap.wilson_loops import wilson_loop  # noqa: E402
from cuntz_bootstrap.optimize import optimize_cuntz  # noqa: E402
from cuntz_bootstrap.qcd2_exact import qcd2_wilson_loop  # noqa: E402
from cuntz_bootstrap.cyclicity import cyclicity_loss  # noqa: E402
from cuntz_bootstrap.reflection_positivity import (  # noqa: E402
    positive_half_open_paths,
    reflection_positivity_loss,
)
from cuntz_bootstrap.lattice_symmetry import (  # noqa: E402
    b_d_generators,
    lattice_symmetry_loss,
)
from cuntz_bootstrap.lattice_loop_eq import (  # noqa: E402
    perpendicular_dirs,
    detour_staple,
    plaquette,
)

D = 2
PLAQ = (1, 2, -1, -2)


def rect(a, b):
    """Simple a x b rectangle loop: a right, b up, a left, b down (area a*b)."""
    return tuple([1] * a + [2] * b + [-1] * a + [-2] * b)


# ---------------------------------------------------------------------------
# The honest loss: exact loop equation (detours direct, contacts factorized)
# ---------------------------------------------------------------------------
def exact_loop_eq_sq(U_list, fock, D, lam, loops, all_edges=True):
    """Sum of squared exact-loop-eq residuals over `loops`, at base edges."""
    inv = 1.0 / lam
    inv2 = 1.0 / (2.0 * lam)
    total = jnp.zeros((), dtype=jnp.float64)
    for C in loops:
        n = len(C)
        edges = range(n) if all_edges else (0,)
        for i in edges:
            Crot = C[i:] + C[:i]
            mu = Crot[0]
            WC = jnp.real(wilson_loop(U_list, Crot, fock, D))
            det = jnp.zeros((), dtype=jnp.float64)
            con = jnp.zeros((), dtype=jnp.float64)
            for nu in perpendicular_dirs(mu, D):
                B = detour_staple(Crot, nu)
                det = det + jnp.real(wilson_loop(U_list, B, fock, D))
                Wp = jnp.real(wilson_loop(U_list, plaquette(mu, nu), fock, D))
                con = con + Wp * WC  # base-point factorization (NOT area law)
            r = inv * det - 2.0 * WC - inv2 * con
            total = total + r * r
    return total


def make_gate_loss(fock, D, loops, weights, rp_paths, rp_time_axis,
                   sym_gens, cyc_loops, all_edges=True):
    wp = build_word_pairs(fock)
    w_le = float(weights.get("loop_eq", 1.0))
    w_cyc = float(weights.get("cyc", 0.0))
    w_rp = float(weights.get("rp", 0.0))
    w_sym = float(weights.get("sym", 0.0))

    def loss_fn(params, lam):
        U = build_forward_link_ops_matfree(params, fock, wp)
        L = w_le * exact_loop_eq_sq(U, fock, D, lam, loops, all_edges)
        if w_cyc > 0:
            L = L + w_cyc * cyclicity_loss(U, cyc_loops, fock, D)
        if w_rp > 0:
            L = L + w_rp * reflection_positivity_loss(
                U, rp_paths, fock, D, time_axis=rp_time_axis)
        if w_sym > 0:
            L = L + w_sym * lattice_symmetry_loss(
                U, cyc_loops, sym_gens, fock, D)
        return L

    return loss_fn, wp


# ---------------------------------------------------------------------------
# Evaluation: area law as a held-out prediction
# ---------------------------------------------------------------------------
CHECK_LOOPS = [
    ("plaq(A=1)", PLAQ, 1),
    ("2x1(A=2)", rect(2, 1), 2),
    ("1x2(A=2)", rect(1, 2), 2),
    ("3x1(A=3)", rect(3, 1), 3),
    ("2x2(A=4)", rect(2, 2), 4),
    ("3x2(A=6)", rect(3, 2), 6),
]


def evaluate(params, fock, wp, D, lam):
    U = build_forward_link_ops_matfree(params, fock, wp)
    W_plaq = float(jnp.real(wilson_loop(U, PLAQ, fock, D)))
    rows = []
    for name, C, area in CHECK_LOOPS:
        w_model = float(jnp.real(wilson_loop(U, C, fock, D)))
        w_exact = float(qcd2_wilson_loop(C, lam))         # w_+^area
        w_arealaw = W_plaq ** area                         # emergent-from-plaq
        rows.append({
            "name": name, "area": area,
            "model": w_model, "exact": w_exact,
            "err_abs": abs(w_model - w_exact),
            "err_rel": abs(w_model - w_exact) / max(abs(w_exact), 1e-15),
            "plaq_pow": w_arealaw,
            "dev_from_plaqpow": abs(w_model - w_arealaw),
        })
    return W_plaq, rows


def print_report(tag, lam, W_plaq, rows, final_loss):
    wp_exact = 1.0 / (2.0 * lam)
    print(f"\n{'='*74}\n{tag}: lam={lam}  w_+(exact)={wp_exact:.5f}  "
          f"W_plaq(model)={W_plaq:.5f}  final_loss={final_loss:.2e}\n{'='*74}")
    print(f"  {'loop':11s} {'area':>4s} {'model':>10s} {'exact w+^A':>11s} "
          f"{'err%':>7s} {'|W-Wplaq^A|':>12s}")
    for r in rows:
        print(f"  {r['name']:11s} {r['area']:>4d} {r['model']:>10.5f} "
              f"{r['exact']:>11.5f} {r['err_rel']*100:>6.2f}% "
              f"{r['dev_from_plaqpow']:>12.2e}")


def jit_optimize(loss_fn, params0, lam, n_steps, lr=5e-3, warmup=200,
                 grad_clip=1.0, tol=1e-13, log_every=None):
    """JIT'd Adam with warmup-cosine + the Impl-19 complex-grad conjugate fix.

    Compiles the (large, Python-unrolled) loss graph ONCE per lam, then each
    step is fast XLA -- makes all_edges=True with RP affordable."""
    warmup_c = max(1, min(warmup, max(1, n_steps - 1)))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01, peak_value=lr, warmup_steps=warmup_c,
        decay_steps=max(warmup_c + 1, n_steps), end_value=lr * 0.01)
    opt = optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(sched))
    params = [jnp.asarray(c) for c in params0]
    state = opt.init(params)
    lamf = float(lam)

    @jax.jit
    def step(params, state):
        loss, grads = jax.value_and_grad(lambda ps: loss_fn(ps, lamf))(params)
        grads = [jnp.conj(g) for g in grads]           # Impl-19 fix
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    if log_every is None:
        log_every = max(1, n_steps // 6)
    final = float("nan")
    for it in range(n_steps):
        params, state, loss = step(params, state)
        if it % log_every == 0 or it == n_steps - 1:
            final = float(loss)
            print(f"  step {it:6d}  L = {final:.6e}", flush=True)
            if final < tol:
                break
    return params, final


def run(L_trunc=4, lam_schedule=(2.0,), cold_lam=None, n_steps=2500, lr=5e-3,
        scale=0.1, seed=0, loops=None, all_edges=True,
        weights=None, tag="run"):
    if loops is None:
        loops = [PLAQ, rect(2, 1), rect(1, 2), rect(2, 2), rect(3, 1)]
    if weights is None:
        weights = {"loop_eq": 1.0, "cyc": 10.0, "rp": 1.0, "sym": 1.0}
    n_labels = 2 * D
    fock = CuntzFockJAX(n_labels=n_labels, L_trunc=L_trunc)
    cyc_loops = [C for C in loops if len(C) >= 4]
    rp_paths = positive_half_open_paths(D=D, length_cutoff=2, time_axis=D)
    sym_gens = b_d_generators(D)
    loss_fn, wp = make_gate_loss(
        fock, D, loops, weights, rp_paths, rp_time_axis=D,
        sym_gens=sym_gens, cyc_loops=cyc_loops, all_edges=all_edges)

    print(f"\n### {tag}: D={D} L_trunc={L_trunc} dim={fock.dim} "
          f"loops={len(loops)} all_edges={all_edges} weights={weights}")
    params = init_hermitian_params(n_matrices=D, fock=fock, seed=seed,
                                   scale=scale)
    t0 = time.time()
    history = []
    schedule = lam_schedule if cold_lam is None else (cold_lam,)
    for k, lam in enumerate(schedule):
        print(f"  --- {tag}: optimizing at lam={lam} ---", flush=True)
        params, final_loss = jit_optimize(
            loss_fn, params, float(lam),
            n_steps=n_steps, lr=lr, warmup=min(200, n_steps // 5))
        Wp, rows = evaluate(params, fock, wp, D, float(lam))
        print_report(f"{tag} [step {k}]", float(lam), Wp, rows, final_loss)
        history.append({"lam": float(lam), "W_plaq": Wp,
                        "final_loss": float(final_loss), "rows": rows})
    print(f"\n### {tag} wall_time={time.time()-t0:.1f}s")
    return history


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    outdir = os.path.join(REPO, "results")
    os.makedirs(outdir, exist_ok=True)

    if mode == "smoke":
        # Fast plumbing check: 3 loops, base edge 0 only, cold lam=2.
        h = run(L_trunc=4, cold_lam=2.0, n_steps=600, lr=5e-3,
                loops=[PLAQ, rect(2, 1), rect(1, 2)], all_edges=False,
                weights={"loop_eq": 1.0, "cyc": 10.0, "rp": 0.0, "sym": 0.0},
                tag="SMOKE-cold")
        json.dump(h, open(os.path.join(outdir, "qcd2_gate_smoke.json"), "w"),
                  indent=2)

    elif mode == "cold":
        h = run(L_trunc=4, cold_lam=2.0, n_steps=3000, lr=5e-3, tag="COLD")
        json.dump(h, open(os.path.join(outdir, "qcd2_gate_cold.json"), "w"),
                  indent=2)

    elif mode == "cont":
        # Continuation from strong coupling (easy) down to lam=2.
        h = run(L_trunc=4, lam_schedule=(8.0, 5.0, 3.0, 2.0),
                n_steps=2500, lr=5e-3, tag="CONT")
        json.dump(h, open(os.path.join(outdir, "qcd2_gate_cont.json"), "w"),
                  indent=2)

    elif mode == "cont5":
        # Disambiguate Fock truncation vs under-determination: SAME loop set,
        # SAME continuation, but larger Fock cutoff L_trunc=5 (dim 1365).
        h = run(L_trunc=5, lam_schedule=(8.0, 5.0, 3.0, 2.0),
                n_steps=2000, lr=5e-3, tag="CONT5")
        json.dump(h, open(os.path.join(outdir, "qcd2_gate_cont5.json"), "w"),
                  indent=2)

    elif mode == "scanhi":
        # Higher LOOP CUTOFF at the SAME Fock (L_trunc=5): constrain loops up
        # to area ~6. Compared with cont5 (base cutoff), isolates the
        # loop-cutoff axis (reach-vs-cutoff) at fixed Fock size.
        hi_loops = [PLAQ, rect(2, 1), rect(1, 2), rect(3, 1), rect(1, 3),
                    rect(2, 2), rect(4, 1), rect(1, 4), rect(3, 2), rect(2, 3)]
        h = run(L_trunc=5, lam_schedule=(8.0, 5.0, 3.0, 2.0),
                n_steps=2000, lr=5e-3, loops=hi_loops, tag="SCANHI")
        json.dump(h, open(os.path.join(outdir, "qcd2_gate_scanhi.json"), "w"),
                  indent=2)
