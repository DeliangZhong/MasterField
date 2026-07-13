"""Matrix-free 2D area-law gate -- push the loop cutoff + Fock size to measure
REACH vs COST (the one remaining lever for a master-field-vs-bootstrap claim).

Same non-circular loss as qcd2_area_gate.py (exact lattice loop equation:
detours direct, contacts factorized; NO area law fed in), but evaluated fully
matrix-free via expm_iH_v -- never forms a dense U -- so L_trunc can go past 5
(dense wall) and the loop cutoff can grow.

Modes:
  smoke      : L3, 3 loops, 100 steps, single lam=2 + diagnose. Plumbing check.
  smoke2     : one engine, cont(3->2) then a fresh identical solve; the step-0
               wall prints prove ONE XLA compile is reused across every solve.
  smokeckpt  : L3, 2 loops, 60 steps, lam=(3,2), checkpointed. Resume-proof:
               run, drop the last-stage ckpt (simulated crash), rerun -> the
               resumed lam=2 stage reproduces the first run digit-for-digit.
  est5/est6  : build the reach5 / reach6lite engine, time 5 steps, project cost.
  seeds      : L4, 5 seeds x cold(lam=2)+cont(8,5,3,2); selection spread.
  validate5  : L_trunc=5, base cutoff, continuation -> MUST reproduce the dense
               cont5 (area-2 ~1.4%, area-3/4 off). Correctness gate.
  reach5     : L5, area<=4 loops, reps edges, rp=2, continuation, checkpointed.
  reach6     : L_trunc=6, HIGHER loop cutoff (up to area 6), all-edges cont.
  reach6lite : L_trunc=6, area<=4 loops, reps edges, rp=3, checkpointed -- the
               converged L6 lam=2 area-3/4 measurement.

Convention (pinned): SU(N) N=inf, lam='t Hooft, w_+ = 1/(2 lam), lam >= 1.

Run:
  uv run --no-project --with numpy --with matplotlib --with jax --with optax \
      python qcd2_area_gate_matfree.py validate5
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

REPO = "/Users/deliangzhong/Documents/Working/Master Field"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "cluster"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import jax.nn  # noqa: E402
import optax  # noqa: E402

from cuntz_bootstrap.fock import CuntzFockJAX  # noqa: E402
from cuntz_bootstrap.hermitian_operator import init_hermitian_params  # noqa: E402
from cuntz_bootstrap.matfree_expm import (  # noqa: E402
    build_word_pairs, expm_iH_v, h_matvec, expm_iH_v_norm_check,
)
from cuntz_bootstrap.wilson_loops import wilson_loop_matfree  # noqa: E402
from cuntz_bootstrap.cyclicity import cyclicity_loss_matfree  # noqa: E402
from cuntz_bootstrap.reflection_positivity import (  # noqa: E402
    positive_half_open_paths,
    reflect_path,
)
from cuntz_bootstrap.lattice_symmetry import b_d_generators  # noqa: E402
from cuntz_bootstrap.qcd2_exact import qcd2_wilson_loop  # noqa: E402
from cuntz_bootstrap.lattice_loop_eq import (  # noqa: E402
    perpendicular_dirs, detour_staple, plaquette,
)

D = 2
PLAQ = (1, 2, -1, -2)
ORDER = 30  # Taylor order for expm_iH_v (1e-13 for ||H||<=1, ~1e-8 for ||H||~5)


def rect(a, b):
    return tuple([1] * a + [2] * b + [-1] * a + [-2] * b)


def W_mf(params, C, fock, wp):
    return jnp.real(wilson_loop_matfree(params, C, fock, wp, D, ORDER))


# ---- edge selection ---------------------------------------------------------
def _rep_edges(C):
    """Representative base edges: first |mu|=1 edge and first |mu|=2 edge.

    Cyclic-rotation residuals are near-redundant under the active cyclicity
    loss (W is cyclic-invariant there), and the two direction classes |mu|=1
    vs |mu|=2 are tied by the active B_D symmetry loss; each retained residual
    is the SAME exact loop equation. Keeping one representative per direction
    class per loop is therefore sufficient. (Verified effect in the lost
    version: reach5 loop set 40 residuals -> 12.)
    """
    out = []
    for target in (1, 2):
        for i in range(len(C)):
            if abs(C[i]) == target:
                out.append(i)
                break
    return out


def _edges_for(C, edge_mode):
    if edge_mode == "all":
        return range(len(C))
    if edge_mode == "reps":
        return _rep_edges(C)
    if edge_mode == "base":
        return (0,)
    raise ValueError(f"unknown edge_mode {edge_mode!r}")


# ---- non-circular exact loop-equation loss (matrix-free) ----
def loop_eq_sq_mf(params, fock, wp, lam, loops, edge_mode="all"):
    inv, inv2 = 1.0 / lam, 1.0 / (2.0 * lam)
    total = jnp.zeros((), dtype=jnp.float64)
    for C in loops:
        for i in _edges_for(C, edge_mode):
            Crot = C[i:] + C[:i]
            mu = Crot[0]
            WC = W_mf(params, Crot, fock, wp)
            det = jnp.zeros((), dtype=jnp.float64)
            con = jnp.zeros((), dtype=jnp.float64)
            for nu in perpendicular_dirs(mu, D):
                det = det + W_mf(params, detour_staple(Crot, nu), fock, wp)
                con = con + W_mf(params, plaquette(mu, nu), fock, wp) * WC
            r = inv * det - 2.0 * WC - inv2 * con
            total = total + r * r
    return total


# ---- reflection positivity (matrix-free path vectors) ----
def _path_vec_mf(params, p, fock, wp):
    v = fock.vacuum_state()
    for mu in reversed(p):
        h = params[abs(mu) - 1]
        sign = 1.0 if mu > 0 else -1.0
        v = expm_iH_v(h, v, wp, order=ORDER, sign=sign)
    return v


def rp_loss_mf(params, fock, wp, paths, time_axis):
    vs = jnp.stack([_path_vec_mf(params, p, fock, wp) for p in paths], axis=1)
    tvs = jnp.stack(
        [_path_vec_mf(params, reflect_path(p, time_axis), fock, wp)
         for p in paths], axis=1)
    R = tvs.conj().T @ vs
    R = 0.5 * (R + R.conj().T)
    evals = jnp.linalg.eigvalsh(R)
    return jnp.sum(jax.nn.relu(-evals) ** 2)


# ---- lattice symmetry (matrix-free) ----
def sym_loss_mf(params, fock, wp, loops, gens):
    total = jnp.zeros((), dtype=jnp.float64)
    for C in loops:
        if not C:
            continue
        W0 = W_mf(params, C, fock, wp)
        for sg in gens:
            total = total + (W_mf(params, sg(C), fock, wp) - W0) ** 2
    return total


def make_loss(fock, wp, loops, weights, rp_paths, rp_time_axis, gens,
              cyc_loops, edge_mode="all"):
    w_le = float(weights.get("loop_eq", 1.0))
    w_cyc = float(weights.get("cyc", 0.0))
    w_rp = float(weights.get("rp", 0.0))
    w_sym = float(weights.get("sym", 0.0))

    def loss_fn(params, lam):
        L = w_le * loop_eq_sq_mf(params, fock, wp, lam, loops, edge_mode)
        if w_cyc > 0:
            L = L + w_cyc * cyclicity_loss_matfree(
                params, cyc_loops, fock, wp, D, ORDER)
        if w_rp > 0:
            L = L + w_rp * rp_loss_mf(params, fock, wp, rp_paths, rp_time_axis)
        if w_sym > 0:
            L = L + w_sym * sym_loss_mf(params, fock, wp, cyc_loops, gens)
        return L

    return loss_fn


def make_opt(lr, n_steps, warmup, grad_clip=1.0):
    """Optimizer with schedule fixed by (lr, n_steps, warmup, grad_clip).

    Built ONCE per mode (these are constant within a mode). `opt.init(params)`
    at the start of each lambda-solve resets the schedule step-count, so one
    opt object across solves is exactly equivalent to the old per-solve build.
    """
    warmup_c = max(1, min(warmup, max(1, n_steps - 1)))
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01, peak_value=lr, warmup_steps=warmup_c,
        decay_steps=max(warmup_c + 1, n_steps), end_value=lr * 0.01)
    return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(sched))


def make_step(loss_fn, opt):
    """Build the jitted training step ONCE, with lam as a TRACED argument.

    lam enters loss_fn only arithmetically (1/lam, 1/(2 lam) in
    loop_eq_sq_mf) -- no Python control flow depends on it -- so a traced
    float64 scalar keeps one compiled executable valid for EVERY lambda:
    exactly ONE XLA compile per mode instead of one per lambda-step x seed.
    """

    @jax.jit
    def step(params, state, lam_arr):
        loss, grads = jax.value_and_grad(
            lambda ps: loss_fn(ps, lam_arr))(params)
        grads = [jnp.conj(g) for g in grads]
        updates, state = opt.update(grads, state, params)
        params = optax.apply_updates(params, updates)
        return params, state, loss

    return step


def jit_optimize(step_fn, opt, params0, lam, n_steps, tol=1e-14,
                 log_every=None, ckpt_cb=None, ckpt_every=1000):
    """One lambda-solve reusing a prebuilt (step_fn, opt). Prints the step-0
    wall time: large on the FIRST solve of a mode (XLA compile), ~per-step
    time on every subsequent solve (compile-cache hit).

    ckpt_cb, if given, is called ckpt_cb(params) every `ckpt_every` steps to
    write params_rolling.npy -- DISASTER INSURANCE ONLY (a mid-stage kill can
    salvage the latest weights). The supported resume path is STAGE-LEVEL
    (per-lambda .npy); the rolling file is never consulted by _try_resume.
    """
    params = [jnp.asarray(c) for c in params0]
    state = opt.init(params)
    lam_arr = jnp.asarray(float(lam), dtype=jnp.float64)
    if log_every is None:
        log_every = max(1, n_steps // 6)
    final = float("nan")
    for it in range(n_steps):
        if it == 0:
            t_s0 = time.time()
        params, state, loss = step_fn(params, state, lam_arr)
        if it == 0:
            loss.block_until_ready()
            print(f"  step-0 wall = {time.time()-t_s0:.2f}s", flush=True)
        if it % log_every == 0 or it == n_steps - 1:
            final = float(loss)
            print(f"  step {it:6d}  L = {final:.6e}", flush=True)
            if final < tol:
                break
        if ckpt_cb is not None and it > 0 and it % ckpt_every == 0:
            ckpt_cb(params)
    return params, final


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
CHECK = [("plaq", PLAQ, 1), ("2x1", rect(2, 1), 2), ("1x2", rect(1, 2), 2),
         ("3x1", rect(3, 1), 3), ("2x2", rect(2, 2), 4),
         ("3x2", rect(3, 2), 6), ("3x3", rect(3, 3), 9)]


def evaluate(params, fock, wp, lam):
    Wp = float(W_mf(params, PLAQ, fock, wp))
    rows = []
    for name, C, area in CHECK:
        wm = float(W_mf(params, C, fock, wp))
        we = float(qcd2_wilson_loop(C, lam))
        rows.append({"name": name, "area": area, "model": wm, "exact": we,
                     "err_rel": abs(wm - we) / max(abs(we), 1e-18)})
    return Wp, rows


def print_report(tag, lam, Wp, rows, loss):
    print(f"\n{'='*70}\n{tag}: lam={lam} w+={1/(2*lam):.5f} "
          f"Wplaq={Wp:.5f} loss={loss:.2e}\n{'='*70}")
    for r in rows:
        print(f"  {r['name']:5s} A={r['area']:>2d} model={r['model']:>10.5f} "
              f"exact={r['exact']:>11.5f} err={r['err_rel']*100:>8.2f}%")


# ---------------------------------------------------------------------------
# A. Diagnostics (audit item F)
# ---------------------------------------------------------------------------
def _spectral_norm_H(h, wp, dim, iters=30, seed=0):
    """Estimate ||H||_2 via power iteration on H using h_matvec.

    H = sum_w h_w C_w + h.c. is Hermitian, so power iteration converges to the
    eigenvalue of largest magnitude; with v normalised each step, ||H v|| ->
    |lambda|_max = ||H||_2 (Rayleigh quotient <v|H|v> is real and tends to the
    same magnitude). Purely diagnostic (a Taylor-order sanity gate)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    v = jnp.asarray(v.astype(np.complex128))
    v = v / jnp.linalg.norm(v)
    est = 0.0
    for _ in range(iters):
        Hv = h_matvec(h, v, wp)
        est = float(jnp.linalg.norm(Hv))
        v = Hv / (est + 1e-300)
    return est


def diagnose(params, fock, wp, tag):
    """Per direction: ||H|| estimate, relative last-Taylor-term norm, and a
    unitarity probe |||expm_iH_v(h,vac)|| - 1|. Flags [ok]/[WARN] so a run
    that pushed ||H|| into the poorly-converged Taylor regime is caught."""
    print(f"### diagnose [{tag}]:", flush=True)
    vac = fock.vacuum_state()
    ok_all = True
    for d, h in enumerate(params):
        hn = _spectral_norm_H(h, wp, fock.dim)
        _, last_rel = expm_iH_v_norm_check(h, vac, wp, order=ORDER)
        u = expm_iH_v(h, vac, wp, order=ORDER)
        unit = abs(float(jnp.linalg.norm(u)) - 1.0)
        ok = (last_rel < 1e-10) and (unit < 1e-8)
        ok_all = ok_all and ok
        mark = "[ok]" if ok else "[WARN]"
        print(f"  dir {d+1}: ||H||~{hn:.4f} last_Taylor_rel={last_rel:.2e} "
              f"|‖u‖-1|={unit:.2e} {mark}", flush=True)
    return ok_all


# ---------------------------------------------------------------------------
# B. Higher-Fock cross-check (audit item E)
# ---------------------------------------------------------------------------
def embed_params(params, small_fock, big_fock):
    """Map each small-Fock coefficient vector into the big-Fock basis by
    basis-word IDENTITY (small basis is a subset of big basis); zero on the
    words that exist only in the bigger space."""
    idx_map = np.array(
        [big_fock.basis_to_idx[w] for w in small_fock.basis], dtype=np.int64)
    out = []
    for h in params:
        big = np.zeros(big_fock.dim, dtype=np.complex128)
        big[idx_map] = np.asarray(h)
        out.append(jnp.asarray(big))
    return out


def verify_embedding(params, small_fock, small_wp, big_fock, big_wp, tag=""):
    """Exact algebraic index-mapping check (max_coeff_err MUST be 0.0,
    n_misplaced MUST be 0) plus a low-||H|| dynamical W_plaq gap probe (the
    embedded field must give the SAME short-loop amplitudes)."""
    emb = embed_params(params, small_fock, big_fock)
    idx_map = {si: big_fock.basis_to_idx[w]
               for si, w in enumerate(small_fock.basis)}
    mapped_big = set(idx_map.values())
    max_coeff_err = 0.0
    n_misplaced = 0
    for h_small, h_big in zip(params, emb):
        h_small = np.asarray(h_small)
        h_big = np.asarray(h_big)
        for si, bi in idx_map.items():
            max_coeff_err = max(max_coeff_err, abs(h_big[bi] - h_small[si]))
        for k in np.nonzero(h_big)[0]:
            if int(k) not in mapped_big:
                n_misplaced += 1
    Wp_small = float(W_mf(params, PLAQ, small_fock, small_wp))
    Wp_big = float(W_mf(emb, PLAQ, big_fock, big_wp))
    gap = abs(Wp_small - Wp_big)
    print(f"### verify_embedding [{tag}]: max_coeff_err={max_coeff_err:.1e} "
          f"n_misplaced={n_misplaced} W_plaq_gap={gap:.2e} "
          f"(small={Wp_small:.6f} big={Wp_big:.6f})", flush=True)
    return {"max_coeff_err": float(max_coeff_err), "n_misplaced": n_misplaced,
            "wplaq_gap": gap}


def eval_at(params, small_fock, small_wp, L_eval, lam, tag=""):
    """Rebuild (fock, word_pairs) at L_eval and re-evaluate the CHECK loops
    with the EMBEDDED params. Reports model@L_train, model@L_eval, exact, and
    the ABSOLUTE deviation at L_eval (relative error is meaningless where the
    exact value underflows -- flagged)."""
    big_fock = CuntzFockJAX(n_labels=2 * D, L_trunc=L_eval)
    big_wp = build_word_pairs(big_fock)
    verify_embedding(params, small_fock, small_wp, big_fock, big_wp, tag)
    emb = embed_params(params, small_fock, big_fock)
    print(f"\n### eval_at [{tag}]: L_train={small_fock.L_trunc} "
          f"L_eval={L_eval} lam={lam}", flush=True)
    print(f"  {'loop':5s} {'A':>2s} {'model@Ltr':>11s} {'model@Lev':>11s} "
          f"{'exact':>11s} {'|dev@Lev|':>11s}", flush=True)
    rows = []
    for name, C, area in CHECK:
        wtr = float(W_mf(params, C, small_fock, small_wp))
        wev = float(W_mf(emb, C, big_fock, big_wp))
        we = float(qcd2_wilson_loop(C, lam))
        dev = abs(wev - we)
        note = "  (exact<1e-4: rel-err is noise)" if abs(we) < 1e-4 else ""
        print(f"  {name:5s} {area:>2d} {wtr:>11.5f} {wev:>11.5f} "
              f"{we:>11.5f} {dev:>11.2e}{note}", flush=True)
        rows.append({"name": name, "area": area, "model_Ltrain": wtr,
                     "model_Leval": wev, "exact": we, "abs_dev_Leval": dev})
    return rows


# ---------------------------------------------------------------------------
# C. Engine (build once, reuse the compiled step across every lambda-solve)
# ---------------------------------------------------------------------------
def build_engine(L_trunc, loops, n_steps, lr=5e-3, warmup=None,
                 rp_length_cutoff=2, edge_mode="all", weights=None):
    if weights is None:
        weights = {"loop_eq": 1.0, "cyc": 10.0, "rp": 1.0, "sym": 1.0}
    if warmup is None:
        warmup = min(200, max(1, n_steps // 5))
    fock = CuntzFockJAX(n_labels=2 * D, L_trunc=L_trunc)
    t_wp = time.time()
    wp = build_word_pairs(fock)
    print(f"### engine: L_trunc={L_trunc} dim={fock.dim} nnz={wp.n_nnz} "
          f"(word_pairs {time.time()-t_wp:.1f}s) loops={len(loops)} "
          f"edge_mode={edge_mode} rp_cut={rp_length_cutoff} n_steps={n_steps} "
          f"lr={lr} weights={weights}", flush=True)
    cyc_loops = [C for C in loops if len(C) >= 4]
    rp_paths = positive_half_open_paths(D=D, length_cutoff=rp_length_cutoff,
                                        time_axis=D)
    print(f"### rp_paths: n={len(rp_paths)} (length_cutoff={rp_length_cutoff})",
          flush=True)
    gens = b_d_generators(D)
    loss_fn = make_loss(fock, wp, loops, weights, rp_paths, D, gens, cyc_loops,
                        edge_mode)
    opt = make_opt(lr, n_steps, warmup)
    step_fn = make_step(loss_fn, opt)
    return {"fock": fock, "wp": wp, "loss_fn": loss_fn, "opt": opt,
            "step_fn": step_fn, "loops": loops, "cyc_loops": cyc_loops,
            "rp_paths": rp_paths, "n_steps": n_steps, "L_trunc": L_trunc,
            "edge_mode": edge_mode, "rp_length_cutoff": rp_length_cutoff,
            "weights": weights, "lr": lr}


# ---------------------------------------------------------------------------
# E. Checkpointing (stage-level resume + rolling disaster insurance)
# ---------------------------------------------------------------------------
def _stack_params(params):
    return np.stack([np.asarray(h) for h in params], axis=0)  # (D, dim) c128


def _save_rolling(ckpt_dir, params):
    np.save(os.path.join(ckpt_dir, "params_rolling.npy"), _stack_params(params))


def _save_stage(ckpt_dir, params, hist, tag, L_trunc, seed, n_steps, lam,
                stage_index, edge_mode, rp_length_cutoff):
    np.save(os.path.join(ckpt_dir, f"params_lam{lam}.npy"),
            _stack_params(params))
    meta = {"tag": tag, "L_trunc": int(L_trunc), "seed": int(seed),
            "n_steps": int(n_steps), "lam_done": float(lam),
            "stage_index": int(stage_index), "edge_mode": edge_mode,
            "rp_length_cutoff": int(rp_length_cutoff)}
    json.dump(meta, open(os.path.join(ckpt_dir, f"meta_lam{lam}.json"), "w"),
              indent=2)
    json.dump(hist, open(os.path.join(ckpt_dir, "history.json"), "w"), indent=2)


def _try_resume(ckpt_dir, lam_schedule, tag, L_trunc, seed, fock):
    """Longest completed prefix of lam_schedule with valid meta (matching
    L_trunc/seed, correct stage_index/lam, params shape). Skip those, warm-
    start from the last saved params, restore the matching history records.
    Any mismatch -> warn + fresh start (return None)."""
    completed = []
    last_params = None
    for k, lam in enumerate(lam_schedule):
        meta_path = os.path.join(ckpt_dir, f"meta_lam{lam}.json")
        pf = os.path.join(ckpt_dir, f"params_lam{lam}.npy")
        if not (os.path.exists(meta_path) and os.path.exists(pf)):
            break
        meta = json.load(open(meta_path))
        arr = np.load(pf)
        if (meta.get("L_trunc") != int(L_trunc)
                or meta.get("seed") != int(seed)
                or meta.get("stage_index") != k
                or float(meta.get("lam_done")) != float(lam)
                or arr.shape != (D, fock.dim)):
            print(f"### RESUME [{tag}]: WARNING checkpoint for lam={lam} "
                  f"MISMATCH (meta L={meta.get('L_trunc')} seed={meta.get('seed')}"
                  f" stage={meta.get('stage_index')} shape={arr.shape} vs "
                  f"L={L_trunc} seed={seed} stage={k} ({D},{fock.dim})); "
                  f"starting FRESH", flush=True)
            return None
        completed.append(lam)
        last_params = arr
    if not completed:
        return None
    hist_path = os.path.join(ckpt_dir, "history.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else []
    hist = hist[:len(completed)]
    params = [jnp.asarray(last_params[d]) for d in range(D)]
    nxt = (lam_schedule[len(completed)]
           if len(completed) < len(lam_schedule) else "DONE")
    print(f"### RESUME [{tag}]: found completed stages lam={completed} "
          f"(matching L_trunc={L_trunc} seed={seed}); SKIPPING them, "
          f"warm-starting from lam={completed[-1]} (next lam={nxt}, "
          f"{len(hist)} history record(s) restored)", flush=True)
    return params, hist, len(completed)


# ---------------------------------------------------------------------------
# run: one continuation (optionally engine-shared and/or checkpointed)
# ---------------------------------------------------------------------------
def run(L_trunc, loops, lam_schedule, n_steps, lr=5e-3, scale=0.1, seed=0,
        weights=None, edge_mode="all", rp_length_cutoff=2, tag="MF",
        engine=None, ckpt_dir=None, tol=1e-14):
    if engine is None:
        engine = build_engine(L_trunc, loops, n_steps, lr=lr,
                              rp_length_cutoff=rp_length_cutoff,
                              edge_mode=edge_mode, weights=weights)
    else:
        assert engine["n_steps"] == n_steps, \
            f"engine n_steps {engine['n_steps']} != {n_steps}"
        assert engine["L_trunc"] == L_trunc, \
            f"engine L_trunc {engine['L_trunc']} != {L_trunc}"
        edge_mode = engine["edge_mode"]
        rp_length_cutoff = engine["rp_length_cutoff"]
    fock, wp = engine["fock"], engine["wp"]
    step_fn, opt = engine["step_fn"], engine["opt"]
    print(f"### {tag}: L_trunc={L_trunc} dim={fock.dim} seed={seed} "
          f"lams={tuple(float(x) for x in lam_schedule)}", flush=True)

    params = init_hermitian_params(n_matrices=D, fock=fock, seed=seed,
                                   scale=scale)
    lam_schedule = [float(x) for x in lam_schedule]
    hist = []
    start_index = 0
    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        resumed = _try_resume(ckpt_dir, lam_schedule, tag, L_trunc, seed, fock)
        if resumed is not None:
            params, hist, start_index = resumed
        else:
            # Disaster recovery: no completed stage, but a mid-stage rolling
            # save exists (kill between stage boundaries). Warm-start the
            # WHOLE schedule from it: the current stage restarts its step
            # budget from the saved params (optimizer state fresh) -- loses
            # at most 1000 steps of progress, never hours.
            rolling = os.path.join(ckpt_dir, "params_rolling.npy")
            if os.path.exists(rolling):
                arr = np.load(rolling)
                if arr.shape == (D, fock.dim):
                    params = [jnp.asarray(arr[i]) for i in range(D)]
                    print(f"### DISASTER-RECOVERY [{tag}]: no completed stage "
                          f"ckpt; warm-starting from params_rolling.npy "
                          f"(mid-stage save; stage restarts, optimizer fresh)",
                          flush=True)
                else:
                    print(f"### [{tag}] ignoring params_rolling.npy: shape "
                          f"{arr.shape} != {(D, fock.dim)}", flush=True)

    ckpt_cb = (lambda p: _save_rolling(ckpt_dir, p)) if ckpt_dir else None
    t0 = time.time()
    for k in range(start_index, len(lam_schedule)):
        lam = lam_schedule[k]
        print(f"  --- {tag}: lam={lam} ---", flush=True)
        params, loss = jit_optimize(step_fn, opt, params, lam, n_steps=n_steps,
                                    tol=tol, ckpt_cb=ckpt_cb)
        Wp, rows = evaluate(params, fock, wp, lam)
        print_report(f"{tag}[{k}]", lam, Wp, rows, loss)
        hist.append({"lam": lam, "W_plaq": Wp, "final_loss": float(loss),
                     "dim": int(fock.dim), "nnz": int(wp.n_nnz), "rows": rows})
        if ckpt_dir is not None:
            _save_stage(ckpt_dir, params, hist, tag, L_trunc, seed, n_steps,
                        lam, k, edge_mode, rp_length_cutoff)
    print(f"### {tag} wall={time.time()-t0:.1f}s", flush=True)
    return hist, params, fock, wp


# ---------------------------------------------------------------------------
# reach engine configs (shared by est5/est6 and reach5/reach6lite)
# ---------------------------------------------------------------------------
REACH_LOOPS = [PLAQ, rect(2, 1), rect(1, 2), rect(3, 1), rect(1, 3), rect(2, 2)]


def reach5_engine():
    # area <= 4 loops, reps edges, rp=2 (2*2=4 <= L_trunc=5).
    return build_engine(5, REACH_LOOPS, n_steps=4000, lr=5e-3,
                        rp_length_cutoff=2, edge_mode="reps")


def reach6lite_engine():
    # area <= 4 loops, reps edges, rp=3 (2*3=6 <= L_trunc=6, boundary).
    return build_engine(6, REACH_LOOPS, n_steps=4000, lr=5e-3,
                        rp_length_cutoff=3, edge_mode="reps")


def _cost_probe(engine, mode, project_steps=16000):
    fock, wp = engine["fock"], engine["wp"]
    step_fn, opt = engine["step_fn"], engine["opt"]
    params = init_hermitian_params(n_matrices=D, fock=fock, seed=0, scale=0.1)
    p = [jnp.asarray(c) for c in params]
    state = opt.init(p)
    lam_arr = jnp.asarray(2.0, dtype=jnp.float64)
    t0 = time.time()
    p, state, loss = step_fn(p, state, lam_arr)
    loss.block_until_ready()
    t_compile = time.time() - t0
    t1 = time.time()
    for _ in range(4):
        p, state, loss = step_fn(p, state, lam_arr)
    loss.block_until_ready()
    s_per_step = (time.time() - t1) / 4.0
    proj = s_per_step * project_steps
    print(f"### {mode}: dim={fock.dim} nnz={wp.n_nnz} "
          f"compile+step0={t_compile:.1f}s s/step={s_per_step:.2f} "
          f"projected {project_steps} steps = {proj/3600:.2f}h ({proj:.0f}s)",
          flush=True)


# ---------------------------------------------------------------------------
# F. Modes
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    outdir = os.path.join(REPO, "results")
    SCRATCH = os.path.join(REPO, "scratch")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(SCRATCH, exist_ok=True)

    if mode == "smoke":
        loops = [PLAQ, rect(2, 1), rect(1, 2)]
        hist, params, fock, wp = run(3, loops, (2.0,), n_steps=100,
                                     edge_mode="all", rp_length_cutoff=2,
                                     tag="SMOKE")
        diagnose(params, fock, wp, "SMOKE")

    elif mode == "smoke2":
        # Compile reuse: ONE engine, two full solves. Only the very first
        # lambda-solve pays the XLA compile (large step-0 wall); every later
        # solve is a compile-cache hit (step-0 ~ per-step).
        loops = [PLAQ, rect(2, 1), rect(1, 2)]
        eng = build_engine(3, loops, n_steps=100, lr=5e-3,
                           rp_length_cutoff=2, edge_mode="all")
        print("### smoke2: solve #1 (continuation lam 3->2)")
        run(3, loops, (3.0, 2.0), n_steps=100, engine=eng, tag="SMOKE2a")
        print("### smoke2: solve #2 (fresh identical solve, SAME engine)")
        run(3, loops, (3.0, 2.0), n_steps=100, engine=eng, tag="SMOKE2b")

    elif mode == "smokeckpt":
        # Resume-proof. Run once (saves lam=3 then lam=2). To exercise resume,
        # drop the last-stage ckpt (simulated crash after lam=3) and rerun:
        # _try_resume skips lam=3 and reruns lam=2 from the lossless .npy,
        # reproducing the first run digit-for-digit.
        loops = [PLAQ, rect(2, 1)]
        ckpt = os.path.join(SCRATCH, "ckpt_smoke")
        run(3, loops, (3.0, 2.0), n_steps=60, edge_mode="all",
            rp_length_cutoff=2, tag="SMOKECKPT", ckpt_dir=ckpt)

    elif mode == "est5":
        _cost_probe(reach5_engine(), "est5")

    elif mode == "est6":
        _cost_probe(reach6lite_engine(), "est6")

    elif mode == "seeds":
        # Selection spread across seeds (data already banked in
        # results/qcd2_gate_seeds.json; this mode is for reproducibility).
        loops = [PLAQ, rect(2, 1), rect(1, 2), rect(2, 2), rect(3, 1)]
        eng = build_engine(4, loops, n_steps=1500, lr=5e-3,
                           rp_length_cutoff=2, edge_mode="all")
        fock, wp = eng["fock"], eng["wp"]
        step_fn, opt = eng["step_fn"], eng["opt"]
        recs = []
        for seed in range(5):
            for kind, sched in (("cold", (2.0,)),
                                ("cont", (8.0, 5.0, 3.0, 2.0))):
                params = init_hermitian_params(D, fock, seed=seed, scale=0.1)
                loss = float("nan")
                for lam in sched:
                    params, loss = jit_optimize(step_fn, opt, params, lam,
                                                n_steps=1500)
                Wp, rows = evaluate(params, fock, wp, 2.0)
                wmap = {r["name"]: r["model"] for r in rows}
                any_neg = any(r["model"] < 0 for r in rows)
                recs.append({"seed": seed, "kind": kind,
                             "final_loss": float(loss), "W_plaq": Wp,
                             "W_2x1": wmap["2x1"], "W_2x2": wmap["2x2"],
                             "any_negative": bool(any_neg),
                             "physical": bool(not any_neg)})
                print(f"  seed {seed} {kind}: loss={loss:.2e} Wp={Wp:.5f} "
                      f"physical={not any_neg}", flush=True)
        print("\n### seeds summary")
        print(f"  {'seed':>4s} {'kind':>5s} {'loss':>10s} {'W_plaq':>8s} "
              f"{'W_2x1':>8s} {'W_2x2':>8s} {'phys':>6s}")
        for r in recs:
            print(f"  {r['seed']:>4d} {r['kind']:>5s} {r['final_loss']:>10.2e} "
                  f"{r['W_plaq']:>8.5f} {r['W_2x1']:>8.5f} {r['W_2x2']:>8.5f} "
                  f"{str(r['physical']):>6s}")
        json.dump(recs, open(os.path.join(outdir, "qcd2_gate_seeds.json"), "w"),
                  indent=2)

    elif mode == "validate5":
        # MUST reproduce dense cont5: base cutoff, L_trunc=5, continuation,
        # all edges, rp=2 (2*2=4 <= 5).
        base = [PLAQ, rect(2, 1), rect(1, 2), rect(2, 2), rect(3, 1)]
        hist, params, fock, wp = run(5, base, (8.0, 5.0, 3.0, 2.0),
                                     n_steps=2000, edge_mode="all",
                                     rp_length_cutoff=2, tag="MF-VAL5")
        json.dump(hist, open(os.path.join(outdir, "qcd2_gate_mf_val5.json"),
                             "w"), indent=2)

    elif mode == "reach5":
        # RP consistency: a path of length l contributes Gram entries that are
        # words of length 2l, so we need 2*rp_cut <= L_trunc; otherwise those
        # words are past the Fock truncation and RP is corrupted (an
        # un-optimizable floor ~5e-2, measured 10/10 runs at rp=4/L=4).
        # Corollary: RP protects loops only out to half-perimeter L_trunc/2.
        # Here L=5, rp=2 -> 2*2=4 <= 5. OK.
        ckpt = os.path.join(SCRATCH, "ckpt_reach5")
        eng = reach5_engine()
        hist, params, fock, wp = run(5, REACH_LOOPS, (8.0, 5.0, 3.0, 2.0),
                                     n_steps=4000, tag="MF-REACH5",
                                     engine=eng, ckpt_dir=ckpt, tol=1e-14)
        diagnose(params, fock, wp, "MF-REACH5")
        eval_at(params, fock, wp, L_eval=6, lam=2.0, tag="MF-REACH5")
        json.dump(hist, open(os.path.join(outdir, "qcd2_gate_mf_reach5.json"),
                             "w"), indent=2)

    elif mode == "reach6lite":
        # Purpose: the converged L6, lam=2, area-3/4 measurement. The killed
        # all-edges reach6 salvage showed 3x1 at 6.7% at lam=3 but
        # under-converged; scanhi (results/qcd2_gate_scanhi.json, dense L5,
        # converged 9.3e-8) shows L5 cannot pin area-3 (42% at lam=2; length-10
        # staple words corrupted at L5). This run closes the gap, checkpointed
        # so nothing is lost again.
        # RP consistency (see reach5): L=6, rp=3 -> 2*3=6 <= 6 (boundary OK).
        ckpt = os.path.join(SCRATCH, "ckpt_reach6lite")
        eng = reach6lite_engine()
        hist, params, fock, wp = run(6, REACH_LOOPS, (8.0, 5.0, 3.0, 2.0),
                                     n_steps=4000, tag="MF-REACH6LITE",
                                     engine=eng, ckpt_dir=ckpt, tol=1e-14)
        diagnose(params, fock, wp, "MF-REACH6LITE")
        eval_at(params, fock, wp, L_eval=7, lam=2.0, tag="MF-REACH6LITE")
        json.dump(hist,
                  open(os.path.join(outdir, "qcd2_gate_mf_reach6lite.json"),
                       "w"), indent=2)

    elif mode == "reach6":
        # Higher loop cutoff (up to area 6) at L_trunc=6, all-edges cont.
        # RP consistency (see reach5): L=6, rp=3 -> 2*3=6 <= 6 (boundary OK).
        hi = [PLAQ, rect(2, 1), rect(1, 2), rect(3, 1), rect(1, 3),
              rect(2, 2), rect(3, 2), rect(2, 3)]
        hist, params, fock, wp = run(6, hi, (8.0, 5.0, 3.0, 2.0), n_steps=2000,
                                     edge_mode="all", rp_length_cutoff=3,
                                     tag="MF-REACH6")
        json.dump(hist, open(os.path.join(outdir, "qcd2_gate_mf_reach6.json"),
                             "w"), indent=2)

    else:
        raise SystemExit(f"unknown mode {mode!r}")
