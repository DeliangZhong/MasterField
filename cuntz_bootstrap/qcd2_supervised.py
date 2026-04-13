"""Steps 2, 2.5, 2.6 of Phase 4 v3: supervised representational tests (Q1).

Step 2 (DONE, Impl-27): 6 cherry-picked targets (plaq, 2x1, 1x2, 2x2, 3x1,
fig-8). All fit to machine precision at L_trunc=3, lam=5.

Step 2.5 (this file): stretch test. Supervised fit to ALL canonical D=2
loops up to L_max via master_field/lattice.enumerate_closed_loops. Tests
whether the same 340-parameter ansatz at L_trunc=3 generalizes beyond
the 6 Step 2 targets.

Step 2.6 (this file): multi-coupling test. Four independent supervised
runs at lam in {2, 3, 5, 10}. Tests whether the ansatz structure works
across the full physical coupling range.

Run:
  python3 -m cuntz_bootstrap.qcd2_supervised                  # Step 2 ladder
  python3 -c "from cuntz_bootstrap.qcd2_supervised \\
              import run_stretch_test; run_stretch_test(L_max=8)"
  python3 -c "from cuntz_bootstrap.qcd2_supervised \\
              import run_multi_coupling; run_multi_coupling()"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .cyclicity import cyclicity_loss
from .diagnostics import boundary_norm, interior_unitarity
from .fock import CuntzFockJAX
from .hermitian_operator import (
    build_forward_link_ops,
    init_hermitian_params,
)
from .optimize import optimize_cuntz
from .qcd2_exact import qcd2_wilson_loop
from .wilson_loops import wilson_loop

# Same sys.path idiom used by exact_mm.py to import from ../master_field
_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import abs_area_2d, enumerate_closed_loops  # noqa: E402


# --- target set ----------------------------------------------------------

PLAQ = (1, 2, -1, -2)
RECT_2x1 = (1, 1, 2, -1, -1, -2)
RECT_1x2 = (1, 2, 2, -1, -2, -2)
RECT_2x2 = (1, 1, 2, 2, -1, -1, -2, -2)
RECT_3x1 = (1, 1, 1, 2, -1, -1, -1, -2)
FIG8 = (1, 2, -1, -2, -1, 2, 1, -2)

TARGET_NAMES: dict[tuple[int, ...], str] = {
    PLAQ: "plaq",
    RECT_2x1: "2x1",
    RECT_1x2: "1x2",
    RECT_2x2: "2x2",
    RECT_3x1: "3x1",
    FIG8: "fig8",
}


def build_targets(lam: float) -> list[tuple[tuple[int, ...], float]]:
    """(loop, exact W[loop]) pairs at coupling lam.

    Exact values come from `qcd2_wilson_loop`, which uses window
    decomposition so that the figure-8 evaluates to w_+^2 rather than
    w_+^0 = 1.
    """
    loops = [PLAQ, RECT_2x1, RECT_1x2, RECT_2x2, RECT_3x1, FIG8]
    return [(C, float(qcd2_wilson_loop(C, lam))) for C in loops]


# --- loss ---------------------------------------------------------------

def make_supervised_loss(
    targets: list[tuple[tuple[int, ...], float]],
    cyc_words: list[tuple[int, ...]],
    fock: CuntzFockJAX,
    D: int,
    w_cyc: float = 10.0,
) -> Callable[[list[jnp.ndarray], float], jnp.ndarray]:
    """L = sum_C |W_model[C] - W_exact[C]|^2 + w_cyc * L_cyc.

    Uses |.|^2 rather than (Re(.))^2 so the imaginary part is also driven
    to zero (a diagnostic that the ansatz respects planar reality).
    """
    def loss_fn(params: list[jnp.ndarray], _lam: float) -> jnp.ndarray:
        U_list = build_forward_link_ops(params, fock)
        L_sup = jnp.zeros((), dtype=jnp.float64)
        for loop, w_exact in targets:
            w_model = wilson_loop(U_list, loop, fock, D)
            L_sup = L_sup + jnp.abs(w_model - w_exact) ** 2
        L_cyc = cyclicity_loss(U_list, cyc_words, fock, D)
        return L_sup + w_cyc * L_cyc

    return loss_fn


# --- main ---------------------------------------------------------------

def run_step2(
    D: int = 2,
    L_trunc: int = 3,
    lam: float = 5.0,
    n_steps: int = 3000,
    lr: float = 1e-3,
    warmup: int = 200,
    seed: int = 0,
    scale: float = 0.05,
    w_cyc: float = 10.0,
    output_dir: Path = Path("results/step2"),
) -> dict:
    """Single supervised run. n_labels = 2D per v3 convention.

    Memory note: _build_word_operators caches fock.dim matrices of size
    (fock.dim, fock.dim). For n=4:
        L_trunc=3  dim=85    ~10 MB build
        L_trunc=4  dim=341   ~630 MB build
        L_trunc=5  dim=1365  ~40 GB build  (INFEASIBLE)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 2 * D

    fock = CuntzFockJAX(n_labels=n_labels, L_trunc=L_trunc)
    params = init_hermitian_params(
        n_matrices=D, fock=fock, seed=seed, scale=scale,
    )

    targets = build_targets(lam)
    # Cyclicity on three structurally-distinct loops (no length-2 degenerates)
    cyc_words = [PLAQ, RECT_2x1, FIG8]

    loss_fn = make_supervised_loss(targets, cyc_words, fock, D, w_cyc=w_cyc)

    print(f"\n>>> Step 2: D={D}, L_trunc={L_trunc}, dim={fock.dim}, "
          f"lam={lam}, n_steps={n_steps}, lr={lr}, scale={scale} <<<")
    print(f">>> Params: {D} x {fock.dim} complex = {2 * D * fock.dim} real <<<")
    for loop, w_exact in targets:
        print(f"  target W[{TARGET_NAMES[loop]}] = {w_exact:+.6e}")

    res = optimize_cuntz(
        loss_fn=loss_fn,
        params0=params,
        lam=lam,
        n_steps=n_steps,
        lr=lr,
        warmup=warmup,
        log_every=max(1, n_steps // 10),
        verbose=True,
    )

    return _final_report(
        res, fock, D, targets, cyc_words, lam, L_trunc, scale, output_dir
    )


def _final_report(
    res,
    fock: CuntzFockJAX,
    D: int,
    targets: list[tuple[tuple[int, ...], float]],
    cyc_words: list[tuple[int, ...]],
    lam: float,
    L_trunc: int,
    scale: float,
    output_dir: Path,
) -> dict:
    U_list = build_forward_link_ops(res.params, fock)

    per_target: dict[str, dict] = {}
    for loop, w_exact in targets:
        w_complex = wilson_loop(U_list, loop, fock, D)
        w_real = float(jnp.real(w_complex))
        w_imag = float(jnp.imag(w_complex))
        if abs(w_exact) > 1e-15:
            err_rel = abs(w_real - w_exact) / abs(w_exact)
        else:
            err_rel = abs(w_real - w_exact)
        per_target[TARGET_NAMES[loop]] = {
            "model_real": w_real,
            "model_imag": w_imag,
            "exact": w_exact,
            "err_rel": err_rel,
        }

    cyc_res = float(cyclicity_loss(U_list, cyc_words, fock, D))
    vac = fock.vacuum_state()
    bn = max(boundary_norm(U @ vac, fock) for U in U_list)
    unit = max(interior_unitarity(U, fock) for U in U_list)

    gates = {
        "plaq_err": per_target["plaq"]["err_rel"] < 0.01,
        "2x1_err": per_target["2x1"]["err_rel"] < 0.05,
        "1x2_err": per_target["1x2"]["err_rel"] < 0.05,
        "2x2_err": per_target["2x2"]["err_rel"] < 0.10,
        "3x1_err": per_target["3x1"]["err_rel"] < 0.10,
        "fig8_err": per_target["fig8"]["err_rel"] < 0.05,
        "cyclicity": cyc_res < 1e-6,
        "boundary": bn < 1e-2,
        "unitarity": unit < 1e-6,
    }
    all_pass = all(gates.values())

    report = {
        "D": D,
        "L_trunc": L_trunc,
        "lam": lam,
        "scale_init": scale,
        "dim": int(fock.dim),
        "per_target": per_target,
        "cyclicity_residual": cyc_res,
        "boundary_norm": bn,
        "interior_unitarity": unit,
        "final_loss": float(res.final_loss),
        "n_steps_run": int(res.n_steps_run),
        "final_grad_norm": float(res.grad_norm),
        "gates": gates,
        "Q1_verdict": "YES" if all_pass else "NO",
    }
    filename = f"D{D}_Ltrunc{L_trunc}_lam{lam}.json"
    (output_dir / filename).write_text(json.dumps(report, indent=2))
    _print_report(report)
    return report


def _print_report(report: dict) -> None:
    print()
    print("=" * 72)
    print(f"STEP 2 final — D={report['D']} L_trunc={report['L_trunc']} "
          f"dim={report['dim']} lam={report['lam']}")
    print("=" * 72)
    for name, d in report["per_target"].items():
        mark = "PASS" if d["err_rel"] < 0.10 else "FAIL"
        print(f"  W[{name:<5}] = {d['model_real']:+.6e}  "
              f"(Im {d['model_imag']:+.3e})  "
              f"exact {d['exact']:+.6e}  "
              f"err {d['err_rel'] * 100:>7.2f}%  [{mark}]")
    print(f"  cyclicity   = {report['cyclicity_residual']:.3e}  "
          f"(gate < 1e-6, {'PASS' if report['gates']['cyclicity'] else 'FAIL'})")
    print(f"  boundary    = {report['boundary_norm']:.3e}  "
          f"(gate < 1e-2, {'PASS' if report['gates']['boundary'] else 'FAIL'})")
    print(f"  unitarity   = {report['interior_unitarity']:.3e}  "
          f"(gate < 1e-6, {'PASS' if report['gates']['unitarity'] else 'FAIL'})")
    print(f"  final_loss  = {report['final_loss']:.3e}")
    print(f"  n_steps_run = {report['n_steps_run']}")
    print(f"  *** Q1 verdict: {report['Q1_verdict']} ***")


def run_ladder(lam: float = 5.0, n_steps: int = 3000) -> list[dict]:
    """Scale ladder: L_trunc=3 -> L_trunc=4 (stop at first pass)."""
    reports: list[dict] = []
    for L_trunc in [3, 4]:
        print(f"\n{'#' * 72}\n# Ladder rung: L_trunc={L_trunc}\n{'#' * 72}")
        rpt = run_step2(D=2, L_trunc=L_trunc, lam=lam, n_steps=n_steps)
        reports.append(rpt)
        if rpt["Q1_verdict"] == "YES":
            print(f"\n>>> Q1 passed at L_trunc={L_trunc}. Ready for Step 3. <<<")
            return reports
    print("\n>>> Q1 did NOT pass up to L_trunc=4. Next: add mixed a^dag a "
          "terms or conclude ansatz inadequate. <<<")
    return reports


# =====================================================================
# Step 2.5: stretch test — supervised fit to all canonical D=2 loops
# =====================================================================


def build_targets_stretch(
    lam: float, L_max: int, min_length: int = 4,
) -> list[tuple[tuple[int, ...], float]]:
    """All canonical D=2 closed loops with min_length <= |C| <= L_max."""
    loops = enumerate_closed_loops(D=2, max_length=L_max)
    out: list[tuple[tuple[int, ...], float]] = []
    for C in loops:
        if min_length <= len(C) <= L_max:
            out.append((C, float(qcd2_wilson_loop(C, lam))))
    return out


def multi_step_boundary_probe(
    U_list: list[jnp.ndarray], fock: CuntzFockJAX, max_depth: int = 8,
) -> dict[int, list[float]]:
    """{mu_idx: [boundary_norm(U_mu^k |Omega>) for k in 1..max_depth]}.

    Diagnostic for truncation adequacy at deeper chain lengths than a
    single Uhat application. Not a gate; informational only.
    """
    out: dict[int, list[float]] = {}
    v0 = fock.vacuum_state()
    for mu, U in enumerate(U_list):
        norms: list[float] = []
        v = v0
        for _ in range(max_depth):
            v = U @ v
            norms.append(boundary_norm(v, fock))
        out[mu] = norms
    return out


def _final_report_stretch(
    res,
    fock: CuntzFockJAX,
    D: int,
    targets: list[tuple[tuple[int, ...], float]],
    cyc_words: list[tuple[int, ...]],
    lam: float,
    L_trunc: int,
    L_max: int,
    scale: float,
    output_dir: Path,
    wall_time_s: float,
    err_abs_thresh: float = 5e-4,
    err_rel_thresh: float = 0.05,
) -> dict:
    """Grouped report for stretch-test runs (many targets).

    Per-target stored in per_target (keyed by str(loop)). Aggregated
    statistics by loop length in by_length. Pass gate: for each target,
    EITHER err_rel < err_rel_thresh OR abs(W_model - W_exact) < err_abs_thresh
    (tiny exact values get relaxed relative-err gate)."""
    U_list = build_forward_link_ops(res.params, fock)

    per_target: dict[str, dict] = {}
    by_length: dict[int, list[float]] = {}
    all_imag_abs: list[float] = []
    worst_over_thresh: list[tuple[str, float]] = []

    n_fail_abs = 0
    for loop, w_exact in targets:
        w_complex = wilson_loop(U_list, loop, fock, D)
        w_real = float(jnp.real(w_complex))
        w_imag = float(jnp.imag(w_complex))
        err_abs = abs(w_real - w_exact)
        err_rel = err_abs / abs(w_exact) if abs(w_exact) > 1e-15 else err_abs
        # pass = lenient OR rel
        passed = (err_abs < err_abs_thresh) or (err_rel < err_rel_thresh)
        if not passed:
            n_fail_abs += 1
            worst_over_thresh.append((str(loop), err_rel))
        per_target[str(loop)] = {
            "loop": list(loop),
            "length": len(loop),
            "area": int(abs_area_2d(loop)),
            "model_real": w_real,
            "model_imag": w_imag,
            "exact": w_exact,
            "err_abs": err_abs,
            "err_rel": err_rel,
            "passed": passed,
        }
        by_length.setdefault(len(loop), []).append(err_rel)
        all_imag_abs.append(abs(w_imag))

    length_stats = {
        L: {
            "n": len(errs),
            "mean_err_rel": float(sum(errs) / len(errs)),
            "max_err_rel": float(max(errs)),
        }
        for L, errs in sorted(by_length.items())
    }

    cyc_res = float(cyclicity_loss(U_list, cyc_words, fock, D))
    unit = max(interior_unitarity(U, fock) for U in U_list)
    vac = fock.vacuum_state()
    bn_single = max(boundary_norm(U @ vac, fock) for U in U_list)
    depth_probe = multi_step_boundary_probe(
        U_list, fock, max_depth=max(8, L_max // 2 + 2)
    )
    # serialise depth_probe keys to str for JSON
    depth_probe_json = {str(k): v for k, v in depth_probe.items()}

    worst_err_rel = max(
        (t["err_rel"] for t in per_target.values()), default=0.0
    )
    max_imag_abs = max(all_imag_abs, default=0.0)

    all_targets_pass = n_fail_abs == 0
    cyc_pass = cyc_res < 1e-6
    unit_pass = unit < 1e-6

    report = {
        "mode": "stretch",
        "D": D,
        "L_trunc": L_trunc,
        "L_max": L_max,
        "lam": lam,
        "scale_init": scale,
        "dim": int(fock.dim),
        "n_targets": len(targets),
        "cyc_words": [list(c) for c in cyc_words],
        "per_target": per_target,
        "by_length_stats": length_stats,
        "worst_err_rel": worst_err_rel,
        "max_imag_abs": max_imag_abs,
        "n_fail": n_fail_abs,
        "worst_over_thresh": worst_over_thresh[:20],
        "cyclicity_residual": cyc_res,
        "boundary_norm_single": bn_single,
        "boundary_probe_depth": depth_probe_json,
        "interior_unitarity": unit,
        "final_loss": float(res.final_loss),
        "n_steps_run": int(res.n_steps_run),
        "final_grad_norm": float(res.grad_norm),
        "wall_time_s": wall_time_s,
        "gates": {
            "all_targets": all_targets_pass,
            "cyclicity": cyc_pass,
            "unitarity": unit_pass,
        },
        "Q1_verdict": (
            "YES" if (all_targets_pass and cyc_pass and unit_pass) else "NO"
        ),
    }

    filename = f"stretch_D{D}_Lmax{L_max}_Ltrunc{L_trunc}_lam{lam}.json"
    (output_dir / filename).write_text(json.dumps(report, indent=2))
    _print_report_stretch(report)
    return report


def _print_report_stretch(r: dict) -> None:
    print()
    print("=" * 72)
    print(
        f"STRETCH final — D={r['D']} L_trunc={r['L_trunc']} "
        f"L_max={r['L_max']} dim={r['dim']} lam={r['lam']} "
        f"n_targets={r['n_targets']}"
    )
    print("=" * 72)
    print("  By-length statistics (relative error):")
    for L, s in r["by_length_stats"].items():
        print(
            f"    len={L:>2}: n={s['n']:>4}  mean_err={s['mean_err_rel']:.3e}  "
            f"max_err={s['max_err_rel']:.3e}"
        )
    print(f"  worst_err_rel   = {r['worst_err_rel']:.3e}")
    print(f"  max_imag_abs    = {r['max_imag_abs']:.3e}  (should be ~0)")
    print(
        f"  n_fail          = {r['n_fail']} / {r['n_targets']}  "
        f"(|err| >= 5e-4 AND rel >= 5%)"
    )
    if r["worst_over_thresh"]:
        print("  top failing loops:")
        for loop_str, err in r["worst_over_thresh"][:5]:
            print(f"    {loop_str}  err_rel={err:.3e}")
    print(
        f"  cyclicity       = {r['cyclicity_residual']:.3e}  "
        f"({'PASS' if r['gates']['cyclicity'] else 'FAIL'})"
    )
    print(f"  boundary_single = {r['boundary_norm_single']:.3e}")
    print(
        f"  unitarity       = {r['interior_unitarity']:.3e}  "
        f"({'PASS' if r['gates']['unitarity'] else 'FAIL'})"
    )
    print(f"  final_loss      = {r['final_loss']:.3e}")
    print(f"  n_steps_run     = {r['n_steps_run']}")
    print(f"  wall_time_s     = {r['wall_time_s']:.1f}")
    print("  boundary_probe_depth (Û_μ^k|Ω⟩ boundary mass for k=1..):")
    for mu, norms in r["boundary_probe_depth"].items():
        print(
            f"    mu={mu}: "
            + "  ".join(f"{v:.2e}" for v in norms[: min(8, len(norms))])
        )
    print(f"  *** Stretch verdict: {r['Q1_verdict']} ***")


def _pick_cyc_words(
    targets: list[tuple[tuple[int, ...], float]], n: int = 3
) -> list[tuple[int, ...]]:
    """Pick n structurally diverse cyclicity test loops from the target set.

    Prefers PLAQ, RECT_2x2, FIG8 if present; falls back to first n loops
    of distinct lengths >= 4.
    """
    chosen: list[tuple[int, ...]] = []
    preferred = [PLAQ, RECT_2x2, FIG8]
    loops_in = {C for C, _ in targets}
    for C in preferred:
        if C in loops_in and C not in chosen:
            chosen.append(C)
    if len(chosen) >= n:
        return chosen[:n]
    # Fallback: distinct lengths, longest first
    by_len: dict[int, tuple[int, ...]] = {}
    for C, _ in targets:
        if len(C) >= 4 and len(C) not in by_len:
            by_len[len(C)] = C
    for L in sorted(by_len, reverse=True):
        if len(chosen) >= n:
            break
        if by_len[L] not in chosen:
            chosen.append(by_len[L])
    return chosen[:n] if chosen else [PLAQ]


def run_stretch_test(
    L_max: int = 8,
    L_trunc: int = 3,
    D: int = 2,
    lam: float = 5.0,
    n_steps: int = 5000,
    lr: float = 5e-3,
    warmup: int = 200,
    seed: int = 0,
    scale: float = 0.05,
    w_cyc: float = 10.0,
    output_dir: Path = Path("results/step2_5"),
) -> dict:
    """Supervised fit to ALL canonical D=2 loops up to L_max. Step 2.5."""
    output_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 2 * D
    fock = CuntzFockJAX(n_labels=n_labels, L_trunc=L_trunc)
    params = init_hermitian_params(
        n_matrices=D, fock=fock, seed=seed, scale=scale,
    )

    targets = build_targets_stretch(lam, L_max=L_max)
    cyc_words = _pick_cyc_words(targets, n=3)

    loss_fn = make_supervised_loss(targets, cyc_words, fock, D, w_cyc=w_cyc)

    print(
        f"\n>>> Step 2.5 stretch: D={D}, L_trunc={L_trunc}, dim={fock.dim}, "
        f"L_max={L_max}, n_targets={len(targets)}, lam={lam}, "
        f"n_steps={n_steps}, lr={lr}, scale={scale}, w_cyc={w_cyc} <<<"
    )
    print(
        f">>> Params: {D} x {fock.dim} complex = {2 * D * fock.dim} real <<<"
    )
    # Print length distribution of targets
    length_count: dict[int, int] = {}
    for C, _ in targets:
        length_count[len(C)] = length_count.get(len(C), 0) + 1
    print("  target length distribution: "
          + ", ".join(
              f"len={L}:{n}" for L, n in sorted(length_count.items())
          ))
    print("  cyc_words: " + ", ".join(str(c) for c in cyc_words))

    t0 = time.time()
    res = optimize_cuntz(
        loss_fn=loss_fn,
        params0=params,
        lam=lam,
        n_steps=n_steps,
        lr=lr,
        warmup=warmup,
        log_every=max(1, n_steps // 10),
        verbose=True,
    )
    wall_time_s = time.time() - t0

    return _final_report_stretch(
        res, fock, D, targets, cyc_words, lam, L_trunc, L_max, scale,
        output_dir, wall_time_s,
    )


# =====================================================================
# Step 2.6: multi-coupling test
# =====================================================================


def run_multi_coupling(
    lams: tuple[float, ...] = (2.0, 3.0, 5.0, 10.0),
    L_max: int = 6,
    L_trunc: int = 3,
    D: int = 2,
    n_steps: int = 5000,
    lr: float = 5e-3,
    seed: int = 0,
    scale: float = 0.05,
    w_cyc: float = 10.0,
    output_dir: Path = Path("results/step2_6"),
) -> dict:
    """Run independent supervised fits at each lam; verify all converge."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_lam: list[dict] = []
    for lam in lams:
        print(f"\n{'#' * 72}\n# Multi-coupling rung: lam={lam}\n{'#' * 72}")
        report = run_stretch_test(
            L_max=L_max,
            L_trunc=L_trunc,
            D=D,
            lam=lam,
            n_steps=n_steps,
            lr=lr,
            seed=seed,
            scale=scale,
            w_cyc=w_cyc,
            output_dir=output_dir,
        )
        per_lam.append({
            "lam": lam,
            "final_loss": report["final_loss"],
            "worst_err_rel": report["worst_err_rel"],
            "n_fail": report["n_fail"],
            "n_targets": report["n_targets"],
            "cyclicity_residual": report["cyclicity_residual"],
            "boundary_norm_single": report["boundary_norm_single"],
            "interior_unitarity": report["interior_unitarity"],
            "Q1_verdict": report["Q1_verdict"],
            "wall_time_s": report["wall_time_s"],
        })

    all_pass = all(r["Q1_verdict"] == "YES" for r in per_lam)
    summary = {
        "mode": "multi_coupling",
        "lams": list(lams),
        "L_max": L_max,
        "L_trunc": L_trunc,
        "per_lam": per_lam,
        "all_lams_pass": all_pass,
        "worst_across_lams_err_rel": max(
            r["worst_err_rel"] for r in per_lam
        ),
        "worst_across_lams_loss": max(r["final_loss"] for r in per_lam),
    }
    (output_dir / f"multi_coupling_Lmax{L_max}_Ltrunc{L_trunc}.json").write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("=" * 72)
    print("STEP 2.6 multi-coupling SUMMARY")
    print("=" * 72)
    print(
        f"  lams={list(lams)}  L_max={L_max}  L_trunc={L_trunc}  "
        f"n_targets={per_lam[0]['n_targets']}"
    )
    for r in per_lam:
        mark = "PASS" if r["Q1_verdict"] == "YES" else "FAIL"
        print(
            f"  lam={r['lam']:>5.1f}: loss={r['final_loss']:.2e}  "
            f"worst_err_rel={r['worst_err_rel']:.2e}  "
            f"bdry={r['boundary_norm_single']:.2e}  [{mark}]"
        )
    print(f"  *** all lams pass: {all_pass} ***")
    return summary


if __name__ == "__main__":
    run_ladder(lam=5.0, n_steps=3000)
