"""Step 2 of Phase 4 v3: supervised representational test (Q1).

THE decisive experiment. Fit Uhat_1, Uhat_2 in the exp-Hermitian Cuntz-Fock
ansatz to exact QCD2 Wilson loops (plaquette through 2x2, figure-8) under
supervised optimisation at fixed lambda.

If all targets pass simultaneously, the ansatz is adequate (Q1=YES) and
unsupervised homotopy (Q2) becomes meaningful. If not, enlarge the ansatz.

Gate (all simultaneously):
  |W[plaq] - w_+| / w_+         < 1%
  |W[2x1]  - w_+^2| / w_+^2     < 5%
  |W[1x2]  - w_+^2| / w_+^2     < 5%
  |W[2x2]  - w_+^4| / w_+^4     < 10%   # Phase 3 failed by 900x
  |W[3x1]  - w_+^3| / w_+^3     < 10%
  |W[fig8] - w_+^2| / w_+^2     < 5%
  cyclicity residual             < 1e-6
  boundary norm                  < 1e-2
  ||UU^dag - I||_interior        < 1e-6

Run:   python3 -m cuntz_bootstrap.qcd2_supervised
"""
from __future__ import annotations

import json
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


if __name__ == "__main__":
    run_ladder(lam=5.0, n_steps=3000)
