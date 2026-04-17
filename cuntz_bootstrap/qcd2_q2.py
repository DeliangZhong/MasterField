"""Step 3 pipeline validation (A'): D=2 unsupervised Q2 test.

The plaquette MM equation (Impl-32) is the only exact MM we have at D=2
strong coupling. Via factorization it reduces to the Gross-Witten formula
w_+ = 1/(2 lam). This script tests whether the ansatz, trained under

  L = w_mm_plaq * |plaquette_mm_residual|^2
      + w_cyc * L_cyc
      + w_rp  * L_RP
      + w_sym * L_sym

from RANDOM initialisation, converges to the master field (simultaneous
area-law Wilson loops). If yes, the Q2 pipeline works at D=2 and we move
to D=3 (Phase C) for the genuinely novel test.

Plaquette MM equation (derived in Impl-32):

  (1/lam) * [W[empty] + W[1x2]] - 2 * W[plaq] - (1/lam) * W[plaq]^2 = 0

This collapses to 1/lam = 2 W[plaq] via the area-law identity
W[1x2] = W[plaq]^2.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .cyclicity import cyclicity_loss
from .diagnostics import boundary_norm, interior_unitarity
from .fock import CuntzFockJAX
from .hermitian_operator import init_hermitian_params
from .lattice_symmetry import b_d_generators, lattice_symmetry_loss
from .matfree_expm import build_forward_link_ops_matfree, build_word_pairs
from .optimize import optimize_cuntz
from .qcd2_exact import qcd2_wilson_loop
from .reflection_positivity import (
    positive_half_open_paths,
    reflection_positivity_loss,
)
from .wilson_loops import wilson_loop


PLAQ = (1, 2, -1, -2)
RECT_1x2 = (1, 2, 2, -1, -2, -2)


def plaquette_mm_residual(
    U_list: list[jnp.ndarray], fock: CuntzFockJAX, D: int, lam: float,
) -> jnp.ndarray:
    """The exact plaquette MM residual at D=2:

        (1/lam) * [W[empty] + W[1x2]] - 2 * W[plaq] - (1/lam) * W[plaq]^2

    Returns a real scalar. Zero at the master field (strong coupling).
    """
    W_empty = jnp.ones((), dtype=jnp.float64)                 # <Omega|Omega>=1
    W_plaq = jnp.real(wilson_loop(U_list, PLAQ, fock, D))
    W_1x2 = jnp.real(wilson_loop(U_list, RECT_1x2, fock, D))
    return (
        (W_empty + W_1x2) / lam - 2.0 * W_plaq - (W_plaq ** 2) / lam
    )


def make_q2_loss(
    fock: CuntzFockJAX,
    D: int,
    word_pairs,
    cyc_test_loops: list[tuple[int, ...]],
    rp_paths: list[tuple[int, ...]],
    rp_time_axis: int,
    sym_generators: list[Callable],
    weights: dict,
) -> Callable:
    """Unsupervised loss: plaquette MM + cyclicity + RP + sym.

    weights keys: 'mm_plaq', 'cyc', 'rp', 'sym'."""
    w_mm = float(weights.get("mm_plaq", 0.0))
    w_cyc = float(weights.get("cyc", 0.0))
    w_rp = float(weights.get("rp", 0.0))
    w_sym = float(weights.get("sym", 0.0))

    def loss_fn(params: list[jnp.ndarray], lam: float) -> jnp.ndarray:
        U_list = build_forward_link_ops_matfree(params, fock, word_pairs)

        L_mm = jnp.zeros((), dtype=jnp.float64)
        if w_mm > 0.0:
            r = plaquette_mm_residual(U_list, fock, D, lam)
            L_mm = r * r

        L_cyc = jnp.zeros((), dtype=jnp.float64)
        if w_cyc > 0.0:
            L_cyc = cyclicity_loss(U_list, cyc_test_loops, fock, D)

        L_rp = jnp.zeros((), dtype=jnp.float64)
        if w_rp > 0.0:
            L_rp = reflection_positivity_loss(
                U_list, rp_paths, fock, D, time_axis=rp_time_axis
            )

        L_sym = jnp.zeros((), dtype=jnp.float64)
        if w_sym > 0.0:
            L_sym = lattice_symmetry_loss(
                U_list, cyc_test_loops, sym_generators, fock, D
            )

        return w_mm * L_mm + w_cyc * L_cyc + w_rp * L_rp + w_sym * L_sym

    return loss_fn


def run_q2_validation(
    D: int = 2,
    L_trunc: int = 4,
    lam: float = 5.0,
    n_steps: int = 5000,
    lr: float = 5e-3,
    warmup: int = 200,
    seed: int = 0,
    scale: float = 0.05,
    weights: dict | None = None,
    cyc_min_length: int = 4,
    rp_length_cutoff: int = 2,
    output_dir: Path = Path("results/step3_q2"),
) -> dict:
    """Random-init unsupervised training at D=2 with exact plaquette MM."""
    if weights is None:
        weights = {"mm_plaq": 1.0, "cyc": 10.0, "rp": 1.0, "sym": 1.0}

    output_dir.mkdir(parents=True, exist_ok=True)
    n_labels = 2 * D

    fock = CuntzFockJAX(n_labels=n_labels, L_trunc=L_trunc)
    params = init_hermitian_params(
        n_matrices=D, fock=fock, seed=seed, scale=scale,
    )
    word_pairs = build_word_pairs(fock)

    # Test loops: plaquette, 2x1, 1x2, 2x2, 3x1, fig-8 — same as Step 2
    test_loops = [
        PLAQ,
        (1, 1, 2, -1, -1, -2),
        RECT_1x2,
        (1, 1, 2, 2, -1, -1, -2, -2),
        (1, 1, 1, 2, -1, -1, -1, -2),
        (1, 2, -1, -2, -1, 2, 1, -2),
    ]
    cyc_test_loops = [C for C in test_loops if len(C) >= cyc_min_length]
    rp_paths = positive_half_open_paths(
        D=D, length_cutoff=rp_length_cutoff, time_axis=D
    )
    sym_generators = b_d_generators(D)

    loss_fn = make_q2_loss(
        fock=fock, D=D, word_pairs=word_pairs,
        cyc_test_loops=cyc_test_loops,
        rp_paths=rp_paths, rp_time_axis=D,
        sym_generators=sym_generators,
        weights=weights,
    )

    print(
        f"\n>>> Q2 validation: D={D}, L_trunc={L_trunc}, dim={fock.dim}, "
        f"lam={lam}, n_steps={n_steps}, lr={lr}, scale={scale} <<<"
    )
    print(f">>> Params: {D} x {fock.dim} complex = {2*D*fock.dim} real <<<")
    print(f">>> Weights: {weights}")
    print(f">>> Cyc loops: {[str(C) for C in cyc_test_loops]}")
    print(f">>> RP paths: {len(rp_paths)} open paths")
    print(f">>> Sym gens: {len(sym_generators)}")

    t0 = time.time()
    res = optimize_cuntz(
        loss_fn=loss_fn, params0=params, lam=lam,
        n_steps=n_steps, lr=lr, warmup=warmup,
        log_every=max(1, n_steps // 10), verbose=True,
    )
    wall_time = time.time() - t0

    return _final_report(
        res, fock, D, lam, L_trunc, scale, word_pairs,
        test_loops, cyc_test_loops, weights, output_dir, wall_time,
    )


def _final_report(
    res, fock, D, lam, L_trunc, scale, word_pairs,
    test_loops, cyc_test_loops, weights, output_dir, wall_time,
):
    U_list = build_forward_link_ops_matfree(res.params, fock, word_pairs)

    per_target: dict[str, dict] = {}
    for C in test_loops:
        w_model = float(jnp.real(wilson_loop(U_list, C, fock, D)))
        w_exact = float(qcd2_wilson_loop(C, lam))
        err_abs = abs(w_model - w_exact)
        err_rel = err_abs / abs(w_exact) if abs(w_exact) > 1e-15 else err_abs
        per_target[str(C)] = {
            "loop": list(C),
            "length": len(C),
            "model": w_model,
            "exact": w_exact,
            "err_abs": err_abs,
            "err_rel": err_rel,
        }

    mm_res_final = float(
        plaquette_mm_residual(U_list, fock, D, lam)
    )
    cyc_final = float(
        cyclicity_loss(U_list, cyc_test_loops, fock, D)
    )
    unit_final = max(interior_unitarity(U, fock) for U in U_list)
    vac = fock.vacuum_state()
    bn_final = max(boundary_norm(U @ vac, fock) for U in U_list)

    report = {
        "D": D,
        "L_trunc": L_trunc,
        "dim": int(fock.dim),
        "lam": lam,
        "scale_init": scale,
        "weights": weights,
        "per_target": per_target,
        "plaquette_mm_residual": mm_res_final,
        "cyclicity_residual": cyc_final,
        "interior_unitarity": unit_final,
        "boundary_norm_single": bn_final,
        "final_loss": float(res.final_loss),
        "n_steps_run": int(res.n_steps_run),
        "final_grad_norm": float(res.grad_norm),
        "wall_time_s": wall_time,
    }
    filename = f"q2_D{D}_Ltrunc{L_trunc}_lam{lam}.json"
    (output_dir / filename).write_text(json.dumps(report, indent=2))
    _print_report(report)
    return report


def _print_report(r: dict) -> None:
    print()
    print("=" * 72)
    print(
        f"Q2 validation final — D={r['D']} L_trunc={r['L_trunc']} "
        f"dim={r['dim']} lam={r['lam']}"
    )
    print("=" * 72)
    for name, d in r["per_target"].items():
        err = d["err_rel"]
        mark = "PASS" if err < 0.10 else "FAIL"
        print(
            f"  W[{name}] len={d['length']}: "
            f"model={d['model']:+.6e} exact={d['exact']:+.6e} "
            f"err={err * 100:>7.2f}% [{mark}]"
        )
    print(f"  plaq_mm_resid     = {r['plaquette_mm_residual']:.3e}")
    print(f"  cyclicity_resid   = {r['cyclicity_residual']:.3e}")
    print(f"  interior_unitarity= {r['interior_unitarity']:.3e}")
    print(f"  boundary_single   = {r['boundary_norm_single']:.3e}")
    print(f"  final_loss        = {r['final_loss']:.3e}")
    print(f"  n_steps_run       = {r['n_steps_run']}")
    print(f"  wall_time_s       = {r['wall_time_s']:.1f}")


if __name__ == "__main__":
    run_q2_validation(D=2, L_trunc=4, lam=5.0, n_steps=5000, lr=5e-3)
