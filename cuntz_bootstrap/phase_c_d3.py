"""Phase C: D=3 Cuntz-Fock bootstrap — Q1 supervised test.

At D=3 the master field has no closed-form area law. Leading-order
strong-coupling expansion gives W[plaq] = 1/(2*lam) for EACH plaquette
plane, with corrections at O(1/lam^4) from virtual perpendicular
plaquettes.

This module runs a SUPERVISED Q1 test at D=3 using leading-order
strong-coupling targets:

    Target W[C] = (1/(2*lam))^area    for planar simple loops
    Target W[C] = Pi_i w_+^area_i     for window-decomposed self-intersecting

where `area` is the lattice area of the loop's projection into its
2D plane. At leading order this is the same as the D=2 area law applied
plane-by-plane.

If this Q1 test passes (all targets fit to < 5% relative error), the
Cuntz-Fock ansatz at D=3 is viable. Next steps: null-space MM scanner
at D=3, then Q2 unsupervised.

Supersedes the v2 `phase_c_d3` (which used candidate-D MM via LoopSystem).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .cyclicity import cyclicity_loss
from .diagnostics import boundary_norm, interior_unitarity
from .fock import CuntzFockJAX
from .hermitian_operator import init_hermitian_params
from .matfree_expm import build_forward_link_ops_matfree, build_word_pairs
from .optimize import optimize_cuntz
from .qcd2_exact import qcd2_wilson_loop
from .wilson_loops import wilson_loop


# --- D=3 loop target set ------------------------------------------------

PLAQ_12 = (1, 2, -1, -2)
PLAQ_13 = (1, 3, -1, -3)
PLAQ_23 = (2, 3, -2, -3)

RECT_2x1_12 = (1, 1, 2, -1, -1, -2)
RECT_2x1_13 = (1, 1, 3, -1, -1, -3)
RECT_2x1_23 = (2, 2, 3, -2, -2, -3)

RECT_2x2_12 = (1, 1, 2, 2, -1, -1, -2, -2)
RECT_2x2_13 = (1, 1, 3, 3, -1, -1, -3, -3)
RECT_2x2_23 = (2, 2, 3, 3, -2, -2, -3, -3)

FIG8_12 = (1, 2, -1, -2, -1, 2, 1, -2)

ALL_PLAQS = [PLAQ_12, PLAQ_13, PLAQ_23]
ALL_LOOPS = [
    PLAQ_12, PLAQ_13, PLAQ_23,
    RECT_2x1_12, RECT_2x1_13, RECT_2x1_23,
    RECT_2x2_12, RECT_2x2_13, RECT_2x2_23,
    FIG8_12,
]


def planar_target(loop: tuple[int, ...], lam: float) -> float:
    """Leading-order strong-coupling target for a D=3 loop confined to
    a single 2D plane.

    For any loop entirely in directions {mu, nu}, the lattice area law
    at strong coupling (leading order in 1/lam) gives W[C] = w_+^|area|,
    identical to the D=2 result. We identify the two directions and
    relabel them as (1, 2) to reuse qcd2_wilson_loop.
    """
    dirs = sorted({abs(s) for s in loop})
    if len(dirs) != 2:
        raise ValueError(
            f"planar_target: loop must use exactly 2 directions, got {dirs}"
        )
    d_low, d_high = dirs[0], dirs[1]
    relabel = {d_low: 1, d_high: 2}
    remapped = tuple(
        (1 if s > 0 else -1) * relabel[abs(s)] for s in loop
    )
    return float(qcd2_wilson_loop(remapped, lam))


def build_targets_d3(lam: float) -> list[tuple[tuple[int, ...], float]]:
    """(loop, leading-order-target) pairs for D=3 Q1 test."""
    return [(C, planar_target(C, lam)) for C in ALL_LOOPS]


def make_supervised_loss_d3(
    targets: list[tuple[tuple[int, ...], float]],
    cyc_words: list[tuple[int, ...]],
    fock: CuntzFockJAX,
    word_pairs,
    D: int,
    w_cyc: float = 10.0,
):
    """Supervised loss using matfree hybrid."""
    def loss_fn(params, _lam):
        U_list = build_forward_link_ops_matfree(params, fock, word_pairs)
        L_sup = jnp.zeros((), dtype=jnp.float64)
        for loop, w_exact in targets:
            w_model = wilson_loop(U_list, loop, fock, D)
            L_sup = L_sup + jnp.abs(w_model - w_exact) ** 2
        L_cyc = cyclicity_loss(U_list, cyc_words, fock, D)
        return L_sup + w_cyc * L_cyc

    return loss_fn


def run_q1_d3(
    L_trunc: int = 3,
    lam: float = 10.0,
    n_steps: int = 3000,
    lr: float = 5e-3,
    warmup: int = 200,
    seed: int = 0,
    scale: float = 0.05,
    output_dir: Path = Path("results/phase_c_d3"),
) -> dict:
    """Q1 supervised run at D=3 with leading-order strong-coupling targets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    D = 3
    n_labels = 2 * D
    fock = CuntzFockJAX(n_labels=n_labels, L_trunc=L_trunc)
    params = init_hermitian_params(
        n_matrices=D, fock=fock, seed=seed, scale=scale,
    )
    word_pairs = build_word_pairs(fock)

    targets = build_targets_d3(lam)
    cyc_words = [PLAQ_12, RECT_2x1_12, RECT_2x2_12]

    loss_fn = make_supervised_loss_d3(
        targets, cyc_words, fock, word_pairs, D,
    )

    print(
        f"\n>>> Phase C D=3 Q1: L_trunc={L_trunc}, dim={fock.dim}, "
        f"lam={lam}, n_steps={n_steps}, lr={lr} <<<"
    )
    print(f">>> Params: {D} x {fock.dim} = {2*D*fock.dim} real <<<")
    print(f">>> Matfree nnz = {word_pairs.n_nnz}")
    print(">>> Targets (leading-order strong-coupling):")
    for C, w in targets:
        print(f"    W[{C}] = {w:+.6e}")

    t0 = time.time()
    res = optimize_cuntz(
        loss_fn=loss_fn, params0=params, lam=lam,
        n_steps=n_steps, lr=lr, warmup=warmup,
        log_every=max(1, n_steps // 10), verbose=True,
    )
    wall = time.time() - t0
    return _final_report(
        res, fock, word_pairs, D, targets, cyc_words,
        lam, L_trunc, scale, output_dir, wall,
    )


def _final_report(
    res, fock, word_pairs, D, targets, cyc_words,
    lam, L_trunc, scale, output_dir, wall,
):
    U_list = build_forward_link_ops_matfree(res.params, fock, word_pairs)
    per_target: dict[str, dict] = {}
    for C, w_exact in targets:
        w_c = wilson_loop(U_list, C, fock, D)
        w_real = float(jnp.real(w_c))
        w_imag = float(jnp.imag(w_c))
        err_abs = abs(w_real - w_exact)
        err_rel = err_abs / abs(w_exact) if abs(w_exact) > 1e-15 else err_abs
        per_target[str(C)] = {
            "loop": list(C),
            "length": len(C),
            "model_real": w_real,
            "model_imag": w_imag,
            "exact_leading_order": w_exact,
            "err_rel": err_rel,
        }
    cyc_final = float(cyclicity_loss(U_list, cyc_words, fock, D))
    unit_final = max(interior_unitarity(U, fock) for U in U_list)
    vac = fock.vacuum_state()
    bn_final = max(boundary_norm(U @ vac, fock) for U in U_list)

    report = {
        "D": D,
        "L_trunc": L_trunc,
        "dim": int(fock.dim),
        "lam": lam,
        "scale_init": scale,
        "per_target": per_target,
        "cyclicity_residual": cyc_final,
        "interior_unitarity": unit_final,
        "boundary_norm_single": bn_final,
        "final_loss": float(res.final_loss),
        "n_steps_run": int(res.n_steps_run),
        "final_grad_norm": float(res.grad_norm),
        "wall_time_s": wall,
    }
    filename = f"q1_d3_Ltrunc{L_trunc}_lam{lam}.json"
    (output_dir / filename).write_text(json.dumps(report, indent=2))
    _print_report(report)
    return report


def _print_report(r: dict) -> None:
    print()
    print("=" * 72)
    print(
        f"Phase C D=3 Q1 final — L_trunc={r['L_trunc']} "
        f"dim={r['dim']} lam={r['lam']}"
    )
    print("=" * 72)
    for name, d in r["per_target"].items():
        err = d["err_rel"]
        mark = "PASS" if err < 0.10 else "FAIL"
        print(
            f"  W[{name}]: "
            f"model={d['model_real']:+.6e} target={d['exact_leading_order']:+.6e} "
            f"err={err * 100:>7.2f}% [{mark}]"
        )
    print(f"  cyclicity   = {r['cyclicity_residual']:.3e}")
    print(f"  boundary    = {r['boundary_norm_single']:.3e}")
    print(f"  unitarity   = {r['interior_unitarity']:.3e}")
    print(f"  final_loss  = {r['final_loss']:.3e}")
    print(f"  n_steps_run = {r['n_steps_run']}")
    print(f"  wall_time_s = {r['wall_time_s']:.1f}")


if __name__ == "__main__":
    run_q1_d3(L_trunc=3, lam=10.0, n_steps=3000, lr=5e-3)
