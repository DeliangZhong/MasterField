"""Phase B QCD_2 critical test (v2).

D=2, L_poly = L_trunc = 3 (d = 85 Fock space). Two Hermitian generators
Hhat_1, Hhat_2 with 338 real DOFs total (2 x (2 d_L - 1)). Unsupervised loss:

    L = w_MM L_MM + w_cyc L_cyc + w_RP L_RP + w_sym L_sym

Coupling continuation lam in {10, 5, 2} (strong to intermediate) with h=0
init at lam=10.

Gate at lam = 5 (the critical test — Phase 3 failed W[2x2] by 900x):

  W[plaquette] = 1/(2 lam) = 0.1
  W[2x1]       = W[plaq]^2 = 0.01
  W[2x2]       = W[plaq]^4 = 1e-4
  Figure-8 factorisation W[fig-8] = W[lobe_1] * W[lobe_2]
  Cyclicity residuals < 1e-6
  B_2 symmetry residuals < 1e-6

Run: python3 -m cuntz_bootstrap.phase_b_qcd2
"""
from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .cyclicity import build_cyclicity_test_loops, cyclicity_loss
from .fock import CuntzFockJAX
from .hermitian_operator import build_forward_link_ops, init_hermitian_params
from .lattice_symmetry import b_d_generators, lattice_symmetry_loss
from .mm_loss import _load_loop_system, compute_all_wilson_loops
from .optimize import optimize_cuntz
from .reflection_positivity import (
    positive_half_open_paths,
    reflection_positivity_loss,
)
from .total_loss import make_total_loss_fn
from .wilson_loops import wilson_loop


def _w_plus(lam: float) -> float:
    """GW strong-coupling plaquette."""
    return 0.5 / lam if lam >= 1.0 else 1.0 - lam / 2.0


def run_phase_b(
    L_poly: int = 3,
    L_trunc: int = 3,
    L_max_loops: int = 4,
    lams: tuple[float, ...] = (10.0, 5.0, 2.0),
    n_steps: int = 3000,
    lr: float = 1e-2,
    weights: dict | None = None,
    seed: int = 42,
    output_dir: Path = Path("results/phase_b_v2"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if weights is None:
        weights = {"mm": 1.0, "cyc": 1.0, "rp": 1.0, "sym": 1.0}

    loop_sys = _load_loop_system(D=2, L_max=L_max_loops)
    fock = CuntzFockJAX(n_labels=4, L_trunc=L_trunc)

    loss_fn = make_total_loss_fn(
        loop_sys=loop_sys, fock=fock, D=2,
        weights=weights, return_components=False,
    )
    loss_components_fn = make_total_loss_fn(
        loop_sys=loop_sys, fock=fock, D=2,
        weights=weights, return_components=True,
    )

    results: dict = {}
    p_warm = None

    for lam in lams:
        if p_warm is None:
            # Use small nonzero random init to break the h=0 symmetry
            # (at h=0 all W[C] = 1 and the gradient of Re[W[C]] vanishes).
            p0 = init_hermitian_params(
                n_matrices=2, fock=fock, seed=seed, scale=0.02,
            )
        else:
            p0 = p_warm

        res = optimize_cuntz(
            loss_fn=loss_fn, params0=p0, lam=lam,
            n_steps=n_steps, lr=lr, warmup=200, log_every=200, verbose=True,
        )
        p_warm = [jnp.array(c) for c in res.params]

        # Report Wilson loops and diagnostics
        U_list = build_forward_link_ops(res.params, fock=fock)
        W_plaq = float(jnp.real(wilson_loop(U_list, (1, 2, -1, -2), fock, D=2)))
        W_2x1 = float(
            jnp.real(wilson_loop(U_list, (1, 1, 2, -1, -1, -2), fock, D=2))
        )
        W_2x2 = float(
            jnp.real(
                wilson_loop(U_list, (1, 1, 2, 2, -1, -1, -2, -2), fock, D=2)
            )
        )
        comps = loss_components_fn(res.params, lam)
        target_plaq = _w_plus(lam)
        result = {
            "lam": float(lam),
            "final_loss": float(res.final_loss),
            "L_MM": float(comps.L_MM),
            "L_cyc": float(comps.L_cyc),
            "L_RP": float(comps.L_RP),
            "L_sym": float(comps.L_sym),
            "W_plaq": W_plaq,
            "W_2x1": W_2x1,
            "W_2x2": W_2x2,
            "target_W_plaq": target_plaq,
            "target_W_2x1": target_plaq ** 2,
            "target_W_2x2": target_plaq ** 4,
            "rel_err_plaq": abs(W_plaq - target_plaq) / max(target_plaq, 1e-12),
            "rel_err_2x1": abs(W_2x1 - target_plaq ** 2)
            / max(target_plaq ** 2, 1e-12),
            "rel_err_2x2": abs(W_2x2 - target_plaq ** 4)
            / max(target_plaq ** 4, 1e-12),
        }
        results[str(lam)] = result

        print()
        print(f"=== lam = {lam} ===")
        print(f"  final_loss = {res.final_loss:.3e}")
        print(
            f"  L_MM={comps.L_MM:.3e}  L_cyc={comps.L_cyc:.3e}  "
            f"L_RP={comps.L_RP:.3e}  L_sym={comps.L_sym:.3e}"
        )
        print(
            f"  W[plaq]={W_plaq:.5f}  target={target_plaq:.5f}  "
            f"rel_err={result['rel_err_plaq']:.2%}"
        )
        print(
            f"  W[2x1] ={W_2x1:.5e}  target={target_plaq ** 2:.5e}  "
            f"rel_err={result['rel_err_2x1']:.2%}"
        )
        print(
            f"  W[2x2] ={W_2x2:.5e}  target={target_plaq ** 4:.5e}  "
            f"rel_err={result['rel_err_2x2']:.2%}  <-- CRITICAL"
        )

    (output_dir / "phase_b_v2_summary.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


if __name__ == "__main__":
    res = run_phase_b()
    lam_ref = "5.0"
    if lam_ref in res:
        r = res[lam_ref]
        worst_rel = max(r["rel_err_plaq"], r["rel_err_2x1"], r["rel_err_2x2"])
        worst_constraint = max(r["L_cyc"], r["L_sym"])
        print()
        print("=== Phase B gate (lam = 5) ===")
        print(f"  worst Wilson-loop rel err = {worst_rel:.2%}  (gate: < 1%)")
        print(f"  L_cyc + L_sym             = {worst_constraint:.3e}  (gate: < 1e-6)")
        if worst_rel < 0.01 and worst_constraint < 1e-6:
            print("Phase B PASSES. Phase 4 succeeds where Phase 3 failed.")
        elif worst_rel < 0.1:
            print(
                "Phase B partial: W[C] within 10% but not 1%. Consider enabling "
                "supervised anchor or indirect MM equations."
            )
        else:
            print(
                "Phase B does not meet the gate. Log-check diagnostic "
                "residuals to diagnose."
            )
