"""Phase C D=3 stretch test (v2).

D=3, L_poly = L_trunc = 2 (d = 43 Fock space, n_labels=6). Three Hermitian
generators with ~510 real DOFs total. Unsupervised loss as in Phase B.

Compared to Kazakov-Zheng bootstrap bounds (arXiv:2203.11360) and published
MC at large N.

Run: python3 -m cuntz_bootstrap.phase_c_d3
Conditional on Phase B passing.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX
from .hermitian_operator import build_forward_link_ops, init_hermitian_params
from .mm_loss import _load_loop_system
from .optimize import optimize_cuntz
from .total_loss import make_total_loss_fn
from .wilson_loops import wilson_loop


def run_phase_c(
    L_poly: int = 2,
    L_trunc: int = 2,
    L_max_loops: int = 4,
    lams: tuple[float, ...] = (10.0, 5.0, 2.0),
    n_steps: int = 3000,
    lr: float = 1e-2,
    weights: dict | None = None,
    seed: int = 42,
    output_dir: Path = Path("results/phase_c_v2"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if weights is None:
        weights = {"mm": 1.0, "cyc": 1.0, "rp": 1.0, "sym": 1.0}

    loop_sys = _load_loop_system(D=3, L_max=L_max_loops)
    fock = CuntzFockJAX(n_labels=6, L_trunc=L_trunc)

    loss_fn = make_total_loss_fn(
        loop_sys=loop_sys, fock=fock, D=3,
        weights=weights, return_components=False,
    )
    loss_components_fn = make_total_loss_fn(
        loop_sys=loop_sys, fock=fock, D=3,
        weights=weights, return_components=True,
    )

    results: dict = {}
    p_warm = None
    for lam in lams:
        p0 = (
            p_warm
            if p_warm is not None
            else init_hermitian_params(
                n_matrices=3, fock=fock, seed=seed, scale=0.02
            )
        )
        res = optimize_cuntz(
            loss_fn=loss_fn, params0=p0, lam=lam,
            n_steps=n_steps, lr=lr, warmup=200, log_every=200, verbose=True,
        )
        p_warm = [jnp.array(c) for c in res.params]
        U_list = build_forward_link_ops(res.params, fock=fock)
        comps = loss_components_fn(res.params, lam)
        # Plaquette in (1,2) plane (no time axis here; use axes 1 and 2)
        W_plaq = float(
            jnp.real(wilson_loop(U_list, (1, 2, -1, -2), fock, D=3))
        )
        results[str(lam)] = {
            "lam": float(lam),
            "final_loss": float(res.final_loss),
            "L_MM": float(comps.L_MM),
            "L_cyc": float(comps.L_cyc),
            "L_RP": float(comps.L_RP),
            "L_sym": float(comps.L_sym),
            "W_plaq_12": W_plaq,
        }
        print(
            f"lam={lam}: W[plaq]={W_plaq:.5f}  L_MM={comps.L_MM:.3e}  "
            f"L_cyc={comps.L_cyc:.3e}"
        )

    (output_dir / "phase_c_v2_summary.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


if __name__ == "__main__":
    run_phase_c()
