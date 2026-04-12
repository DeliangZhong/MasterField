#!/usr/bin/env python3
"""phase_b_mm.py — Phase B revisited with MM loop-equation loss.

Per Discussion-20 / R6 fix option 2, this script runs the same Phase B setup
(untwisted EK at D=2) but replaces the classical action S with the
Makeenko-Migdal residual loss from `mm_loss.py`.

Target comparison is the Gross-Witten exact answer for the 2D lattice:

    W[C] = w_+^{|Area(C)|}   with w_+ = 1/(2λ) (strong)  or  1 - λ/2 (weak)

So W[plaquette] = w_+, W[2×1] = w_+², W[2×2] = w_+⁴, etc. If MM loss on TEK
matrices recovers this, R6 is resolved at D=2 and we can push to twisted/C/D.
If only plaquette is captured (MM underdetermined), that reproduces Phase 1b's
finding and motivates adding positivity constraints next.
"""

from __future__ import annotations

import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

from mm_loss import optimize_tek_mm  # noqa: E402


LAM_SCHEDULE = [10.0, 5.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.5]
D = 2
L_MAX = 6

# Canonical keys for the loops we care about. These come from the LoopSystem
# enumeration (cyclic-smallest, backtrack-reduced). Discovered empirically —
# see mm_loss.compute_all_wilson_loops + loop_sys.loops.
KEY_PLAQUETTE = (-2, -1, 2, 1)
KEY_2X1 = (-2, -2, -1, 2, 2, 1)
KEY_2X2 = (-2, -2, -1, -1, 2, 2, 1, 1)


def gw_w_plus(lam: float) -> float:
    return 1.0 / (2.0 * lam) if lam >= 1.0 else 1.0 - lam / 2.0


def _run(N: int, ansatz: str, n_steps: int = 2000, lr: float = 0.02) -> list[dict]:
    records: list[dict] = []
    prev = None
    for lam in LAM_SCHEDULE:
        t0 = time.time()
        res = optimize_tek_mm(
            D=D, N=N, lam=lam, L_max=L_MAX,
            ansatz=ansatz, twist=False,
            init_params=prev, n_steps=n_steps, lr=lr,
            log_every=10000, verbose=False,
        )
        elapsed = time.time() - t0
        W = res.final_wilson_loops
        w_plus = gw_w_plus(lam)

        plaq = W.get(KEY_PLAQUETTE, float("nan"))
        r21 = W.get(KEY_2X1, float("nan"))
        r22 = W.get(KEY_2X2, float("nan"))

        gw_plaq = w_plus
        gw_21 = w_plus ** 2
        gw_22 = w_plus ** 4

        rec = {
            "lam": lam,
            "ansatz": ansatz,
            "N": N,
            "L_max": L_MAX,
            "W[plaq]": plaq,
            "W[2x1]": r21,
            "W[2x2]": r22,
            "GW w_plus": gw_plaq,
            "GW W[plaq]": gw_plaq,
            "GW W[2x1]": gw_21,
            "GW W[2x2]": gw_22,
            "err_plaq_rel": abs(plaq - gw_plaq) / abs(gw_plaq) if gw_plaq != 0 else None,
            "err_2x1_rel": abs(r21 - gw_21) / abs(gw_21) if gw_21 != 0 else None,
            "mm_loss": res.final_loss,
            "grad_norm": res.final_grad_norm,
            "converged": res.converged,
            "elapsed_s": elapsed,
        }
        records.append(rec)
        print(
            f"  {ansatz:<12s} N={N:3d} λ={lam:5.2f}  "
            f"W[plaq]={plaq:+.4f} (GW {gw_plaq:+.4f}, err {rec['err_plaq_rel'] * 100 if rec['err_plaq_rel'] is not None else 0:.1f}%)  "
            f"W[2×1]={r21:+.4f} (GW {gw_21:+.4f})  "
            f"mm_loss={res.final_loss:.2e}  "
            f"elapsed={elapsed:.1f}s"
        )
        prev = res.params
    return records


def main() -> int:
    print("=" * 78)
    print(f"  Phase B-MM — untwisted EK D={D}, MM-loop-equation loss, L_max={L_MAX}")
    print(f"  couplings: {LAM_SCHEDULE}")
    print("=" * 78)
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    all_records: list[dict] = []
    for (ansatz, N) in [("orientation", 9), ("full", 9)]:
        print(f"\n── ansatz={ansatz}, N={N} ──")
        all_records.extend(_run(N, ansatz, n_steps=2000, lr=0.02))

    summary = {
        "config": {"D": D, "L_max": L_MAX, "schedule": LAM_SCHEDULE, "twist": False},
        "records": all_records,
        "keys": {
            "KEY_PLAQUETTE": KEY_PLAQUETTE,
            "KEY_2X1": KEY_2X1,
            "KEY_2X2": KEY_2X2,
        },
    }
    out_path = os.path.join(out_dir, "phase_b_mm_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
