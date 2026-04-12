#!/usr/bin/env python3
"""phase_b.py — Phase B experiment: untwisted Eguchi-Kawai at D=2.

Runs direct optimization of the untwisted EK action (z_μν = 1) with both
orientation and full U(N) ansätze across a coupling schedule, and measures:

  - final plaquette P = (1/N) Tr(U_1 U_2 U_1† U_2†)
  - Polyakov loops P_μ = (1/N) Tr(U_μ)
  - center-symmetry order |P_1|² + |P_2|²
  - eigenvalue phase density of U_1 (histogram, for uniformity check)

Expected MC behavior for untwisted EK D=2 at large N (Okawa 1982 and later):
  - strong coupling (λ large): plaquette → 0, eigenvalues uniform, |P_μ| → 0
  - weak coupling (λ small): plaquette → 1, center symmetry BREAKS, |P_μ| > 0

Expected direct-action-optimization behavior (our approach):
  - At H=0 or M=0: plaquette = 1 exactly (classical minimum of -Tr[U U U† U†]/N).
  - Loss -tek_loss is coupling-independent by construction; saddle = classical
    minimum regardless of λ. Optimizer should stay at plaquette ≈ 1 across
    the whole schedule.

The phase-B gate is either:
  (a) observe coupling-dependent plaquette (surprise; ansatz is finding something
      beyond the classical saddle, perhaps via the measure structure implicit in
      the gradient flow), OR
  (b) observe plaquette pinned at ~1 for every λ (expected; tells us the loss
      function needs a Haar-entropy term to reproduce MC — R6, a new risk).

Outcome is logged to results/phase_b_summary.json and printed.
"""

from __future__ import annotations

import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from observables import center_symmetry_order, eigenvalue_phases, polyakov_loop  # noqa: E402
from optimize import optimize_tek  # noqa: E402
from tek import build_clock_matrix, build_link_matrices, build_link_matrices_full  # noqa: E402


LAM_SCHEDULE = [20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.3]
D = 2
N = 49
L = 7
N_STEPS = 2000
LR = 0.01


def _observables(res) -> dict:
    """Compute observables from the optimizer result's params."""
    Gamma = build_clock_matrix(N)
    if res.ansatz == "orientation":
        U = build_link_matrices(res.params, Gamma)
    else:
        U = build_link_matrices_full(res.params)

    p1 = complex(polyakov_loop(U, mu=0))
    p2 = complex(polyakov_loop(U, mu=1))
    cs = float(center_symmetry_order(U))

    phases_U1 = [float(x) for x in np.asarray(eigenvalue_phases(U, mu=0))]
    counts, edges = np.histogram(phases_U1, bins=20, range=(-np.pi, np.pi))

    # Max deviation from uniform density (uniform would be N/20 per bin).
    uniform = N / 20.0
    max_dev_from_uniform = float(np.max(np.abs(counts - uniform))) / N

    return {
        "plaquette": res.final_plaquette,
        "loss": res.final_loss,
        "grad_norm": res.final_grad_norm,
        "converged": res.converged,
        "P1_re": p1.real, "P1_im": p1.imag, "|P1|": abs(p1),
        "P2_re": p2.real, "P2_im": p2.imag, "|P2|": abs(p2),
        "center_symmetry_order": cs,
        "eigenvalue_max_dev_from_uniform_U1": max_dev_from_uniform,
        "histogram_U1": counts.tolist(),
        "histogram_edges": edges.tolist(),
    }


def _run_one(ansatz: str) -> dict:
    """Run the coupling schedule with the given ansatz. Returns per-λ records."""
    records: list[dict] = []
    p_init = None  # use default random init on first stage

    for i, lam in enumerate(LAM_SCHEDULE):
        print(f"\n── λ = {lam:.4f}  (stage {i + 1}/{len(LAM_SCHEDULE)}, ansatz={ansatz}) ──")
        t0 = time.time()
        res = optimize_tek(
            D=D, N=N, lam=lam,
            n_steps=N_STEPS, lr=LR, warmup=200,
            k=1, twist=False,           # UNTWISTED EK — Phase B
            ansatz=ansatz,
            init_params=p_init,
            seed=42, verbose=False,
            log_every=200,
        )
        elapsed = time.time() - t0

        obs = _observables(res)
        obs["lam"] = lam
        obs["elapsed_s"] = elapsed
        records.append(obs)
        print(
            f"   plaq={obs['plaquette']:+.6f}  |grad|/N={obs['grad_norm']:.2e}  "
            f"|P1|={obs['|P1|']:.3f}  |P2|={obs['|P2|']:.3f}  "
            f"cs_order={obs['center_symmetry_order']:.3e}  "
            f"elapsed={elapsed:.1f}s"
        )

        p_init = res.params  # warm-start next λ

    return {"ansatz": ansatz, "records": records}


def main() -> int:
    print("=" * 78)
    print(f"  PHASE B — untwisted EK, D={D}, N={N} (L={L}), both ansätze")
    print(f"  schedule: {LAM_SCHEDULE}")
    print(f"  n_steps={N_STEPS}, lr={LR}")
    print("=" * 78)

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, dict] = {}
    for ansatz in ("orientation", "full"):
        print("\n" + "─" * 78)
        print(f"  ANSATZ: {ansatz}")
        print("─" * 78)
        results[ansatz] = _run_one(ansatz)

    summary = {
        "config": {
            "D": D, "N": N, "L": L,
            "n_steps": N_STEPS, "lr": LR,
            "schedule": LAM_SCHEDULE,
            "twist": False,
        },
        "results": results,
    }
    out_path = os.path.join(out_dir, "phase_b_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 78)
    print("  Phase B summary")
    print("=" * 78)
    print(f"{'ansatz':<14s} {'λ':>8s}  {'plaq':>10s}  {'|P1|':>6s}  {'|P2|':>6s}  "
          f"{'cs':>8s}  {'|grad|/N':>8s}")
    for ansatz, res in results.items():
        for rec in res["records"]:
            print(
                f"{ansatz:<14s} {rec['lam']:8.3f}  {rec['plaquette']:+10.6f}  "
                f"{rec['|P1|']:6.3f}  {rec['|P2|']:6.3f}  "
                f"{rec['center_symmetry_order']:8.2e}  {rec['grad_norm']:8.2e}"
            )

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
