"""CLI entry point for the Cuntz-Fock bootstrap (Phase 4 v2).

Usage:
    python3 -m cuntz_bootstrap.train --phase a
    python3 -m cuntz_bootstrap.train --phase b --l_poly 3 --l_trunc 3
    python3 -m cuntz_bootstrap.train --phase b --schedule 10.0,5.0,2.0 \\
                                     --weights mm=1,cyc=1,rp=1,sym=1

Dispatches to phase_a_gw / phase_b_qcd2 / phase_c_d3 / phase_d_d4 modules
with the requested configuration. See reference/cuntz_bootstrap.md for
physics details.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .phase_a_gw import run_phase_a
from .phase_b_qcd2 import run_phase_b


def _parse_schedule(s: str) -> tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


def _parse_weights(s: str) -> dict:
    out: dict[str, float] = {}
    for pair in s.split(","):
        if not pair.strip():
            continue
        k, v = pair.split("=")
        out[k.strip()] = float(v)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Cuntz-Fock bootstrap — Phase 4 v2 runner"
    )
    p.add_argument("--phase", choices=["a", "b", "c", "d"], required=True)
    p.add_argument("--l_poly", type=int, default=3)
    p.add_argument("--l_trunc", type=int, default=3)
    p.add_argument("--l_max_loops", type=int, default=4)
    p.add_argument("--schedule", type=str, default="10.0,5.0,2.0")
    p.add_argument("--n_steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--weights",
        type=str,
        default="mm=1,cyc=1,rp=1,sym=1",
        help="comma-separated key=value loss-weight spec",
    )
    p.add_argument("--output_dir", type=str, default="results")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    schedule = _parse_schedule(args.schedule)
    weights = _parse_weights(args.weights)

    if args.phase == "a":
        res = run_phase_a(
            L_trunc=args.l_trunc,
            seed=args.seed,
            output_dir=output_dir / "phase_a_v2",
            lams=schedule if schedule else (10.0, 5.0, 2.0, 1.0, 0.8, 0.5),
            n_steps=args.n_steps,
            lr=args.lr,
        )
    elif args.phase == "b":
        res = run_phase_b(
            L_poly=args.l_poly,
            L_trunc=args.l_trunc,
            L_max_loops=args.l_max_loops,
            lams=schedule,
            n_steps=args.n_steps,
            lr=args.lr,
            weights=weights,
            seed=args.seed,
            output_dir=output_dir / "phase_b_v2",
        )
    else:
        raise NotImplementedError(
            f"Phase {args.phase} not yet implemented. "
            "Phase c / d are stretch goals; start with a or b."
        )

    print(f"\nCompleted phase={args.phase}.")
    print(f"Results written to {output_dir}.")
    _ = res


if __name__ == "__main__":
    main()
