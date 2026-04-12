#!/usr/bin/env python3
"""train.py — CLI entry point for TEK master-field training.

Examples:
    # Phase A (Gross-Witten infrastructure check)
    python3 train.py --model gw --lam 1.0 --validate

    # Phase B (untwisted EK in D=2)
    python3 train.py --model ek --D 2 --N 49 --schedule default

    # Phase C (D=2 TEK at a single λ)
    python3 train.py --model tek --D 2 --N 49 --lam 1.0 --n_steps 3000

    # Phase C (D=2 TEK coupling continuation)
    python3 train.py --model tek --D 2 --N 49 --schedule default --n_steps 1500

    # Phase D (D=4 TEK; flag R1 — pick k via --k carefully)
    python3 train.py --model tek --D 4 --N 49 --k 1 --lam 1.0 --n_steps 3000
"""

from __future__ import annotations

import argparse
import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

DEFAULT_SCHEDULE = [20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0,
                    1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TEK master field — direct optimization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=["gw", "ek", "tek"], default="tek",
                   help="gw: Gross-Witten (1-matrix, Phase A). "
                        "ek: untwisted Eguchi-Kawai (Phase B). "
                        "tek: Twisted Eguchi-Kawai (Phase C/D).")
    p.add_argument("--D", type=int, default=2, choices=[2, 3, 4], help="Spacetime dimension")
    p.add_argument("--N", type=int, default=49, help="Matrix size (must be L² with L prime)")
    p.add_argument("--k", type=int, default=1, help="Twist flux integer (n_μν = k·L on twisted planes)")
    p.add_argument("--lam", type=float, default=1.0, help="'t Hooft coupling (single-λ mode)")
    p.add_argument("--schedule", type=str, default=None,
                   help="Comma-separated λ schedule (strong→weak); 'default' uses built-in")
    p.add_argument("--n_steps", type=int, default=3000, help="Adam steps (per λ if continuation)")
    p.add_argument("--lr", type=float, default=1e-2, help="Peak learning rate")
    p.add_argument("--warmup", type=int, default=200, help="Warmup steps (cosine decay)")
    p.add_argument("--seed", type=int, default=42, help="PRNG seed")
    p.add_argument("--output_dir", type=str, default="results", help="Output directory")
    p.add_argument("--validate", action="store_true", help="Compare with exact/benchmark where possible")
    p.add_argument("--quiet", action="store_true", help="Suppress per-step logging")
    return p.parse_args()


def _parse_schedule(s: str | None, lam: float) -> list[float]:
    if s is None:
        return [lam]
    if s == "default":
        return DEFAULT_SCHEDULE
    return [float(x) for x in s.split(",")]


def _check_L_prime(N: int) -> int:
    L = int(round(N ** 0.5))
    if L * L != N:
        raise SystemExit(f"N must be a perfect square; got N={N}")
    # Not enforcing L prime (TEK's strict requirement) — we allow non-prime L for
    # exploratory runs. The user is responsible. Warn if composite.
    if L > 1:
        from math import isqrt
        is_prime = L > 1 and all(L % d != 0 for d in range(2, isqrt(L) + 1))
        if not is_prime:
            print(f"  [warn] L={L} is not prime — TEK symmetric-twist construction requires prime L.")
    return L


def _run_gw(args: argparse.Namespace) -> int:
    from gross_witten import phase_a_main, solve_gw

    if args.schedule is not None:
        tees = _parse_schedule(args.schedule, args.lam)
        results = [solve_gw(t, validate=args.validate) for t in tees]
        for r in results:
            print(f"  t={r.t:.3f}  {r.phase:6s}  w1={r.w1:.8f} (exact {r.w1_exact:.8f}, err {r.err_w1:.2e})")
    else:
        phase_a_main([args.lam])
    return 0


def _run_tek_or_ek(args: argparse.Namespace, twist: bool) -> int:
    from optimize import coupling_continuation, optimize_tek

    L = _check_L_prime(args.N)
    schedule = _parse_schedule(args.schedule, args.lam)

    print("=" * 70)
    print(f"  {'TEK' if twist else 'EK (untwisted)'} — D={args.D}, N={args.N}, L={L}, k={args.k}")
    print(f"  schedule: {schedule}")
    print(f"  opt: n_steps={args.n_steps}, lr={args.lr}, warmup={args.warmup}")
    print("=" * 70)

    t0 = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    if len(schedule) == 1:
        res = optimize_tek(
            D=args.D, N=args.N, lam=schedule[0],
            n_steps=args.n_steps, lr=args.lr, warmup=args.warmup,
            k=args.k, twist=twist, seed=args.seed,
            verbose=not args.quiet,
        )
        tag = f"{'tek' if twist else 'ek'}_D{args.D}_N{args.N}_k{args.k}_lam{schedule[0]:.4f}"
        _save_result(res, args.output_dir, tag)
        print(f"\nfinal plaq={res.final_plaquette:.8f}  |grad|/N={res.final_grad_norm:.2e}  "
              f"converged={res.converged}")
    else:
        results = coupling_continuation(
            D=args.D, N=args.N, lam_schedule=schedule,
            n_steps_per=args.n_steps, lr=args.lr, k=args.k,
            twist=twist, seed=args.seed, verbose=not args.quiet,
        )
        summary = [(lam, r.final_plaquette, r.final_grad_norm, r.converged) for lam, r in results.items()]
        tag = f"{'tek' if twist else 'ek'}_D{args.D}_N{args.N}_k{args.k}_continuation"
        with open(os.path.join(args.output_dir, f"{tag}.json"), "w") as f:
            json.dump({
                "D": args.D, "N": args.N, "k": args.k, "twist": twist,
                "schedule": schedule,
                "plaquette_by_lam": [(float(lam), float(plaq)) for (lam, plaq, _, _) in summary],
                "grad_norm_by_lam": [(float(lam), float(g)) for (lam, _, g, _) in summary],
                "converged_by_lam": [(float(lam), bool(c)) for (lam, _, _, c) in summary],
            }, f, indent=2)
        # Save final H for each λ
        for lam, res in results.items():
            for mu, H in enumerate(res.H_list):
                np.save(
                    os.path.join(args.output_dir, f"{tag}_H{mu + 2}_lam{lam:.4f}.npy"),
                    np.asarray(H),
                )

        print("\n  λ          plaq          |grad|/N    converged")
        for lam, plaq, gnorm, conv in summary:
            print(f"  {lam:8.4f}   {plaq:+.8f}   {gnorm:.2e}   {conv}")

    print(f"\nelapsed: {time.time() - t0:.1f} s")
    return 0


def _save_result(res, output_dir: str, tag: str) -> None:
    with open(os.path.join(output_dir, f"{tag}.json"), "w") as f:
        json.dump({
            "D": res.D, "N": res.N, "lam": res.lam,
            "final_loss": res.final_loss,
            "final_plaquette": res.final_plaquette,
            "final_grad_norm": res.final_grad_norm,
            "converged": res.converged,
            "history": res.history,
        }, f, indent=2)
    for mu, H in enumerate(res.H_list):
        np.save(os.path.join(output_dir, f"{tag}_H{mu + 2}.npy"), np.asarray(H))


def main() -> int:
    args = _parse_args()
    if args.model == "gw":
        return _run_gw(args)
    if args.model == "ek":
        return _run_tek_or_ek(args, twist=False)
    return _run_tek_or_ek(args, twist=True)


if __name__ == "__main__":
    raise SystemExit(main())
