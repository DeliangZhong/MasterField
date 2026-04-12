"""Phase A Gross-Witten gate (v2: exp-Hermitian ansatz).

Scan lambda in {10, 5, 2, 1, 0.8, 0.5}. For each lambda, train a single
master unitary operator Uhat on D=1 Cuntz-Fock (L_trunc = 6) against a
supervised moment-matching loss. Unitarity is automatic from the exp-
Hermitian parametrization (Uhat = expm(i * Hhat) with Hhat Hermitian).

    L_sup(h, lam) = sum_{k=1..L_trunc} |Re<Omega| Uhat^k |Omega> - w_k^GW|^2

Run:   python3 -m cuntz_bootstrap.phase_a_gw
Gate:  max |w_k - w_k^exact| < 1e-2 for all tested lambda >= 1,
       k in {1, 2}. Weak-coupling (lambda < 1) is informational only.
"""
from __future__ import annotations

import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .fock import CuntzFockJAX
from .gw_validation import gw_moments
from .hermitian_operator import (
    assemble_unitary,
    init_hermitian_params,
)
from .optimize import optimize_cuntz


def _phase_a_loss_fn_builder(fock: CuntzFockJAX, w_exact, K: int):
    """Closure suitable for optimize_cuntz: loss = sum_k (w_k - w_k^GW)^2."""
    targets = [float(w_exact[k]) for k in range(K + 1)]

    def loss_fn(params, lam_):
        U = assemble_unitary(params[0], fock)
        v = fock.vacuum_state()
        L_sup = jnp.zeros((), dtype=jnp.float64)
        for k in range(1, K + 1):
            v = U @ v
            wk = jnp.real(v[0])
            L_sup = L_sup + (wk - targets[k]) ** 2
        return L_sup

    return loss_fn


def run_phase_a(
    L_trunc: int = 6,
    seed: int = 0,
    output_dir: Path = Path("results/phase_a_v2"),
    lams: tuple[float, ...] = (10.0, 5.0, 2.0, 1.0, 0.8, 0.5),
    n_steps: int = 5000,
    lr: float = 5e-3,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    fock = CuntzFockJAX(n_labels=1, L_trunc=L_trunc)
    results: dict = {}

    for lam in lams:
        w_exact = gw_moments(lam=lam, K=L_trunc)
        loss_fn = _phase_a_loss_fn_builder(fock, w_exact, K=L_trunc)
        p0 = init_hermitian_params(
            n_matrices=1, fock=fock, seed=seed, scale=0.05
        )
        res = optimize_cuntz(
            loss_fn=loss_fn, params0=p0, lam=lam,
            n_steps=n_steps, lr=lr, warmup=200, log_every=500, verbose=True,
        )

        U = assemble_unitary(res.params[0], fock)
        I = jnp.eye(fock.dim, dtype=jnp.complex128)
        unit_err = float(
            jnp.sqrt(jnp.sum(jnp.abs(U @ U.conj().T - I) ** 2))
        )
        v = fock.vacuum_state()
        w_got = [1.0]
        for k in range(1, L_trunc + 1):
            v = U @ v
            w_got.append(float(jnp.real(v[0])))

        moment_errs = [
            abs(w_got[k] - float(w_exact[k])) for k in range(L_trunc + 1)
        ]
        results[str(lam)] = {
            "lam": float(lam),
            "final_loss": float(res.final_loss),
            "unitarity_err": unit_err,
            "w_exact": [float(x) for x in w_exact.tolist()],
            "w_got": w_got,
            "moment_errs": moment_errs,
            "max_moment_err": float(max(moment_errs)),
        }
        print(
            f"lam={lam}: final_loss={res.final_loss:.3e}  "
            f"unit_err={unit_err:.3e}  max_moment_err={max(moment_errs):.3e}"
        )

    (output_dir / "phase_a_v2_summary.json").write_text(
        json.dumps(results, indent=2)
    )
    return results


if __name__ == "__main__":
    res = run_phase_a()

    strong = {k: r for k, r in res.items() if r["lam"] >= 1.0}
    weak = {k: r for k, r in res.items() if r["lam"] < 1.0}

    worst_strong_moment = max(r["max_moment_err"] for r in strong.values())
    worst_strong_unit = max(r["unitarity_err"] for r in strong.values())

    print()
    print("=== Phase A gate (strong coupling, lam >= 1) ===")
    print(
        f"  worst moment error    = {worst_strong_moment:.3e}  (gate: < 1e-2)"
    )
    print(
        f"  worst unitarity error = {worst_strong_unit:.3e}  (expected ~1e-10)"
    )
    if worst_strong_moment < 1e-2 and worst_strong_unit < 1e-6:
        print("Phase A strong-coupling PASSES.")
    else:
        print("Phase A strong-coupling does NOT meet the gate.")

    if weak:
        worst_weak_moment = max(r["max_moment_err"] for r in weak.values())
        print()
        print("=== Phase A weak coupling (lam < 1) — informational only ===")
        print(f"  worst moment error    = {worst_weak_moment:.3e}")
        print(
            "  note: weak-coupling GW moments have qualitatively different "
            "structure (gapped eigenvalue density); single-mode L_trunc=6 "
            "ansatz not expected to capture this region. "
            "Excluded from the gate per plan."
        )
