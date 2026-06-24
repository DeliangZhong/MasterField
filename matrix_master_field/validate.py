"""Ansatz comparison study (Milestone 2, Task 7).

Compares the three operator ansätze on the solved quartic one-matrix model
(g=0.5) where the answer is known, scoring: interior-moment accuracy,
parameter count, and restart-robustness (the spurious-solution / basin metric).
Run as a script to print the table; numbers feed the results doc.
"""

import numpy as np
from jax import tree_util

from matrix_master_field import one_matrix as om
from matrix_master_field.amortized import AmortizedMonomial, train_amortized
from matrix_master_field.ansatz import DenseHermitianAnsatz, MonomialAnsatz
from matrix_master_field.fock_jax import FockOps, power_moments
from matrix_master_field.train import solve

G = 0.5
VPRIME = [0.0, 1.0, 0.0, G]
K = 12
INTERIOR = 9  # m_0..m_8
SUCCESS = 1e-2  # interior-error threshold counting as "correct basin"


def _interr(moments, target):
    return float(np.max(np.abs(moments[:INTERIOR] - target[:INTERIOR])))


def compare_ansatze(seeds=range(5)):
    ops = FockOps(1, 16)
    target = om.quartic_moments_from_sd(G, K)
    out = {}

    for name, make in [
        ("monomial(deg3)", lambda: MonomialAnsatz(ops, degree=3)),
        ("dense-hermitian", lambda: DenseHermitianAnsatz(ops)),
    ]:
        ans = make()
        errs = []
        for s in seeds:
            r = solve(ans, VPRIME, ops, K, n_restarts=1, steps=2500, seed=int(s))
            errs.append(_interr(r["moments"], target))
        errs = np.array(errs)
        out[name] = {
            "n_params": int(ans.n_params),
            "best_err": float(errs.min()),
            "median_err": float(np.median(errs)),
            "success_rate": float(np.mean(errs <= SUCCESS)),
        }

    # amortized: one network across the family, evaluated at g=0.5
    am = AmortizedMonomial(ops, degree=3, hidden=64)
    p, loss = train_amortized(am, lambda g: [0.0, 1.0, 0.0, g],
                              [0.1, 0.3, 0.5, 0.7, 0.9], K, steps=12000, seed=0)
    n_net = int(sum(x.size for x in tree_util.tree_leaves(p)))
    am_err = _interr(np.asarray(power_moments(am.build_operators(p, G)[0], K)), target)
    out["amortized(h64)"] = {"n_params": n_net, "best_err": am_err, "loss": float(loss)}
    return out


if __name__ == "__main__":
    res = compare_ansatze()
    print(f"\nQuartic g={G}, K={K}, interior m_0..m_{INTERIOR-1}, success<{SUCCESS}")
    print("-" * 72)
    for name, r in res.items():
        line = f"{name:18s} nparams={r['n_params']:4d}  best_err={r['best_err']:.2e}"
        if "success_rate" in r:
            line += f"  median_err={r['median_err']:.2e}  success={r['success_rate']:.0%}"
        print(line)
