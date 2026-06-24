"""Optimization engine: solve the exact nonlinear loop equations over an
operator ansatz. Positivity is automatic (Hermitian operators, vacuum state).

`solve` (one-matrix) and `solve_two_matrix` run multi-restart / g-homotopy Adam
then an L-BFGS polish on the exact nonlinear residual. This is the genuinely new
method: no SDP relaxation, no dropped positivity — minimize the exact nonlinear
residual over genuine states. The two-matrix objective adds cyclicity/exchange/Z₂
losses, because the Cuntz vacuum is not tracial.

NOTE on truncation: at finite cutoff the loop equations constrain only interior
moments; validate interior moments, and prefer model-appropriate cutoffs.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import scipy.optimize as sopt  # noqa: E402

from matrix_master_field.bootstrap_sdp import HAS_CVXPY, bootstrap_two_matrix  # noqa: E402
from matrix_master_field.fock_jax import power_moments, word_moment  # noqa: E402
from matrix_master_field.loss import (  # noqa: E402
    one_matrix_sd_residual,
    symmetry_losses,
    two_matrix_sd_residual,
    two_matrix_test_words,
)


def _lbfgs_polish(loss_fn, params, maxiter=3000):
    """L-BFGS-B polish of `loss_fn` from `params` (arbitrary pytree array shape)."""
    shape = params.shape
    vg = jax.jit(jax.value_and_grad(loss_fn))

    def fun_and_grad(x):
        loss, grads = vg(jnp.asarray(x).reshape(shape))
        return float(loss), np.asarray(grads, dtype=np.float64).ravel()

    r = sopt.minimize(
        fun_and_grad, np.asarray(params).ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14},
    )
    return jnp.asarray(r.x).reshape(shape)


def _adam_run(loss_fn, params, steps, lr):
    sched = optax.warmup_cosine_decay_schedule(lr * 0.01, lr, min(200, steps // 5), steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))

    def run(p):
        opt_state = opt.init(p)

        def body(carry, _):
            p, s = carry
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, s = opt.update(grads, s, p)
            return (optax.apply_updates(p, updates), s), loss

        (p, _), _ = jax.lax.scan(body, (p, opt_state), None, length=steps)
        return p

    return jax.jit(run)(params)


def solve(ansatz, v_prime, fock_ops, K, *, n_restarts=4, steps=3000, lr=1e-2, seed=0, polish=True):
    """Minimize the 1-matrix SD residual over `ansatz`; return the best run."""
    def loss_fn(params):
        return one_matrix_sd_residual(power_moments(ansatz.build_operators(params)[0], K), v_prime)

    key = jax.random.PRNGKey(seed)
    best = None
    for _ in range(n_restarts):
        key, sk = jax.random.split(key)
        p = _adam_run(loss_fn, ansatz.init_params(sk), steps, lr)
        if polish:
            p = _lbfgs_polish(loss_fn, p)
        fl = float(loss_fn(p))
        if best is None or fl < best["sd_loss"]:
            M = ansatz.build_operators(p)[0]
            best = {
                "moments": np.asarray(power_moments(M, K)),
                "sd_loss": fl, "params": p, "operator": np.asarray(M),
            }
    return best


def solve_two_matrix(
    ansatz, fock_ops, g_target, *, max_word_len=4, w_sym=10.0, g_schedule=None,
    steps=4000, lr=5e-3, polish=True, validate=True, target_word=(0, 0),
    sdp_word_len=6, sd_tol=1e-4, island_tol=1e-3,
):
    """Solve the commutator+mass two-matrix master field at coupling g_target.

    g-homotopy from the exact g=0 free field upward; loss = SD residual +
    w_sym·(cyclicity+exchange+Z₂).

    FAILS CLOSED. The returned dict carries `validated`: True ONLY if the residual
    is below `sd_tol` AND the target moment lies inside the rigorous
    bootstrap_two_matrix island. A low residual alone is NOT sufficient (it can be
    a spurious or truncation-contaminated state); callers MUST check `validated`.
    At g=0 the free field is exact (validated True); at g>0 the current solve is
    typically under-converged and returns validated=False (see `validation`).
    """
    # Truncation guard: the SD residual evaluates commutator words of length
    # |w|+3, so the Fock cutoff must at least represent them. (Necessary, not
    # sufficient — full adequacy is what the SDP-island check below verifies.)
    need = max_word_len + 3
    if fock_ops.max_length < need:
        raise ValueError(
            f"Fock cutoff max_length={fock_ops.max_length} too small for "
            f"max_word_len={max_word_len}: need >= {need} (commutator words have "
            f"length |w|+3). Increase the Fock cutoff or lower max_word_len."
        )

    words = two_matrix_test_words(max_word_len)
    if g_schedule is None:
        g_schedule = [g_target * t for t in (0.2, 0.4, 0.6, 0.8, 1.0)]

    def make_loss(g):
        def loss_fn(params):
            ops = ansatz.build_operators(params)
            return two_matrix_sd_residual(ops, words, g) + w_sym * symmetry_losses(ops, words)
        return loss_fn

    params = ansatz.params_for_free_field()  # exact g=0 solution as warm start
    for g in g_schedule:
        loss_fn = make_loss(g)
        params = _adam_run(loss_fn, params, steps, lr)
        if polish:
            params = _lbfgs_polish(loss_fn, params)

    ops = ansatz.build_operators(params)
    sd_loss = float(two_matrix_sd_residual(ops, words, g_target))
    result = {
        "operators": [np.asarray(o) for o in ops],
        "params": params,
        "g": g_target,
        "sd_loss": sd_loss,
        "sym_loss": float(symmetry_losses(ops, words)),
    }

    if validate:
        tr_target = float(word_moment(ops, target_word))
        sd_ok = sd_loss < sd_tol
        lb = ub = in_island = None
        if HAS_CVXPY:
            lb = bootstrap_two_matrix(g_target, max_word_len=sdp_word_len,
                                      target_word=target_word, maximize=False)
            ub = bootstrap_two_matrix(g_target, max_word_len=sdp_word_len,
                                      target_word=target_word, maximize=True)
            if lb is not None and ub is not None:
                in_island = (lb - island_tol) <= tr_target <= (ub + island_tol)
        result["validation"] = {
            "target_word": target_word, "target_value": tr_target,
            "sd_ok": sd_ok, "island": (lb, ub), "in_island": in_island,
        }
        result["validated"] = bool(sd_ok and in_island is True)
    return result
