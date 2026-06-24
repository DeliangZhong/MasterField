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

from matrix_master_field.fock_jax import power_moments  # noqa: E402
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
    ansatz, fock_ops, g_target, *, max_word_len=4, w_sym=10.0,
    g_schedule=None, steps=4000, lr=5e-3, polish=True,
):
    """Solve the commutator+mass two-matrix master field at coupling g_target.

    g-homotopy from the exact g=0 free field upward; loss = SD residual +
    w_sym·(cyclicity+exchange+Z₂). Returns operators, params, and residuals.

    STATUS (M3): g=0 is exact and the confinement trend is correct, but the g>0
    solve is NOT yet validated — at g=1 its ⟨tr M0²⟩≈0.55 falls BELOW the rigorous
    bootstrap_two_matrix lower bound (~0.62 at L=6), i.e. it is under-converged
    (sd_loss ~1e-3, not machine zero). Landing inside the SDP island requires
    tighter optimization (more steps/restarts, possibly higher truncation); a
    high-budget attempt is expensive. Do NOT treat the g>0 output as the validated
    master field until it sits inside the bootstrap_two_matrix island.
    """
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
    return {
        "operators": [np.asarray(o) for o in ops],
        "params": params,
        "g": g_target,
        "sd_loss": float(two_matrix_sd_residual(ops, words, g_target)),
        "sym_loss": float(symmetry_losses(ops, words)),
    }
