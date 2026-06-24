"""Optimization engine: solve the exact nonlinear loop equations over an
operator ansatz. Positivity is automatic (Hermitian operators, vacuum state).

`solve` runs multi-restart Adam (warmup-cosine) to reach the right basin, then
an L-BFGS polish (jax gradient → scipy) to tighten the residual to ~1e-9. This
is the genuinely new method: no SDP relaxation, no dropped positivity — minimize
the exact nonlinear residual over genuine states.

NOTE on truncation: at finite moment cutoff K the loop equations constrain only
the *interior* moments; the top few (truncation-edge) moments are unconstrained
and should not be used to validate the solution. Choose K so the moments of
interest sit in the interior. Catalan-type (growing) moments also make large-K
optimization harder, so use a model-appropriate K.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import scipy.optimize as sopt  # noqa: E402

from matrix_master_field.fock_jax import power_moments  # noqa: E402
from matrix_master_field.loss import one_matrix_sd_residual  # noqa: E402


def solve(ansatz, v_prime, fock_ops, K, *, n_restarts=4, steps=3000, lr=1e-2, seed=0, polish=True):
    """Minimize the 1-matrix SD residual over `ansatz`; return the best run.

    Returns dict: moments (np [K+1]), sd_loss (float), params, operator (np D×D).
    """
    def loss_fn(params):
        M = ansatz.build_operators(params)[0]
        return one_matrix_sd_residual(power_moments(M, K), v_prime)

    sched = optax.warmup_cosine_decay_schedule(lr * 0.01, lr, min(200, steps // 5), steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))

    @jax.jit
    def adam_run(params):
        opt_state = opt.init(params)

        def body(carry, _):
            p, s = carry
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, s = opt.update(grads, s, p)
            return (optax.apply_updates(p, updates), s), loss

        (p, _), _ = jax.lax.scan(body, (params, opt_state), None, length=steps)
        return p

    vg = jax.jit(jax.value_and_grad(loss_fn))

    def fun_and_grad(x):
        loss, grads = vg(jnp.asarray(x))
        return float(loss), np.asarray(grads, dtype=np.float64)

    key = jax.random.PRNGKey(seed)
    best = None
    for _ in range(n_restarts):
        key, sk = jax.random.split(key)
        p = adam_run(ansatz.init_params(sk))
        if polish:
            r = sopt.minimize(
                fun_and_grad, np.asarray(p), jac=True, method="L-BFGS-B",
                options={"maxiter": 3000, "ftol": 1e-16, "gtol": 1e-14},
            )
            p = jnp.asarray(r.x)
        fl = float(loss_fn(p))
        if best is None or fl < best["sd_loss"]:
            M = ansatz.build_operators(p)[0]
            best = {
                "moments": np.asarray(power_moments(M, K)),
                "sd_loss": fl,
                "params": p,
                "operator": np.asarray(M),
            }
    return best
