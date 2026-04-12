"""Adam + warmup_cosine optimizer for the Cuntz-Fock coefficient bootstrap.

Mirrors tek_master_field/optimize.py. The critical detail is the Impl-19
JAX complex-gradient fix: `jax.grad` on a real-valued loss of complex
parameters returns (∂f/∂x − i ∂f/∂y), which is the CONJUGATE of the
physical descent direction. We conjugate the gradient before passing to
optax.

See discussion_AI.md Implementation-19 for the derivation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import optax


@dataclass
class OptResult:
    params: list[jnp.ndarray]
    losses: list[float]
    final_loss: float
    grad_norm: float
    n_steps_run: int


def _build_schedule(lr: float, n_steps: int, warmup: int) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=max(1, warmup),
        decay_steps=max(1, n_steps),
        end_value=lr * 0.01,
    )


def _build_optimizer(
    lr: float, n_steps: int, warmup: int, grad_clip: float
) -> optax.GradientTransformation:
    sched = _build_schedule(lr, n_steps, warmup)
    return optax.chain(optax.clip_by_global_norm(grad_clip), optax.adam(sched))


def _grad_global_norm(grads: list[jnp.ndarray]) -> float:
    return float(
        jnp.sqrt(sum(jnp.sum(jnp.abs(g) ** 2) for g in grads))
    )


def optimize_cuntz(
    loss_fn: Callable[[list[jnp.ndarray], float], jnp.ndarray],
    params0: list[jnp.ndarray],
    lam: float,
    n_steps: int = 3000,
    lr: float = 1e-2,
    warmup: int = 200,
    grad_clip: float = 1.0,
    tol: float = 1e-12,
    log_every: int = 100,
    verbose: bool = True,
) -> OptResult:
    """Gradient-descent optimiser for a complex-coefficient loss.

    Returns OptResult with the final parameter vectors, logged losses, final
    loss value, final global gradient norm, and the number of steps actually
    run (may be < n_steps if tol is reached early).
    """
    optimizer = _build_optimizer(lr, n_steps, warmup, grad_clip)
    params: list[jnp.ndarray] = [jnp.array(c) for c in params0]
    state = optimizer.init(params)

    grad_fn = jax.grad(lambda ps: loss_fn(ps, lam))

    def step(
        ps: list[jnp.ndarray], st
    ) -> tuple[list[jnp.ndarray], object, list[jnp.ndarray]]:
        grads = grad_fn(ps)
        # Impl-19 JAX complex-gradient fix: conjugate before optax.
        grads = [jnp.conj(g) for g in grads]
        updates, new_st = optimizer.update(grads, st, ps)
        new_ps = optax.apply_updates(ps, updates)
        return new_ps, new_st, grads

    losses: list[float] = []
    grads_last: list[jnp.ndarray] | None = None
    it = 0
    for it in range(n_steps):
        params, state, grads_last = step(params, state)
        if it % log_every == 0 or it == n_steps - 1:
            L = float(loss_fn(params, lam))
            losses.append(L)
            if verbose:
                gn = _grad_global_norm(grads_last)
                print(f"step {it:6d}  L = {L:.6e}  |grad| = {gn:.3e}")
            if L < tol:
                break

    final_loss = losses[-1] if losses else float("nan")
    grad_norm = _grad_global_norm(grads_last) if grads_last else 0.0
    return OptResult(
        params=params,
        losses=losses,
        final_loss=final_loss,
        grad_norm=grad_norm,
        n_steps_run=it + 1,
    )
