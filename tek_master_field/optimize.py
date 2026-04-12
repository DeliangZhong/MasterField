"""optimize.py — Gradient-descent optimization of the TEK action.

Minimizes `tek.tek_loss` over the list of Hermitian generators [H_2, …, H_D].
Uses Adam with a warmup + cosine decay schedule (copied from the master_field
neural trainer pattern). After each gradient step, each H is re-Hermitianized
to prevent drift due to floating-point arithmetic.

Convergence criterion: max_μ ||∂L / ∂H_μ||_F / N < 1e-8.

Coupling continuation: pass a decreasing schedule of λ values; the solution at
each λ warm-starts the next. This is how we traverse strong → weak coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax import grad, jit, random  # noqa: E402

from tek import (  # noqa: E402
    build_clock_matrix,
    build_twist,
    hermitianize,
    init_H_list_random,
    plaquette_average,
    tek_loss,
)


@dataclass
class OptResult:
    """Final state of an optimization run."""

    H_list: list[jnp.ndarray]
    history: dict[str, list[float]]
    final_loss: float
    final_plaquette: float
    final_grad_norm: float
    converged: bool
    D: int
    N: int
    lam: float


def _make_loss_and_grad(
    Gamma: jnp.ndarray,
    z: jnp.ndarray,
    D: int,
) -> tuple[Callable, Callable]:
    """Build JIT-compiled loss and gradient functions bound to Gamma, z, D."""

    def _loss(Hs: list[jnp.ndarray]) -> jnp.ndarray:
        return tek_loss(Hs, Gamma, z, D)

    loss_fn = jit(_loss)
    grad_fn = jit(grad(_loss))
    return loss_fn, grad_fn


def _build_schedule(lr: float, n_steps: int, warmup: int) -> optax.Schedule:
    return optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=min(warmup, max(1, n_steps // 5)),
        decay_steps=n_steps,
        end_value=lr * 0.01,
    )


def _build_optimizer(lr: float, n_steps: int, warmup: int, grad_clip: float) -> optax.GradientTransformation:
    sched = _build_schedule(lr, n_steps, warmup)
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(sched),
    )


def optimize_tek(
    D: int,
    N: int,
    lam: float,
    n_steps: int = 3000,
    lr: float = 0.01,
    warmup: int = 200,
    grad_clip: float = 1.0,
    init_H_list: list[jnp.ndarray] | None = None,
    k: int = 1,
    twist: bool = True,
    log_every: int = 100,
    tol: float = 1e-8,
    seed: int = 42,
    verbose: bool = True,
) -> OptResult:
    """Optimize the TEK action by gradient descent.

    Args:
        D: spacetime dimension (2, 3, or 4).
        N: matrix size. Must satisfy N = L² with L prime for TEK.
        lam: 't Hooft coupling (used for logging only — see tek.tek_loss docstring).
        n_steps: number of Adam steps.
        lr: peak learning rate.
        warmup: warmup steps (clipped to n_steps // 5 maximum).
        grad_clip: global gradient norm clip.
        init_H_list: initial H list; if None, use random with small scale (to
            break the zero-gradient symmetry at H=0).
        k: twist flux integer (1 = symmetric twist; D=4 recommends k > 1).
        twist: if False, sets z_μν = 1 (untwisted EK — for Phase B).
        log_every: how often to record (loss, plaquette, grad_norm).
        tol: convergence tolerance on max_μ ||∂L/∂H_μ||_F / N.
        seed: PRNG seed for random init.

    Returns an OptResult dataclass.
    """
    L = int(round(N**0.5))
    if L * L != N:
        raise ValueError(f"N must be a perfect square; got N={N}")

    Gamma = build_clock_matrix(N)
    if twist:
        z = build_twist(D, N, L, k=k)
    else:
        z = jnp.ones((D, D), dtype=jnp.complex128)

    if init_H_list is None:
        key = random.PRNGKey(seed)
        init_H_list = init_H_list_random(D, N, key, scale=0.01)
    # Defensive: re-Hermitianize.
    H_list = [hermitianize(H) for H in init_H_list]

    loss_fn, grad_fn = _make_loss_and_grad(Gamma, z, D)

    optimizer = _build_optimizer(lr, n_steps, warmup, grad_clip)
    opt_state = optimizer.init(H_list)

    @partial(jit, static_argnums=())
    def _step(Hs: list[jnp.ndarray], state):
        grads = grad_fn(Hs)
        updates, new_state = optimizer.update(grads, state, Hs)
        new_Hs = optax.apply_updates(Hs, updates)
        new_Hs = [hermitianize(H) for H in new_Hs]
        return new_Hs, new_state, grads

    history: dict[str, list[float]] = {"step": [], "loss": [], "plaq": [], "grad_norm": []}
    converged = False
    final_grad_norm = float("inf")

    for step in range(n_steps):
        H_list, opt_state, grads = _step(H_list, opt_state)

        if (step % log_every == 0) or (step == n_steps - 1):
            loss_val = float(loss_fn(H_list))
            plaq_val = float(plaquette_average(H_list, Gamma, z, D))
            gnorm = max(float(jnp.linalg.norm(g)) for g in grads) / N
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["plaq"].append(plaq_val)
            history["grad_norm"].append(gnorm)
            final_grad_norm = gnorm
            if verbose:
                print(
                    f"  step {step:6d}  loss={loss_val:+.8f}  "
                    f"plaq={plaq_val:+.6f}  |grad|/N={gnorm:.2e}"
                )
            if gnorm < tol:
                converged = True
                if verbose:
                    print(f"  converged at step {step} (|grad|/N < {tol:.0e})")
                break

    final_loss = float(loss_fn(H_list))
    final_plaquette = float(plaquette_average(H_list, Gamma, z, D))

    return OptResult(
        H_list=H_list,
        history=history,
        final_loss=final_loss,
        final_plaquette=final_plaquette,
        final_grad_norm=final_grad_norm,
        converged=converged,
        D=D,
        N=N,
        lam=lam,
    )


def coupling_continuation(
    D: int,
    N: int,
    lam_schedule: list[float],
    n_steps_per: int = 1500,
    lr: float = 0.01,
    k: int = 1,
    twist: bool = True,
    seed: int = 42,
    verbose: bool = True,
) -> dict[float, OptResult]:
    """Solve TEK across a decreasing schedule of couplings, warm-starting.

    Order lam_schedule from large (strong coupling, near disordered saddle)
    to small (weak coupling, ordered saddle) for the best warm-start.
    """
    results: dict[float, OptResult] = {}

    # Initial H list: small random
    key = random.PRNGKey(seed)
    H_init: list[jnp.ndarray] | None = init_H_list_random(D, N, key, scale=0.01)

    for i, lam in enumerate(lam_schedule):
        if verbose:
            print(f"\n── λ = {lam:.4f}  (stage {i + 1}/{len(lam_schedule)}) ──")
        res = optimize_tek(
            D=D,
            N=N,
            lam=lam,
            n_steps=n_steps_per,
            lr=lr,
            init_H_list=H_init,
            k=k,
            twist=twist,
            seed=seed,
            verbose=verbose,
        )
        results[lam] = res
        H_init = res.H_list  # warm-start next stage

    return results


# Default coupling schedule (strong → weak)
DEFAULT_LAM_SCHEDULE: list[float] = [
    20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0,
    1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3,
]
