"""optimize.py — Gradient-descent optimization of the TEK action.

Supports two ansätze for the link matrices:

    ansatz="orientation"  (default)
        U_1 = Γ (clock-matrix, kron(P_L, I_L))
        U_μ = Ω_μ Γ Ω_μ† for μ ≥ 2, Ω_μ = expm(i H_μ), H_μ Hermitian.
        Parameters: (D − 1) Hermitian N×N matrices.
        Eigenvalues of every U_μ are locked to the L-th roots of unity (L-fold).
        Cannot break center symmetry.

    ansatz="full"
        U_μ = expm(i M_μ)  for μ = 1, …, D, M_μ Hermitian.
        Parameters: D Hermitian N×N matrices.
        Eigenvalues are free. Can break center symmetry if the saddle requires.
        No gauge fixing — a flat N²−1-dim gauge direction is tolerated.

Both ansätze minimize the same `_plaquette_traces`-based loss with the
R3-fixed sign convention. After each gradient step we re-Hermitianize the
parameters to prevent drift.

Convergence criterion: max_μ ||∂L / ∂params_μ||_F / N < tol.

Coupling continuation: pass a decreasing schedule of λ values; the solution at
each λ warm-starts the next.
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
    init_M_list_random,
    plaquette_average,
    plaquette_average_full,
    tek_loss,
    tek_loss_full,
)


@dataclass
class OptResult:
    """Final state of an optimization run."""

    params: list[jnp.ndarray]  # H_list (orientation) or M_list (full)
    ansatz: str                # "orientation" or "full"
    history: dict[str, list[float]]
    final_loss: float
    final_plaquette: float
    final_grad_norm: float
    converged: bool
    D: int
    N: int
    lam: float

    @property
    def H_list(self) -> list[jnp.ndarray]:
        """Backward-compat alias for orientation ansatz."""
        return self.params

    @property
    def M_list(self) -> list[jnp.ndarray]:
        """Full-ansatz alias."""
        return self.params


def _make_loss_and_grad_orientation(
    Gamma: jnp.ndarray,
    z: jnp.ndarray,
    D: int,
) -> tuple[Callable, Callable, Callable]:
    """Build JIT loss, grad, and plaquette fns for the orientation ansatz."""

    def _loss(Hs: list[jnp.ndarray]) -> jnp.ndarray:
        return tek_loss(Hs, Gamma, z, D)

    def _plaq(Hs: list[jnp.ndarray]) -> jnp.ndarray:
        return plaquette_average(Hs, Gamma, z, D)

    return jit(_loss), jit(grad(_loss)), jit(_plaq)


def _make_loss_and_grad_full(
    z: jnp.ndarray,
    D: int,
) -> tuple[Callable, Callable, Callable]:
    """Build JIT loss, grad, and plaquette fns for the full U(N) ansatz."""

    def _loss(Ms: list[jnp.ndarray]) -> jnp.ndarray:
        return tek_loss_full(Ms, z, D)

    def _plaq(Ms: list[jnp.ndarray]) -> jnp.ndarray:
        return plaquette_average_full(Ms, z, D)

    return jit(_loss), jit(grad(_loss)), jit(_plaq)


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


def _default_init_params(
    ansatz: str, D: int, N: int, seed: int, scale: float = 0.01
) -> list[jnp.ndarray]:
    key = random.PRNGKey(seed)
    if ansatz == "orientation":
        return init_H_list_random(D, N, key, scale=scale)
    if ansatz == "full":
        return init_M_list_random(D, N, key, scale=scale)
    raise ValueError(f"Unknown ansatz: {ansatz!r}")


def optimize_tek(
    D: int,
    N: int,
    lam: float,
    n_steps: int = 3000,
    lr: float = 0.01,
    warmup: int = 200,
    grad_clip: float = 1.0,
    init_params: list[jnp.ndarray] | None = None,
    k: int = 1,
    twist: bool = True,
    ansatz: str = "orientation",
    log_every: int = 100,
    tol: float = 1e-8,
    seed: int = 42,
    verbose: bool = True,
) -> OptResult:
    """Optimize the TEK action by gradient descent.

    Args:
        D: spacetime dimension (2, 3, or 4).
        N: matrix size (N = L², L should be prime for TEK).
        lam: 't Hooft coupling (for logging only — loss is λ-independent).
        n_steps: number of Adam steps.
        lr: peak learning rate.
        warmup: warmup steps (clipped to n_steps // 5 maximum).
        grad_clip: global gradient-norm clip.
        init_params: initial parameter list.
            - If ansatz="orientation": (D−1) Hermitian N×N matrices
            - If ansatz="full":         D Hermitian N×N matrices
            If None, random Hermitian with small scale.
        k: twist flux integer (1 = symmetric twist; D=4 recommends k > 1).
        twist: if False, z_μν = 1 (untwisted EK — Phase B).
        ansatz: "orientation" or "full". See module docstring.
        log_every: how often to record (loss, plaquette, grad_norm).
        tol: convergence tolerance on max_μ ||∂L/∂params_μ||_F / N.
        seed: PRNG seed for random init.

    Returns an OptResult dataclass. `res.params` is H_list (orientation) or
    M_list (full).
    """
    L = int(round(N**0.5))
    if L * L != N:
        raise ValueError(f"N must be a perfect square; got N={N}")
    if ansatz not in ("orientation", "full"):
        raise ValueError(f"Unknown ansatz {ansatz!r}; expected 'orientation' or 'full'")

    Gamma = build_clock_matrix(N)
    if twist:
        z = build_twist(D, N, L, k=k)
    else:
        z = jnp.ones((D, D), dtype=jnp.complex128)

    if init_params is None:
        init_params = _default_init_params(ansatz, D, N, seed)
    params = [hermitianize(p) for p in init_params]

    expected_len = D - 1 if ansatz == "orientation" else D
    if len(params) != expected_len:
        raise ValueError(
            f"ansatz={ansatz} expects {expected_len} matrices, got {len(params)}"
        )

    if ansatz == "orientation":
        loss_fn, grad_fn, plaq_fn = _make_loss_and_grad_orientation(Gamma, z, D)
    else:
        loss_fn, grad_fn, plaq_fn = _make_loss_and_grad_full(z, D)

    optimizer = _build_optimizer(lr, n_steps, warmup, grad_clip)
    opt_state = optimizer.init(params)

    @partial(jit, static_argnums=())
    def _step(ps: list[jnp.ndarray], state):
        grads = grad_fn(ps)
        # JAX convention for real loss of complex z: grad returns
        # ∂f/∂x − i·∂f/∂y = conj(descent direction). For gradient descent via
        # `params − lr · update` we need `update = conj(grad_JAX)`.
        grads = [jnp.conj(g) for g in grads]
        # Project onto the Hermitian tangent space (we constrain params = params†).
        grads = [hermitianize(g) for g in grads]
        updates, new_state = optimizer.update(grads, state, ps)
        new_ps = optax.apply_updates(ps, updates)
        new_ps = [hermitianize(p) for p in new_ps]
        return new_ps, new_state, grads

    history: dict[str, list[float]] = {"step": [], "loss": [], "plaq": [], "grad_norm": []}
    converged = False
    final_grad_norm = float("inf")

    for step in range(n_steps):
        params, opt_state, grads = _step(params, opt_state)

        if (step % log_every == 0) or (step == n_steps - 1):
            loss_val = float(loss_fn(params))
            plaq_val = float(plaq_fn(params))
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

    final_loss = float(loss_fn(params))
    final_plaquette = float(plaq_fn(params))

    return OptResult(
        params=params,
        ansatz=ansatz,
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
    ansatz: str = "orientation",
    seed: int = 42,
    verbose: bool = True,
) -> dict[float, OptResult]:
    """Solve TEK across a decreasing schedule of couplings, warm-starting.

    Order lam_schedule from large (strong coupling, near disordered saddle)
    to small (weak coupling, ordered saddle) for the best warm-start.
    """
    results: dict[float, OptResult] = {}
    p_init: list[jnp.ndarray] | None = _default_init_params(ansatz, D, N, seed)

    for i, lam in enumerate(lam_schedule):
        if verbose:
            print(f"\n── λ = {lam:.4f}  (stage {i + 1}/{len(lam_schedule)}, "
                  f"ansatz={ansatz}) ──")
        res = optimize_tek(
            D=D,
            N=N,
            lam=lam,
            n_steps=n_steps_per,
            lr=lr,
            init_params=p_init,
            k=k,
            twist=twist,
            ansatz=ansatz,
            seed=seed,
            verbose=verbose,
        )
        results[lam] = res
        p_init = res.params  # warm-start next stage

    return results


# Default coupling schedule (strong → weak)
DEFAULT_LAM_SCHEDULE: list[float] = [
    20.0, 15.0, 10.0, 8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0,
    1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3,
]
