"""mm_loss.py — Makeenko-Migdal loop-equation loss for TEK.

Computes Wilson loops W[C] from TEK link matrices U_μ (orientation or full
ansatz) for every loop C in a precomputed LoopSystem (from master_field/
lattice.py), then evaluates the MM equation residuals and returns their
sum-of-squares. Minimizing this loss pushes W[C] toward the master-field
values satisfying the Makeenko-Migdal equations.

Wilson-loop convention (D = 2 only for now):

    W[C] = Re[ z_12^{signed_area_2d(C)} · Tr(Π U_{μ_i}) ] / N

The twist phase z_12^{area} generalizes the rectangular-case factor from
arXiv:1708.00841 eq. (2.4) to arbitrary simple closed loops. For D > 2 or
self-intersecting loops the twist-factor construction is more subtle (not
implemented).

MM equation (candidate D from `master_field/mm_equations.py`):

    (1/λ) Σ_{k ∈ lhs} w[k]  =  c_self · w[loop]  +  Σ_{(i,j) ∈ splits} w[i] w[j]

with c_self = 2 for the leading-order staple form. The residual per equation
is LHS − RHS; the loss is the squared sum over all equations for loops up to
length L_max.

Reusable parts from master_field/lattice.py: LoopSystem, build_loop_system,
signed_area_2d.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402
from jax import grad, jit, random  # noqa: E402

_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import LoopSystem, abs_area_2d, build_loop_system, signed_area_2d  # noqa: E402

from tek import (  # noqa: E402
    build_clock_matrix,
    build_link_matrices,
    build_link_matrices_full,
    build_twist,
    hermitianize,
    init_H_list_random,
    init_M_list_random,
)


# ═══════════════════════════════════════════════════════════════
# Twist-phase precomputation for each loop
# ═══════════════════════════════════════════════════════════════


def twist_factors_for_loops(loops: list[tuple[int, ...]], z: jnp.ndarray, D: int) -> list[jnp.ndarray]:
    """For each canonical loop, precompute the scalar twist factor.

    For D=2: Z(C) = z_12^{signed_area_2d(C)}. The empty loop gets Z = 1.
    For D>2: NOT IMPLEMENTED; raises. Extend via plane-wise signed area.
    """
    if D != 2:
        raise NotImplementedError(
            f"twist_factors_for_loops: D={D} not supported yet. D=2 only."
        )

    out: list[jnp.ndarray] = []
    for word in loops:
        if not word:
            out.append(jnp.ones((), dtype=jnp.complex128))
            continue
        area = signed_area_2d(word)
        # z_{μν}^{area}. For D=2, only μ=1, ν=2 is twisted.
        twist = z[0, 1] ** area
        out.append(twist.astype(jnp.complex128))
    return out


# ═══════════════════════════════════════════════════════════════
# Wilson loop evaluation
# ═══════════════════════════════════════════════════════════════


def wilson_loop_from_U(
    U_list: list[jnp.ndarray],
    word: tuple[int, ...],
    twist: jnp.ndarray,
    N: int,
) -> jnp.ndarray:
    """W[C] = Re[ twist · Tr(product over steps of word) ] / N.

    Each step μ_i ∈ ±{1,…,D}; positive → U_{|μ_i|-1}, negative → U†_{|μ_i|-1}.
    """
    if not word:
        return jnp.ones((), dtype=jnp.float64)
    prod = None
    for step in word:
        op = U_list[step - 1] if step > 0 else jnp.conj(U_list[-step - 1].T)
        prod = op if prod is None else prod @ op
    return jnp.real(twist * jnp.trace(prod)) / N


def compute_all_wilson_loops(
    U_list: list[jnp.ndarray],
    loops: list[tuple[int, ...]],
    twists: list[jnp.ndarray],
    N: int,
) -> jnp.ndarray:
    """Stack W[C] for every C in loops into a (K,) JAX array."""
    w_vals = [wilson_loop_from_U(U_list, word, tw, N) for word, tw in zip(loops, twists)]
    return jnp.stack(w_vals)


# ═══════════════════════════════════════════════════════════════
# MM-loss factory
# ═══════════════════════════════════════════════════════════════


@dataclass
class MMLossFn:
    """JIT-compiled MM-loss callable bound to a specific LoopSystem and context.

    Call `self(params, lam)` to get the loss; `self.grad(params, lam)` for the
    gradient w.r.t. params. Also exposes `self.wilson_loops(params)` for debug.
    """

    loss: Callable[[list[jnp.ndarray], float], jnp.ndarray]
    grad: Callable[[list[jnp.ndarray], float], list[jnp.ndarray]]
    wilson_loops: Callable[[list[jnp.ndarray]], jnp.ndarray]
    loop_sys: LoopSystem
    ansatz: str
    D: int
    N: int


def _gw_w_plus(lam: jnp.ndarray) -> jnp.ndarray:
    """Gross-Witten single-plaquette value:  1/(2λ) strong,  1 − λ/2 weak."""
    return jnp.where(lam >= 1.0, 0.5 / lam, 1.0 - lam / 2.0)


def _loop_areas(loops: list[tuple[int, ...]], D: int) -> jnp.ndarray:
    """|Area(C)| for each loop (D=2 only). Empty loop → 0."""
    if D != 2:
        return jnp.zeros(len(loops), dtype=jnp.int32)
    areas = []
    for w in loops:
        if not w:
            areas.append(0)
            continue
        try:
            areas.append(abs_area_2d(w))
        except ValueError:
            areas.append(0)
    return jnp.array(areas, dtype=jnp.int32)


def make_mm_loss_fn(
    loop_sys: LoopSystem,
    Gamma: jnp.ndarray,
    z: jnp.ndarray,
    D: int,
    N: int,
    ansatz: str,
    anchor: str = "none",
    anchor_weight: float = 1.0,
) -> MMLossFn:
    """Build a JIT-compiled MM loss callable.

    Args:
        anchor: supervised-anchor mode. One of:
            - "none": pure MM residual.
            - "plaquette": add anchor_weight · (W[plaq] − w_+(λ))² for the elementary
              plaquette loop (canonical form `(-2, -1, 2, 1)`).
            - "area_law": add anchor_weight · Σ_i (W[C_i] − w_+(λ)^|Area(C_i)|)²
              for all loops with a well-defined D=2 absolute area. Strong
              supervision toward the Gross-Witten lattice area law.
        anchor_weight: scalar multiplier for the anchor term.

    The MM equations from `loop_sys.mm_equations` are unrolled at JIT time.
    """
    if ansatz not in ("orientation", "full"):
        raise ValueError(f"Unknown ansatz {ansatz!r}")
    if anchor not in ("none", "plaquette", "area_law"):
        raise ValueError(f"Unknown anchor {anchor!r}")

    loops = loop_sys.loops
    twists = twist_factors_for_loops(loops, z, D)
    eqs = loop_sys.mm_equations
    areas = _loop_areas(loops, D) if anchor != "none" else None

    # Identify the canonical plaquette index for "plaquette" anchor.
    plaq_idx: int | None = None
    if anchor == "plaquette":
        # Plaquette's canonical form is the one of length 4 with |area|=1.
        for i, w in enumerate(loops):
            if len(w) == 4 and abs_area_2d(w) == 1:
                plaq_idx = i
                break
        if plaq_idx is None:
            raise RuntimeError("plaquette not found in LoopSystem — is L_max too small?")

    def _build_U(params: list[jnp.ndarray]) -> list[jnp.ndarray]:
        if ansatz == "orientation":
            return build_link_matrices(params, Gamma)
        return build_link_matrices_full(params)

    def _wilson_vec(params: list[jnp.ndarray]) -> jnp.ndarray:
        U = _build_U(params)
        return compute_all_wilson_loops(U, loops, twists, N)

    def _mm_residual(w: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
        total = jnp.zeros((), dtype=jnp.float64)
        for eq in eqs:
            if eq.lhs_loop_indices:
                lhs = jnp.sum(jnp.stack([w[k] for k in eq.lhs_loop_indices])) / lam
            else:
                lhs = jnp.zeros(())
            rhs = eq.rhs_self_coeff * w[eq.loop_idx]
            for i, j in eq.rhs_split_pairs:
                rhs = rhs + w[i] * w[j]
            total = total + (lhs - rhs) ** 2
        return total

    def _anchor_term(w: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
        if anchor == "none":
            return jnp.zeros((), dtype=jnp.float64)
        w_plus = _gw_w_plus(lam)
        if anchor == "plaquette":
            target = w_plus
            return anchor_weight * (w[plaq_idx] - target) ** 2
        # area_law
        targets = w_plus ** areas.astype(jnp.float64)
        # Only penalize loops with well-defined area (nonzero index or empty set)
        diffs = w - targets
        return anchor_weight * jnp.sum(diffs ** 2)

    def _loss(params: list[jnp.ndarray], lam: jnp.ndarray) -> jnp.ndarray:
        w = _wilson_vec(params)
        return _mm_residual(w, lam) + _anchor_term(w, lam)

    loss_jit = jit(_loss)
    grad_jit = jit(grad(_loss, argnums=0))
    wilson_jit = jit(_wilson_vec)

    return MMLossFn(
        loss=loss_jit,
        grad=grad_jit,
        wilson_loops=wilson_jit,
        loop_sys=loop_sys,
        ansatz=ansatz,
        D=D,
        N=N,
    )


# ═══════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════


@dataclass
class MMOptResult:
    """Result of MM-loss optimization run."""

    params: list[jnp.ndarray]
    ansatz: str
    lam: float
    D: int
    N: int
    L_max: int
    final_loss: float
    final_grad_norm: float
    final_wilson_loops: dict[tuple[int, ...], float]
    history: dict[str, list[float]]
    converged: bool


def _build_optimizer(lr: float, n_steps: int, warmup: int, grad_clip: float) -> optax.GradientTransformation:
    sched = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.01,
        peak_value=lr,
        warmup_steps=min(warmup, max(1, n_steps // 5)),
        decay_steps=n_steps,
        end_value=lr * 0.01,
    )
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(sched),
    )


def optimize_tek_mm(
    D: int,
    N: int,
    lam: float,
    L_max: int = 6,
    ansatz: str = "orientation",
    k: int = 1,
    twist: bool = True,
    init_params: list[jnp.ndarray] | None = None,
    n_steps: int = 3000,
    lr: float = 1e-2,
    warmup: int = 200,
    grad_clip: float = 1.0,
    log_every: int = 100,
    tol: float = 1e-8,
    seed: int = 42,
    verbose: bool = True,
    anchor: str = "none",
    anchor_weight: float = 1.0,
) -> MMOptResult:
    """Run Adam on the MM-loss (+ optional supervised anchor), starting from a
    random or supplied init.

    D=2 only for now (twist factor uses signed_area_2d). L_max controls the
    size of the LoopSystem; 6 is the Phase 1b default (35 loops, 32 equations).

    anchor ∈ {"none", "plaquette", "area_law"} — see make_mm_loss_fn.
    """
    if D != 2:
        raise NotImplementedError("optimize_tek_mm: D=2 only (for now)")
    if ansatz not in ("orientation", "full"):
        raise ValueError(f"Unknown ansatz {ansatz!r}")

    L_lat = int(round(N**0.5))
    if L_lat * L_lat != N:
        raise ValueError(f"N must be a perfect square; got N={N}")

    Gamma = build_clock_matrix(N)
    z = build_twist(D, N, L_lat, k=k) if twist else jnp.ones((D, D), dtype=jnp.complex128)
    loop_sys = build_loop_system(D=D, L_max=L_max)

    mm = make_mm_loss_fn(
        loop_sys, Gamma, z, D, N, ansatz,
        anchor=anchor, anchor_weight=anchor_weight,
    )

    if init_params is None:
        key = random.PRNGKey(seed)
        if ansatz == "orientation":
            init_params = init_H_list_random(D, N, key, scale=0.01)
        else:
            init_params = init_M_list_random(D, N, key, scale=0.01)
    params = [hermitianize(p) for p in init_params]

    expected_len = D - 1 if ansatz == "orientation" else D
    if len(params) != expected_len:
        raise ValueError(f"ansatz={ansatz} expects {expected_len} matrices, got {len(params)}")

    optimizer = _build_optimizer(lr, n_steps, warmup, grad_clip)
    opt_state = optimizer.init(params)

    @partial(jit, static_argnums=())
    def _step(ps, state, lam_local):
        grads = mm.grad(ps, lam_local)
        # Same JAX complex-gradient fix as optimize_tek: conjugate then hermitianize.
        grads = [jnp.conj(g) for g in grads]
        grads = [hermitianize(g) for g in grads]
        updates, new_state = optimizer.update(grads, state, ps)
        new_ps = optax.apply_updates(ps, updates)
        new_ps = [hermitianize(p) for p in new_ps]
        return new_ps, new_state, grads

    history: dict[str, list[float]] = {"step": [], "loss": [], "grad_norm": []}
    converged = False
    final_grad_norm = float("inf")

    for step in range(n_steps):
        params, opt_state, grads = _step(params, opt_state, float(lam))

        if step % log_every == 0 or step == n_steps - 1:
            loss_val = float(mm.loss(params, float(lam)))
            gnorm = max(float(jnp.linalg.norm(g)) for g in grads) / N
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["grad_norm"].append(gnorm)
            final_grad_norm = gnorm
            if verbose:
                print(f"  step {step:6d}  mm_loss={loss_val:.6e}  |grad|/N={gnorm:.2e}")
            if gnorm < tol:
                converged = True
                if verbose:
                    print(f"  converged at step {step} (|grad|/N < {tol:.0e})")
                break

    # Final wilson loops for inspection
    w_final = mm.wilson_loops(params)
    final_wilson_loops = {
        tuple(word): float(w_final[i]) for i, word in enumerate(loop_sys.loops)
    }

    return MMOptResult(
        params=params,
        ansatz=ansatz,
        lam=float(lam),
        D=D,
        N=N,
        L_max=L_max,
        final_loss=float(mm.loss(params, float(lam))),
        final_grad_norm=final_grad_norm,
        final_wilson_loops=final_wilson_loops,
        history=history,
        converged=converged,
    )
