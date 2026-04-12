"""
neural_loop.py — Neural loop functional (Direction A of Phase 1).

Direct-table parametrization: an MLP mapping the 't Hooft coupling λ to a vector
of Wilson loop values w = (w_0, ..., w_{K-1}), one per canonical lattice loop in
the precomputed LoopSystem.

Symmetries are satisfied BY CONSTRUCTION because each equivalence class (cyclic,
reversal, B_D lattice symmetry) has a single canonical representative in the
LoopSystem. Unitarity bound is enforced via tanh on the MLP output.

Loss functions:
  - mm_loss:         Σ over MM equations of |residual|²  (unsupervised)
  - supervised_2d:   Σ over loops of |w - w_+^Area|²     (D=2 validation only)
  - empty_loop_loss: penalty forcing w[empty] = 1        (normalization)
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
from jax import jit, random  # noqa: E402

from lattice import LoopSystem, build_loop_system  # noqa: E402


class NeuralLoopFunctional:
    """MLP f_θ: λ → (w_0, ..., w_{K-1}) for a given LoopSystem.

    Architecture:
        Input : λ (scalar)
        Hidden: `n_layers` fully-connected GELU layers of `hidden_dim`
        Output: K real numbers → tanh → Wilson loop estimates

    The empty loop (index 0) is clamped to exactly 1 via `_apply_empty_constraint`.
    """

    def __init__(
        self,
        loop_system: LoopSystem,
        hidden_dim: int = 128,
        n_layers: int = 3,
    ):
        self.ls = loop_system
        self.K = loop_system.K
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Pre-stage MM equation tables as jax arrays + ragged padding
        self._stage_mm_tables()

    def _stage_mm_tables(self) -> None:
        """Pack the MM equations into padded JAX index arrays for JIT-friendly loss."""
        eqs = self.ls.mm_equations
        n_eq = len(eqs)
        max_lhs = max((len(eq.lhs_loop_indices) for eq in eqs), default=1)
        max_splits = max((len(eq.rhs_split_pairs) for eq in eqs), default=1)
        # Use at least 1 column to avoid empty-array indexing issues
        max_lhs = max(max_lhs, 1)
        max_splits = max(max_splits, 1)

        lhs_idx = np.zeros((n_eq, max_lhs), dtype=np.int32)
        lhs_mask = np.zeros((n_eq, max_lhs), dtype=bool)
        split_i = np.zeros((n_eq, max_splits), dtype=np.int32)
        split_j = np.zeros((n_eq, max_splits), dtype=np.int32)
        split_mask = np.zeros((n_eq, max_splits), dtype=bool)
        self_coeff = np.zeros((n_eq,), dtype=np.float64)
        self_idx = np.zeros((n_eq,), dtype=np.int32)

        for e, eq in enumerate(eqs):
            self_coeff[e] = eq.rhs_self_coeff
            self_idx[e] = eq.loop_idx
            for k, idx in enumerate(eq.lhs_loop_indices):
                lhs_idx[e, k] = idx
                lhs_mask[e, k] = True
            for k, (i, j) in enumerate(eq.rhs_split_pairs):
                split_i[e, k] = i
                split_j[e, k] = j
                split_mask[e, k] = True

        self._lhs_idx = jnp.asarray(lhs_idx)
        self._lhs_mask = jnp.asarray(lhs_mask)
        self._split_i = jnp.asarray(split_i)
        self._split_j = jnp.asarray(split_j)
        self._split_mask = jnp.asarray(split_mask)
        self._self_coeff = jnp.asarray(self_coeff)
        self._self_idx = jnp.asarray(self_idx)

        # D=2 areas for supervised validation
        if self.ls.areas is not None:
            areas_arr = np.zeros((self.K,), dtype=np.float64)
            for i, a in self.ls.areas.items():
                areas_arr[i] = a
            self._areas = jnp.asarray(areas_arr)
        else:
            self._areas = None  # type: ignore[assignment]

        # Empty-loop constraint mask (we'll overwrite w[0] = 1 after the MLP)
        # w = tanh(raw) clamps to [-1, 1]; we then set w[0] = 1 exactly.

    def init_params(self, key, out_scale: float = 0.01) -> dict:
        """Xavier-normal init for hidden layers; small scale for output (R1).

        Small `out_scale` sets initial W[C] ≈ 0 for non-empty loops. This is the
        correct basin for moderate/strong coupling (W ~ 1/(2λ)^Area is small).
        """
        keys = random.split(key, 2 * (self.n_layers + 1))
        params: dict = {}
        in_dim = 1
        for i in range(self.n_layers):
            fan_in = in_dim
            scale = jnp.sqrt(2.0 / fan_in)
            params[f"W{i}"] = scale * random.normal(keys[2 * i], (self.hidden_dim, fan_in))
            params[f"b{i}"] = jnp.zeros((self.hidden_dim,))
            in_dim = self.hidden_dim
        params["W_out"] = out_scale * random.normal(keys[-2], (self.K, in_dim))
        params["b_out"] = jnp.zeros((self.K,))
        return params

    @partial(jit, static_argnums=(0,))
    def predict(self, params: dict, lam: float) -> jnp.ndarray:
        """Forward pass. R1: no tanh; unitarity enforced via soft penalty in loss.

        The empty loop is clamped to exactly 1 (hard normalization).
        """
        h = jnp.array([lam], dtype=jnp.float64)
        for i in range(self.n_layers):
            h = jax.nn.gelu(params[f"W{i}"] @ h + params[f"b{i}"])
        raw = params["W_out"] @ h + params["b_out"]
        w = raw
        w = w.at[self.ls.empty_idx].set(1.0)
        return w

    @partial(jit, static_argnums=(0,))
    def unitarity_penalty(self, params: dict, lam: float) -> jnp.ndarray:
        """Soft |W[C]| ≤ 1 via a hinge-like penalty (R1)."""
        w = self.predict(params, lam)
        over = jnp.maximum(jnp.abs(w) - 1.0, 0.0)
        return jnp.sum(over ** 2)

    @partial(jit, static_argnums=(0,))
    def mm_loss(self, params: dict, lam: float) -> jnp.ndarray:
        """Mean-squared MM equation residual."""
        w = self.predict(params, lam)

        # LHS of each equation: (1/λ) Σ_k w[lhs_idx[e,k]] * mask
        lhs_terms = jnp.where(self._lhs_mask, w[self._lhs_idx], 0.0)
        lhs = jnp.sum(lhs_terms, axis=1) / lam

        # RHS: self_coeff * w[self_idx] + Σ splits
        rhs_self = self._self_coeff * w[self._self_idx]
        split_terms = jnp.where(
            self._split_mask, w[self._split_i] * w[self._split_j], 0.0
        )
        rhs = rhs_self + jnp.sum(split_terms, axis=1)

        resid = lhs - rhs
        return jnp.mean(resid ** 2)

    @partial(jit, static_argnums=(0,))
    def supervised_loss_2d(self, params: dict, lam: float) -> jnp.ndarray:
        """MSE against GW lattice-exact answer W[C] = w_+^Area, w_+ = 1/(2λ) for λ ≥ 1."""
        assert self._areas is not None, "supervised_loss_2d requires D=2 LoopSystem"
        w = self.predict(params, lam)
        w_plus = jnp.where(lam >= 1.0, 1.0 / (2.0 * lam), 1.0 - lam / 2.0)
        target = w_plus ** self._areas
        target = target.at[self.ls.empty_idx].set(1.0)
        return jnp.mean((w - target) ** 2)


# ═══════════════════════════════════════════════════════════
# Supervised training (Phase 1a) — D=2 only
# ═══════════════════════════════════════════════════════════


def train_supervised_2d(
    model: NeuralLoopFunctional,
    lr: float = 1e-3,
    n_epochs: int = 5000,
    lam_min: float = 1.0,
    lam_max: float = 5.0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[dict, list[float]]:
    """Phase 1a: train the model to reproduce the 2D area law via MSE.

    Samples λ uniformly from [lam_min, lam_max] each epoch. Target per step is
    the GW lattice-exact Wilson loops at that λ. Returns (final_params, loss_history).
    """
    key = random.PRNGKey(seed)
    params = model.init_params(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @partial(jit, static_argnums=())
    def step(params, opt_state, lam):
        loss, grads = jax.value_and_grad(model.supervised_loss_2d)(params, lam)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses: list[float] = []
    rng = np.random.default_rng(seed + 1)
    for epoch in range(n_epochs):
        lam = float(rng.uniform(lam_min, lam_max))
        params, opt_state, loss = step(params, opt_state, lam)
        losses.append(float(loss))
        if verbose and (epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:5d}: λ={lam:.3f}  loss={float(loss):.3e}")

    return params, losses


def _build_combined_loss(model: NeuralLoopFunctional, unit_weight: float):
    """Factory for JIT-compatible combined MM + unitarity loss."""

    @partial(jit, static_argnums=())
    def combined_loss(params, lam):
        l_mm = model.mm_loss(params, lam)
        l_unit = model.unitarity_penalty(params, lam)
        return l_mm + unit_weight * l_unit

    return combined_loss


def train_mm_2d(
    model: NeuralLoopFunctional,
    lr: float = 1e-3,
    n_epochs: int = 10000,
    lam_min: float = 1.0,
    lam_max: float = 5.0,
    unit_weight: float = 10.0,
    seed: int = 0,
    params_init: dict | None = None,
    verbose: bool = True,
) -> tuple[dict, list[float]]:
    """Phase 1b: train via MM loss + soft unitarity, uniform λ sampling."""
    key = random.PRNGKey(seed)
    params = params_init if params_init is not None else model.init_params(key)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)
    combined_loss = _build_combined_loss(model, unit_weight)

    @partial(jit, static_argnums=())
    def step(params, opt_state, lam):
        loss, grads = jax.value_and_grad(combined_loss)(params, lam)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    losses: list[float] = []
    rng = np.random.default_rng(seed + 2)
    for epoch in range(n_epochs):
        lam = float(rng.uniform(lam_min, lam_max))
        params, opt_state, loss = step(params, opt_state, lam)
        losses.append(float(loss))
        if verbose and (epoch % max(n_epochs // 10, 1) == 0 or epoch == n_epochs - 1):
            print(f"  epoch {epoch:5d}: λ={lam:.3f}  loss={float(loss):.3e}")

    return params, losses


def train_mm_2d_curriculum(
    model: NeuralLoopFunctional,
    lam_schedule: list[float] | None = None,
    epochs_per_stage: int = 1000,
    lr: float = 1e-3,
    unit_weight: float = 10.0,
    seed: int = 0,
    params_init: dict | None = None,
    verbose: bool = True,
) -> tuple[dict, list[float]]:
    """R2: curriculum over λ, starting at high λ (easy, W ≈ 0) and annealing down.

    At each stage the λ is FIXED (no sampling). After a stage, keep params and
    reset the optimizer state for the next stage.
    """
    if lam_schedule is None:
        lam_schedule = [10.0, 8.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0]

    key = random.PRNGKey(seed)
    params = params_init if params_init is not None else model.init_params(key)
    combined_loss = _build_combined_loss(model, unit_weight)

    @partial(jit, static_argnums=())
    def step(params, opt_state, lam):
        loss, grads = jax.value_and_grad(combined_loss)(params, lam)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    all_losses: list[float] = []
    for stage, lam in enumerate(lam_schedule):
        if verbose:
            print(f"--- Stage {stage + 1}/{len(lam_schedule)}  λ={lam} ---")
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(params)
        for epoch in range(epochs_per_stage):
            params, opt_state, loss = step(params, opt_state, lam)
            all_losses.append(float(loss))
            if verbose and (epoch % max(epochs_per_stage // 5, 1) == 0 or epoch == epochs_per_stage - 1):
                print(f"  epoch {epoch:5d}: loss={float(loss):.3e}")
    return params, all_losses


def train_mm_2d_warmstart(
    model: NeuralLoopFunctional,
    n_super: int = 1500,
    n_mm: int = 5000,
    lr_super: float = 3e-3,
    lr_mm: float = 1e-3,
    unit_weight: float = 10.0,
    seed: int = 0,
    verbose: bool = True,
) -> tuple[dict, list[float]]:
    """R4: supervised warm-start, then MM-only fine-tuning."""
    params, losses_super = train_supervised_2d(
        model, lr=lr_super, n_epochs=n_super, seed=seed, verbose=verbose
    )
    if verbose:
        print("\n--- Switching to MM-only loss ---")
    params, losses_mm = train_mm_2d(
        model,
        lr=lr_mm,
        n_epochs=n_mm,
        unit_weight=unit_weight,
        seed=seed + 100,
        params_init=params,
        verbose=verbose,
    )
    return params, losses_super + losses_mm


# ═══════════════════════════════════════════════════════════
# Main: Phase 1a validation
# ═══════════════════════════════════════════════════════════


def phase_1a_main(L_max: int = 6) -> None:
    print("=" * 70)
    print(f"  Phase 1a — Supervised training in D=2 (L_max={L_max})")
    print("=" * 70)
    ls = build_loop_system(D=2, L_max=L_max)
    print(f"LoopSystem: {ls.K} loops, {len(ls.mm_equations)} MM equations")
    model = NeuralLoopFunctional(ls, hidden_dim=128, n_layers=3)
    params, _losses = train_supervised_2d(model, n_epochs=3000, lr=3e-3)

    # Evaluate on a grid of λ
    print("\n  λ   | W[plaquette]_ML  W[plaquette]_exact  W[2x1]_ML  W[2x1]_exact")
    print("  " + "-" * 70)
    import math

    for lam in [1.0, 2.0, 5.0]:
        w = model.predict(params, lam)
        w_plus = 1.0 / (2.0 * lam) if lam >= 1.0 else 1.0 - lam / 2.0
        # Find plaquette and 2x1 indices
        plaq_idx = None
        rect_idx = None
        for idx, loop in enumerate(ls.loops):
            if not loop:
                continue
            if ls.areas[idx] == 1 and plaq_idx is None:
                plaq_idx = idx
            if ls.areas[idx] == 2 and rect_idx is None:
                rect_idx = idx
        w_plaq = float(w[plaq_idx]) if plaq_idx is not None else float("nan")
        w_rect = float(w[rect_idx]) if rect_idx is not None else float("nan")
        print(
            f"  {lam:.1f} | {w_plaq:16.6f}  {w_plus:17.6f}  "
            f"{w_rect:9.6f}  {w_plus ** 2:.6f}"
        )
        _ = math  # suppress unused import lint


if __name__ == "__main__":
    phase_1a_main()
