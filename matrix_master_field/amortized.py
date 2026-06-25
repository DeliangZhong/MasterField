"""Amortized master field: one network g ↦ monomial coefficients, so a single
model represents M̂(g) across a whole coupling family. Seeds the Milestone-5
headline (amortized M̂(λ)). Trained unsupervised on the summed loop-equation
residual over a set of couplings; generalizes to held-out couplings by the
smoothness of the master field in g.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from matrix_master_field.ansatz import MonomialAnsatz  # noqa: E402
from matrix_master_field.fock_jax import power_moments  # noqa: E402
from matrix_master_field.loss import (  # noqa: E402
    kz_sd_residual_from_moment,
    one_matrix_sd_residual,
    symmetry_losses_from_moment,
    two_matrix_test_words,
)
from matrix_master_field.sparse_fock import SuffixSharedMoments  # noqa: E402


class AmortizedMonomial:
    """MLP(g) -> monomial coefficients; M̂(g) via the MonomialAnsatz assembly."""

    def __init__(self, fock_ops, degree, hidden=32):
        self.mono = MonomialAnsatz(fock_ops, degree)
        self.P = self.mono.n_params
        self.hidden = hidden

    def init_params(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        h = self.hidden
        return {
            "w1": 0.5 * jax.random.normal(k1, (1, h), dtype=jnp.float64),
            "b1": jnp.zeros(h, dtype=jnp.float64),
            "w2": 0.5 * jax.random.normal(k2, (h, h), dtype=jnp.float64),
            "b2": jnp.zeros(h, dtype=jnp.float64),
            "w3": 0.1 * jax.random.normal(k3, (h, self.P), dtype=jnp.float64),
            "b3": jnp.zeros(self.P, dtype=jnp.float64),
        }

    def coeffs(self, params, g):
        x = jnp.array([g], dtype=jnp.float64)
        x = jnp.tanh(x @ params["w1"] + params["b1"])
        x = jnp.tanh(x @ params["w2"] + params["b2"])
        return (x @ params["w3"] + params["b3"]).reshape(-1)

    def build_operators(self, params, g):
        return self.mono.build_operators(self.coeffs(params, g))


def train_amortized(model, vprime_fn, g_values, K, *, steps=6000, lr=3e-3, seed=0):
    """Minimize the mean SD residual over `g_values`. vprime_fn(g) -> v' list."""
    g_list = [float(g) for g in g_values]

    def loss_fn(params):
        tot = jnp.asarray(0.0, dtype=jnp.float64)
        for g in g_list:
            M = model.build_operators(params, g)[0]
            tot = tot + one_matrix_sd_residual(power_moments(M, K), vprime_fn(g))
        return tot / len(g_list)

    sched = optax.warmup_cosine_decay_schedule(lr * 0.01, lr, 300, steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))

    @jax.jit
    def run(params):
        st = opt.init(params)

        def body(c, _):
            p, s = c
            loss, grads = jax.value_and_grad(loss_fn)(p)
            u, s = opt.update(grads, s, p)
            return (optax.apply_updates(p, u), s), loss

        (p, _), ls = jax.lax.scan(body, (params, st), None, length=steps)
        return p, ls[-1]

    p, final = run(model.init_params(jax.random.PRNGKey(seed)))
    return p, float(final)


# ─── Two-matrix / two-coupling amortization: a single net (g,h) ↦ M̂(g,h) ─────────

class AmortizedKZ:
    """MLP(g,h) → Kazakov–Zheng master-field coefficients (n, n_monomials) on a
    `SparseMonomialField`. One network represents M̂(g,h) across the whole (g,h)
    coupling plane (novelty-(ii)). The output bias is warm-started to the free field,
    so at init the net returns the exact g=h=0 solution everywhere and learns the
    coupling deformation from there. A↔B exchange / Z2 are enforced by the symmetry
    loss in `train_amortized_kz` (not baked into the architecture)."""

    def __init__(self, field, hidden=64):
        self.field = field
        self.n = field.n
        self.m = field.n_monomials
        self.P = self.n * self.m
        self.hidden = hidden
        self._free = jnp.reshape(field.params_for_free_field(), (-1,))

    def init_params(self, key):
        k1, k2, k3 = jax.random.split(key, 3)
        h = self.hidden
        return {
            "w1": 0.3 * jax.random.normal(k1, (2, h), dtype=jnp.float64),
            "b1": jnp.zeros(h, dtype=jnp.float64),
            "w2": 0.3 * jax.random.normal(k2, (h, h), dtype=jnp.float64),
            "b2": jnp.zeros(h, dtype=jnp.float64),
            "w3": 0.01 * jax.random.normal(k3, (h, self.P), dtype=jnp.float64),
            "b3": self._free,  # warm start: output ≈ free field at init for all (g,h)
        }

    def coeffs(self, params, g, h):
        x = jnp.array([g, h], dtype=jnp.float64)
        x = jnp.tanh(x @ params["w1"] + params["b1"])
        x = jnp.tanh(x @ params["w2"] + params["b2"])
        return (x @ params["w3"] + params["b3"]).reshape(self.n, self.m)


def train_amortized_kz(model, gh_grid, *, max_word_len=2, w_sym=10.0, steps=4000,
                       lr=3e-3, seed=0):
    """Train `AmortizedKZ` on the mean KZ loop-equation + symmetry residual over a
    fixed (g,h) grid. Returns (theta, final_mean_loss). Generalization to held-out
    (g,h) follows from the smoothness of the master field in the couplings."""
    field = model.field
    words = two_matrix_test_words(max_word_len)
    grid = [(float(g), float(h)) for g, h in gh_grid]
    gmax = max((g for g, _ in grid), default=1.0) or 1.0
    hmax = max((h for _, h in grid), default=1.0) or 1.0

    # suffix-shared moment evaluator (record once at the largest couplings = superset)
    needed = []

    def _rec(w):
        needed.append(tuple(w))
        return 0.0

    kz_sd_residual_from_moment(_rec, words, gmax, hmax)
    symmetry_losses_from_moment(_rec, words)
    shared = SuffixSharedMoments(field, needed)

    def loss_fn(theta):
        tot = jnp.asarray(0.0, dtype=jnp.float64)
        for g, h in grid:
            moment = shared.moment_fn(model.coeffs(theta, g, h))
            tot = tot + (kz_sd_residual_from_moment(moment, words, g, h)
                         + w_sym * symmetry_losses_from_moment(moment, words))
        return tot / len(grid)

    sched = optax.warmup_cosine_decay_schedule(lr * 0.01, lr, min(300, steps // 5), steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))

    @jax.jit
    def run(theta):
        st = opt.init(theta)

        def body(c, _):
            p, s = c
            loss, grads = jax.value_and_grad(loss_fn)(p)
            u, s = opt.update(grads, s, p)
            return (optax.apply_updates(p, u), s), loss

        (p, _), ls = jax.lax.scan(body, (theta, st), None, length=steps)
        return p, ls[-1]

    theta, final = run(model.init_params(jax.random.PRNGKey(seed)))
    return theta, float(final)
