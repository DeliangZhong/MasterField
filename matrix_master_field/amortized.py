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
from matrix_master_field.loss import one_matrix_sd_residual  # noqa: E402


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
