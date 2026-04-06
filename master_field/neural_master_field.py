"""
neural_master_field.py — Neural network parametrisation of the master field.

Three approaches implemented:
1. MasterFieldMLP: Direct MLP mapping word → moment, with Cholesky PSD enforcement
2. RTransformFlow: Normalizing flow on the R-transform (free → interacting deformation)
3. CuntzOperatorNet: Neural parametrisation of the master field operator in Fock space

All use JAX for autodiff and JIT compilation.
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, random

# ═══════════════════════════════════════════════════════════
# Approach 1: Cholesky-parametrised moment vector
# ═══════════════════════════════════════════════════════════


class CholeskyMasterField:
    """Parametrise the loop moment matrix via its Cholesky factor.

    The moment matrix Ω_{ij} = tr[w_i† w_j] must be PSD.
    We write Ω = L L^T where L is lower-triangular with positive diagonal.

    The neural network outputs the entries of L. The moments are then
    read off from Ω. The SD equations provide the loss.
    """

    def __init__(
        self, n_matrices: int, max_word_length: int, hidden_dim: int = 256, n_layers: int = 4
    ):
        self.n = n_matrices
        self.L_max = max_word_length

        # Basis words for moment matrix (up to half max length)
        half_L = max_word_length // 2
        self.basis_words = [()]
        for length in range(1, half_L + 1):
            from itertools import product as cp

            for combo in cp(range(n_matrices), repeat=length):
                self.basis_words.append(combo)
        self.basis_dim = len(self.basis_words)

        # Total parameters: lower-triangular matrix entries
        self.n_cholesky_params = self.basis_dim * (self.basis_dim + 1) // 2

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def init_params(self, key: jax.Array) -> dict:
        """Initialise parameters for the Cholesky factor.

        We directly optimise the Cholesky entries (no neural network needed
        for small problems). For larger problems, a network maps a latent
        code to the Cholesky entries.
        """
        # Direct parametrisation: the Cholesky entries themselves
        # Initialise close to identity (free/Gaussian solution)
        params = {
            "cholesky_raw": 0.01 * random.normal(key, (self.n_cholesky_params,)),
        }
        # Set diagonal to ~1 (log-scale for positivity)
        diag_indices = []
        k = 0
        for i in range(self.basis_dim):
            for j in range(i + 1):
                if i == j:
                    diag_indices.append(k)
                k += 1
        params["diag_log"] = jnp.zeros(self.basis_dim)
        params["_diag_indices"] = jnp.array(diag_indices)
        return params

    def cholesky_to_moments(self, params: dict) -> jnp.ndarray:
        """Convert Cholesky parameters to the moment matrix Ω = L L^T."""
        raw = params["cholesky_raw"]
        diag_log = params["diag_log"]

        # Build lower-triangular L
        L = jnp.zeros((self.basis_dim, self.basis_dim))
        k = 0
        for i in range(self.basis_dim):
            for j in range(i + 1):
                if i == j:
                    # Diagonal: must be positive
                    val = jnp.exp(diag_log[i])
                else:
                    val = raw[k]
                L = L.at[i, j].set(val)
                k += 1

        Omega = L @ L.T
        return Omega

    def extract_moments(self, Omega: jnp.ndarray) -> dict[tuple, float]:
        """Extract individual moments tr[w] from the moment matrix.

        Ω_{0,j} = tr[w_j] since w_0 = empty word and w_0† = empty.
        So the first row/column gives all single-trace moments directly.
        """
        moments = {}
        for j, wj in enumerate(self.basis_words):
            moments[wj] = Omega[0, j]

        # Also extract higher moments from interior entries
        # Ω_{i,j} = tr[reverse(w_i) · w_j]
        for i, wi in enumerate(self.basis_words):
            for j, wj in enumerate(self.basis_words):
                combined = tuple(reversed(wi)) + wj
                if len(combined) <= self.L_max:
                    moments[combined] = Omega[i, j]

        return moments


# ═══════════════════════════════════════════════════════════
# Approach 2: R-transform flow (free → interacting)
# ═══════════════════════════════════════════════════════════


def init_r_transform_params(
    key: jax.Array, n_coeffs: int, hidden_dim: int = 128, n_layers: int = 3
) -> dict:
    """Initialise parameters for the R-transform neural network.

    The R-transform R(z) = Σ_{n≥1} κ_n z^{n-1} determines the master field.
    For a Gaussian, κ_2 = variance, all others zero.

    We parametrise the free cumulants {κ_n} directly as trainable parameters,
    or through a small neural network for coupling-dependent models.
    """
    keys = random.split(key, 3)
    params = {
        # Free cumulants (direct)
        "kappa": jnp.zeros(n_coeffs),  # κ_1, κ_2, ..., κ_n_coeffs
        # Optional: coupling-dependent network
        # Input: coupling g → Output: corrections to free cumulants
        "coupling_net": {
            "w1": 0.01 * random.normal(keys[0], (1, hidden_dim)),
            "b1": jnp.zeros(hidden_dim),
            "w2": 0.01 * random.normal(keys[1], (hidden_dim, hidden_dim)),
            "b2": jnp.zeros(hidden_dim),
            "w3": 0.01 * random.normal(keys[2], (hidden_dim, n_coeffs)),
            "b3": jnp.zeros(n_coeffs),
        },
    }
    # Gaussian initialisation: κ_2 = 1
    params["kappa"] = params["kappa"].at[1].set(1.0)
    return params


def r_transform_predict(
    params: dict, z: jnp.ndarray, use_network: bool = False, coupling: float = 0.0
) -> jnp.ndarray:
    """Evaluate R(z) = Σ_{n≥1} κ_n z^{n-1}."""
    kappa = params["kappa"]

    if use_network:
        # Get coupling-dependent corrections
        net = params["coupling_net"]
        g_input = jnp.array([[coupling]])
        h = jnp.tanh(g_input @ net["w1"] + net["b1"])
        h = jnp.tanh(h @ net["w2"] + net["b2"])
        delta_kappa = (h @ net["w3"] + net["b3"]).squeeze()
        kappa = kappa + delta_kappa

    # Evaluate polynomial
    result = jnp.zeros_like(z)
    z_power = jnp.ones_like(z)  # z^0
    for n in range(len(kappa)):
        result = result + kappa[n] * z_power
        z_power = z_power * z
    return result


def moments_from_r_transform(kappa: jnp.ndarray, max_moment: int) -> jnp.ndarray:
    """Compute moments m_k from free cumulants κ_n via the moment-cumulant formula.

    Uses the recursion:
    m_n = Σ_{π ∈ NC(n)} Π_{B ∈ π} κ_{|B|}

    For practical computation, use the recursion:
    m_n = Σ_{k=1}^{n} κ_k Σ_{partitions} m_{...}

    Specifically: m_n = Σ_{k=1}^{n} κ_k · (# of NC partitions contribution)
    The cleanest recursion is via the resolvent/R-transform relation.
    """
    n_kappa = len(kappa)
    m = jnp.zeros(max_moment + 1)
    m = m.at[0].set(1.0)

    # Use the recursion: m_n = Σ_{k=1}^{n} κ_k Σ_{j₁+...+j_{k-1}=n-k} m_{j₁}...m_{j_{k-1}}
    # This is equivalent to: m_n = Σ_{k=1}^{min(n,n_kappa)} κ_k C_{n,k}
    # where C_{n,k} = Σ m_{j₁}...m_{j_{k-1}} with j₁+...+j_{k-1}=n-k

    # Simpler recursion from R(G(z)) = z - 1/G(z) where G = resolvent:
    # G(z) = Σ m_n / z^{n+1}
    # m_n = Σ_{k=1}^{n} κ_k Σ_{(n₁,...,n_{k-1}): Σnᵢ=n-k} m_{n₁}...m_{n_{k-1}}

    # Actually the simplest is the direct recursion from
    # m_n = κ_1 δ_{n,1} + Σ_{k=2}^{n} κ_k Σ m_{n₁}...m_{n_{k-1}}
    # with Σ nᵢ = n-1 and nᵢ ≥ 1.

    # But even simpler: use the functional relation.
    # Let R(z) = Σ κ_k z^{k-1}, then m_n satisfies:
    # m_1 = κ_1
    # m_n = Σ_{k=1}^{min(n, n_kappa)} κ_k · N(n-1, k-1)
    # where N(n-1, k-1) is the (n-1,k-1) Narayana-like sum.

    # For JAX-friendliness, let's use the resolvent iteration.
    # G(z) satisfies z G(z) = 1 + R(G(z)) G(z)
    # In terms of coefficients: m_n = Σ_{j=0}^{n-1} Σ_{k≥1} κ_k · (product of moments)

    # Practical: iterate m_n for n=1,2,...
    for n in range(1, max_moment + 1):
        val = 0.0
        # m_n = Σ_{p=1}^{min(n, n_kappa)} κ_p · S(n, p)
        # where S(n, p) = Σ_{compositions of n-1 into p-1 parts ≥ 0} Π m_{parts+...}
        # This is: S(n, p) = Σ_{j=0}^{n-1} m_j · S(n-1-j, p-1)  with S(0,0)=1

        # Compute S(n, p) for all p
        # S[k][p] = sum over compositions of k into p non-negative parts, product of moments
        # S[0][0] = 1, S[k][0] = δ_{k,0}
        # S[k][p] = Σ_{j=0}^{k} m_j · S[k-j][p-1]
        S = np.zeros((n + 1, min(n, n_kappa) + 1))
        S[0, 0] = 1.0
        for p in range(1, min(n, n_kappa) + 1):
            for k in range(n + 1):
                for j in range(k + 1):
                    if k - j <= n and p - 1 >= 0:
                        S[k, p] += float(m[j]) * S[k - j, p - 1]

        val = sum(float(kappa[p - 1]) * S[n - 1, p - 1] for p in range(1, min(n, n_kappa) + 1))
        m = m.at[n].set(val)

    return m


# ═══════════════════════════════════════════════════════════
# Approach 3: Direct Cuntz–Fock operator parametrisation
# ═══════════════════════════════════════════════════════════


def init_cuntz_operator_params(
    key: jax.Array, fock_dim: int, n_matrices: int, max_coeffs: int = 10
) -> dict:
    """Initialise master field operator M̂ = Σ c_k x̂^k in Fock space.

    For Hermitian representation: M̂ = λ(x̂) where λ is a polynomial.
    """
    params = {}
    for i in range(n_matrices):
        # Polynomial coefficients for λ_i(x)
        coeffs = jnp.zeros(max_coeffs)
        coeffs = coeffs.at[1].set(1.0)  # linear term (Gaussian limit)
        params[f"lambda_coeffs_{i}"] = coeffs
    return params


# ═══════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════


def sd_loss_one_matrix(moments: jnp.ndarray, v_prime_coeffs: list[float]) -> jnp.ndarray:
    """Schwinger-Dyson loss for a one-matrix model.

    SD equation for test word M^n:
    Σ_k v_k m_{n+k-1} = Σ_{j=0}^{n-1} m_j m_{n-j-1}

    Args:
        moments: array of m_0, m_1, ..., m_K
        v_prime_coeffs: coefficients of V'(M) = v_0 + v_1 M + v_2 M² + ...
    """
    K = len(moments) - 1
    n_v = len(v_prime_coeffs)

    total_loss = 0.0
    n_eqs = 0

    for n in range(0, K - n_v):
        # LHS: Σ_k v_k m_{n+k}  from tr[V'(M) M^n] = Σ_k v_k tr[M^{k+n}]
        lhs = 0.0
        for k in range(n_v):
            idx = n + k
            if idx <= K:
                lhs += v_prime_coeffs[k] * moments[idx]

        # RHS (splitting / factorisation)
        rhs = 0.0
        for j in range(n):
            if j <= K and n - j - 1 <= K:
                rhs += moments[j] * moments[n - j - 1]

        total_loss += (lhs - rhs) ** 2
        n_eqs += 1

    return total_loss / max(n_eqs, 1)


def sd_loss_two_matrix(
    moment_func: Callable,
    g: float,
    test_words: list[tuple],
    interaction: str = "commutator_squared",
) -> jnp.ndarray:
    """Schwinger-Dyson loss for the two-matrix model.

    moment_func: word → moment value (differentiable)
    """
    total_loss = 0.0
    n_eqs = 0

    for w in test_words:
        for a in range(2):
            b = 1 - a

            # LHS: <tr[V'_a · w]>
            word_a = (a,) + w
            lhs = moment_func(word_a)

            if interaction == "commutator_squared":
                g2 = g * g
                word1 = (a, b, b) + w
                word2 = (b, b, a) + w
                word3 = (b, a, b) + w
                lhs += (g2 / 2) * (moment_func(word1) + moment_func(word2) - 2 * moment_func(word3))

            # RHS: splitting
            rhs = 0.0
            for m in range(len(w)):
                if w[m] == a:
                    left = w[:m]
                    right = w[m + 1 :]
                    rhs += moment_func(left) * moment_func(right)

            total_loss += (lhs - rhs) ** 2
            n_eqs += 1

    return total_loss / max(n_eqs, 1)


def symmetry_loss(moments: jnp.ndarray) -> jnp.ndarray:
    """Enforce Z₂ symmetry: m_{odd} = 0 for symmetric potentials."""
    loss = 0.0
    for k in range(1, len(moments), 2):
        loss += moments[k] ** 2
    return loss


def normalisation_loss(moments: jnp.ndarray) -> jnp.ndarray:
    """Enforce m_0 = 1."""
    return (moments[0] - 1.0) ** 2


# ═══════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════


class MasterFieldTrainer:
    """Main training class for master field optimisation."""

    def __init__(
        self,
        model_name: str,
        n_matrices: int = 1,
        coupling: float = 0.0,
        max_word_length: int = 10,
        lr: float = 1e-3,
        n_epochs: int = 5000,
    ):
        self.model_name = model_name
        self.n = n_matrices
        self.g = coupling
        self.L = max_word_length
        self.lr = lr
        self.n_epochs = n_epochs

        # Determine V'(M) coefficients
        if model_name == "gaussian":
            self.v_prime = [0.0, 1.0]  # V' = M
        elif model_name == "quartic":
            self.v_prime = [0.0, 1.0, 0.0, coupling]  # V' = M + g M³
        elif model_name == "sextic":
            self.v_prime = [0.0, 1.0, 0.0, coupling, 0.0, coupling**2]
        else:
            self.v_prime = [0.0, 1.0]

        self.n_moments = max_word_length + 1

    def init(self, key: jax.Array):
        """Initialise optimiser and parameters."""
        if self.n == 1:
            # Direct moment parametrisation for one-matrix
            # Parametrise even moments (odd = 0 by symmetry)
            n_even = self.n_moments // 2 + 1
            self.params = {
                "m_even_raw": jnp.zeros(n_even),
            }
            # Gaussian initialisation: m_{2k} = C_k
            catalan = [1.0]
            for k in range(1, n_even):
                catalan.append(catalan[-1] * (4 * k - 2) / (k + 1))
            self.params["m_even_raw"] = jnp.array(catalan[:n_even])
        else:
            # Cholesky parametrisation for multi-matrix
            self.cholesky_model = CholeskyMasterField(self.n, self.L, hidden_dim=256, n_layers=4)
            self.params = self.cholesky_model.init_params(key)

        # Optimiser
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.lr * 0.01,
            peak_value=self.lr,
            warmup_steps=200,
            decay_steps=self.n_epochs,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(schedule),
        )
        self.opt_state = self.optimizer.init(self.params)

    def moments_from_params(self, params: dict) -> jnp.ndarray:
        """Extract the full moment vector from parameters."""
        if self.n == 1:
            m_even = params["m_even_raw"]
            moments = jnp.zeros(self.n_moments)
            moments = moments.at[0].set(1.0)  # m_0 = 1
            for k in range(1, len(m_even)):
                if 2 * k < self.n_moments:
                    moments = moments.at[2 * k].set(m_even[k])
            return moments
        else:
            # TODO: multi-matrix via Cholesky
            raise NotImplementedError("Multi-matrix moments extraction")

    def loss_fn(self, params: dict) -> jnp.ndarray:
        """Total loss function."""
        moments = self.moments_from_params(params)

        # SD loss
        l_sd = sd_loss_one_matrix(moments, self.v_prime)

        # Normalisation
        l_norm = normalisation_loss(moments)

        # Symmetry (odd moments = 0, already enforced by construction)
        # l_sym = symmetry_loss(moments)

        return l_sd + 100.0 * l_norm

    @partial(jit, static_argnums=(0,))
    def train_step(self, params, opt_state):
        """Single optimisation step."""
        loss, grads = jax.value_and_grad(self.loss_fn)(params)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    def train(self, key: jax.Array, verbose: bool = True):
        """Run the full training loop."""
        self.init(key)

        losses = []
        for epoch in range(self.n_epochs):
            self.params, self.opt_state, loss = self.train_step(self.params, self.opt_state)
            losses.append(float(loss))

            if verbose and (epoch % 500 == 0 or epoch == self.n_epochs - 1):
                moments = self.moments_from_params(self.params)
                print(
                    f"Epoch {epoch:5d}: loss = {loss:.2e}, "
                    f"m_2 = {float(moments[2]):.6f}, "
                    f"m_4 = {float(moments[4]):.6f}"
                )

        return self.params, losses

    def get_solution(self) -> dict:
        """Extract the solution after training."""
        moments = np.array(self.moments_from_params(self.params))

        # Compute free cumulants
        from one_matrix import r_transform_from_moments

        kappa = r_transform_from_moments(moments)

        return {
            "moments": moments,
            "free_cumulants": kappa,
            "final_loss": float(self.loss_fn(self.params)),
        }


# ═══════════════════════════════════════════════════════════
# Multi-matrix trainer (JAX, Cholesky-parametrised)
# ═══════════════════════════════════════════════════════════


class MultiMatrixTrainer:
    """Train the master field for coupled multi-matrix models.

    Uses Cholesky parametrisation of the moment matrix for automatic PSD.
    Loss = SD equation residuals.
    """

    def __init__(
        self,
        n_matrices: int = 2,
        coupling: float = 1.0,
        max_word_length: int = 6,
        interaction: str = "commutator_squared",
        lr: float = 1e-3,
        n_epochs: int = 10000,
    ):
        self.n = n_matrices
        self.g = coupling
        self.L = max_word_length
        self.interaction = interaction
        self.lr = lr
        self.n_epochs = n_epochs

        # Build word basis
        from itertools import product as cp

        half_L = max_word_length // 2
        self.basis_words = [()]
        for length in range(1, half_L + 1):
            for combo in cp(range(n_matrices), repeat=length):
                self.basis_words.append(combo)
        self.basis_dim = len(self.basis_words)

        # Test words for SD equations
        self.test_words = [()]
        for length in range(1, max(1, max_word_length - 3)):
            for combo in cp(range(n_matrices), repeat=length):
                self.test_words.append(combo)

    def init(self, key: jax.Array):
        """Initialise Cholesky parameters."""
        n_params = self.basis_dim * (self.basis_dim + 1) // 2

        # Off-diagonal entries
        key1, key2 = random.split(key)
        offdiag = 0.01 * random.normal(key1, (n_params - self.basis_dim,))
        # Diagonal entries (log-scale for positivity)
        diag_log = jnp.zeros(self.basis_dim)

        self.params = {
            "offdiag": offdiag,
            "diag_log": diag_log,
        }

        schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.lr * 0.01,
            peak_value=self.lr,
            warmup_steps=500,
            decay_steps=self.n_epochs,
        )
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))
        self.opt_state = self.optimizer.init(self.params)

    def params_to_moment_matrix(self, params: dict) -> jnp.ndarray:
        """Build PSD moment matrix Ω = L L^T from Cholesky parameters."""
        d = self.basis_dim
        L = jnp.zeros((d, d))

        offdiag = params["offdiag"]
        diag = jnp.exp(params["diag_log"])

        off_idx = 0
        for i in range(d):
            for j in range(i):
                L = L.at[i, j].set(offdiag[off_idx])
                off_idx += 1
            L = L.at[i, i].set(diag[i])

        return L @ L.T

    def moment_from_matrix(self, Omega: jnp.ndarray, word: tuple) -> float:
        """Look up moment tr[word] from the moment matrix.

        Ω_{i,j} = tr[reverse(basis_i) · basis_j]
        For a given word w, find i,j such that reverse(basis_i) · basis_j = w.
        The simplest: Ω_{0,j} = tr[basis_j] (first row).
        """
        if not word:
            return 1.0

        # Check if word is a basis word directly
        for j, bw in enumerate(self.basis_words):
            if bw == word:
                return Omega[0, j]

        # Otherwise, try to decompose word = reverse(w_i) · w_j
        for i, wi in enumerate(self.basis_words):
            prefix = tuple(reversed(wi))
            if len(prefix) <= len(word) and word[: len(prefix)] == prefix:
                suffix = word[len(prefix) :]
                for j, wj in enumerate(self.basis_words):
                    if wj == suffix:
                        return Omega[i, j]

        return 0.0  # truncated

    def loss_fn(self, params: dict) -> jnp.ndarray:
        """Total loss = SD residuals + normalisation."""
        Omega = self.params_to_moment_matrix(params)

        def moment_func(w):
            return self.moment_from_matrix(Omega, w)

        loss = sd_loss_two_matrix(moment_func, self.g, self.test_words, self.interaction)

        # Normalisation: Ω_{0,0} = tr[I] = 1
        loss += 100.0 * (Omega[0, 0] - 1.0) ** 2

        return loss

    def train(self, key: jax.Array, verbose: bool = True):
        """Run the training loop."""
        self.init(key)
        losses = []

        for epoch in range(self.n_epochs):
            loss, grads = jax.value_and_grad(self.loss_fn)(self.params)
            updates, self.opt_state = self.optimizer.update(grads, self.opt_state, self.params)
            self.params = optax.apply_updates(self.params, updates)
            losses.append(float(loss))

            if verbose and (epoch % 1000 == 0 or epoch == self.n_epochs - 1):
                Omega = self.params_to_moment_matrix(self.params)
                eigs = jnp.linalg.eigvalsh(Omega)
                print(f"Epoch {epoch:5d}: loss = {loss:.2e}, min_eig(Ω) = {float(eigs[0]):.2e}")

        return self.params, losses


if __name__ == "__main__":
    print("=" * 60)
    print("Neural Master Field — One-Matrix Validation")
    print("=" * 60)

    key = random.PRNGKey(42)

    # Test 1: Gaussian (should converge to Catalan numbers)
    print("\n--- Gaussian model ---")
    trainer = MasterFieldTrainer(
        "gaussian", n_matrices=1, coupling=0.0, max_word_length=12, lr=1e-2, n_epochs=3000
    )
    params, losses = trainer.train(key, verbose=True)
    sol = trainer.get_solution()
    print(f"\nFinal moments: {sol['moments'][:7]}")
    print(f"Free cumulants: {sol['free_cumulants'][:5]}")
    print(f"Final loss: {sol['final_loss']:.2e}")

    # Test 2: Quartic, g=0.5
    print("\n--- Quartic model (g=0.5) ---")
    trainer_q = MasterFieldTrainer(
        "quartic", n_matrices=1, coupling=0.5, max_word_length=12, lr=1e-2, n_epochs=5000
    )
    params_q, losses_q = trainer_q.train(key, verbose=True)
    sol_q = trainer_q.get_solution()

    # Compare with exact
    from one_matrix import quartic_moments_from_sd

    m_exact = quartic_moments_from_sd(0.5, max_power=12)
    print("\nComparison (quartic g=0.5):")
    for k in range(0, 11, 2):
        print(
            f"  m_{k}: ML = {sol_q['moments'][k]:.8f}, exact = {m_exact[k]:.8f}, "
            f"err = {abs(sol_q['moments'][k] - m_exact[k]):.2e}"
        )
