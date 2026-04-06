"""
schwinger_dyson.py — Loop equations for multi-matrix models at N=∞.

At large N, the Schwinger-Dyson equations become closed recursion relations
for gauge-invariant loop variables (single-trace moments):
    Ω(C) = (1/N) tr[M_{i₁} M_{i₂} ... M_{iₖ}]

The factorisation property ensures multi-trace → products of single-trace.

For a general action S[M₁,...,Mₙ]:
    <(∂S/∂M_i) M_{i₁}...M_{iₖ}> = Σ factorised terms

These are the equations we minimise (as residuals) in the ML optimisation.
"""

from collections.abc import Callable
from itertools import product as cart_product

import numpy as np

# ═══════════════════════════════════════════════════════════
# Word algebra
# ═══════════════════════════════════════════════════════════


def cyclic_reduce(word: tuple[int, ...]) -> tuple[int, ...]:
    """Canonical representative under cyclic permutation."""
    if not word:
        return word
    rotations = [word[i:] + word[:i] for i in range(len(word))]
    return min(rotations)


def all_cyclic_words(n_matrices: int, max_length: int) -> list[tuple[int, ...]]:
    """All distinct single-trace operators up to cyclic equivalence."""
    seen = set()
    words = []
    for length in range(1, max_length + 1):
        for combo in cart_product(range(n_matrices), repeat=length):
            canon = cyclic_reduce(combo)
            if canon not in seen:
                seen.add(canon)
                words.append(canon)
    return words


def word_to_index(words: list[tuple[int, ...]]) -> dict[tuple[int, ...], int]:
    """Map each canonical word to its index in the moment vector."""
    return {cyclic_reduce(w): i for i, w in enumerate(words)}


# ═══════════════════════════════════════════════════════════
# SD equations for specific models
# ═══════════════════════════════════════════════════════════


class SchwingerDysonSystem:
    """Schwinger-Dyson equations for a multi-matrix model.

    The SD equations express:
        <tr[V'_i(M) · w]> = Σ <tr[w_left]> <tr[w_right]>
    where w is a test word and the sum is over all ways to split w.

    At N=∞, factorisation makes these closed in single-trace variables.

    This class generates the full set of SD equations as residual functions
    of the moment vector Ω.
    """

    def __init__(self, n_matrices: int, max_word_length: int):
        self.n = n_matrices
        self.L = max_word_length
        self.words = all_cyclic_words(n_matrices, max_word_length)
        self.word_idx = word_to_index(self.words)
        self.n_vars = len(self.words)

        # Add the empty word (= 1, normalisation)
        # Convention: Ω[empty] = 1 always (not a variable)

    def moment(self, omega: np.ndarray, word: tuple[int, ...]) -> float:
        """Look up the moment Ω(word).
        Empty word → 1. Word not in truncation → 0.
        """
        if not word:
            return 1.0
        canon = cyclic_reduce(word)
        if canon in self.word_idx:
            return omega[self.word_idx[canon]]
        return 0.0  # truncated

    def split_contributions(self, omega: np.ndarray, word: tuple[int, ...]) -> float:
        """Compute Σ_{j=0}^{|w|-1} Ω(w[:j]) · Ω(w[j+1:])

        This is the "splitting" contribution from ∂/∂M_i acting on a word.
        When ∂/∂M_i hits the k-th position in tr[... M_i ...], it splits
        the trace into two traces.

        For a word w = (i₁,...,iₖ) and derivative w.r.t. M_{iₘ} at position m:
        contribution = Ω(i_{m+1}...iₖ i₁...i_{m-1})  [single trace, remaining]

        Wait — the splitting for a single-trace is more subtle. Let me be precise.

        The SD equation from ∂/∂(M_a)_{ij}:

        <Σ_{ij} ∂/∂(M_a)_{ij} [ e^{-S} (M_{i₁}...M_{iₖ})_{ij} ]> = 0

        This gives:
        <tr[∂S/∂M_a · M_{i₁}...M_{iₖ}]> = Σ_{m: iₘ=a} <tr[M_{iₘ₊₁}...M_{iₖ}] tr[M_{i₁}...M_{iₘ₋₁}]>

        The RHS involves products of two shorter traces (factorised at N=∞).
        """
        # This is a general utility; specific models override with their V'
        result = 0.0
        for m in range(len(word)):
            left = word[:m]
            right = word[m + 1 :]
            result += self.moment(omega, left) * self.moment(omega, right)
        return result


class OneMatrixSD(SchwingerDysonSystem):
    """SD equations for V(M) = Σ_k (v_k/k) M^k."""

    def __init__(self, potential_coeffs: list[float], max_word_length: int = 10):
        """
        potential_coeffs: [v_1, v_2, v_3, ...] so that V'(M) = v_1 + v_2 M + v_3 M² + ...
        For V = M²/2 + g M⁴/4:  V' = M + g M³, so potential_coeffs = [0, 1, 0, g]
        """
        super().__init__(n_matrices=1, max_word_length=max_word_length)
        self.v = potential_coeffs

    def sd_residuals(self, omega: np.ndarray) -> np.ndarray:
        """Compute all SD equation residuals.

        SD equation for test word M^n:
        Σ_k v_k m_{n+k} = Σ_{j=0}^{n-1} m_j m_{n-j-1}

        where m_k = Ω((0,)*k) = tr[M^k] and v_k are V' coefficients.
        """

        def m(k):
            if k == 0:
                return 1.0
            word = (0,) * k
            return self.moment(omega, word)

        residuals = []
        max_v_degree = len(self.v) - 1
        for n in range(0, self.L - max_v_degree):
            # LHS: Σ_k v_k m_{n+k}  from tr[V'(M) M^n] = Σ_k v_k tr[M^{k+n}]
            lhs = sum(self.v[k] * m(n + k) for k in range(len(self.v)))

            # RHS: Σ_{j=0}^{n-1} m_j m_{n-j-1}
            rhs = sum(m(j) * m(n - j - 1) for j in range(n))

            residuals.append(lhs - rhs)

        return np.array(residuals)


class TwoMatrixSD(SchwingerDysonSystem):
    """SD equations for two coupled matrices.

    Action: S = N Tr[V₁(M₁) + V₂(M₂) + g W(M₁, M₂)]

    Standard example: V_i = M_i²/2, W = M₁ M₂ M₁ M₂ (Yang-Mills-like)
    or W = (M₁ M₂ - M₂ M₁)² (commutator squared)
    """

    def __init__(
        self, coupling: float, max_word_length: int = 8, interaction: str = "commutator_squared"
    ):
        """
        coupling: coefficient g of the interaction
        interaction: "commutator_squared" for -g Tr[M1,M2]²
                     "quartic_mixed" for g Tr(M1 M2 M1 M2)
        """
        super().__init__(n_matrices=2, max_word_length=max_word_length)
        self.g = coupling
        self.interaction = interaction

    def sd_residuals(self, omega: np.ndarray) -> np.ndarray:
        """Compute SD equation residuals for the two-matrix model.

        For S = Tr[M₁²/2 + M₂²/2 - (g²/4)[M₁,M₂]²]:
        V'₁ = M₁ + (g²/2)(M₁M₂² + M₂²M₁ - 2M₂M₁M₂)
        V'₂ = M₂ + (g²/2)(M₂M₁² + M₁²M₂ - 2M₁M₂M₁)

        SD w.r.t. M_a on test word w = M_{i₁}...M_{iₖ}:
        <tr[V'_a · M_{i₁}...M_{iₖ}]> = Σ_{m:iₘ=a} <tr[w_left]><tr[w_right]>
        """
        residuals = []
        g2 = self.g**2

        # Generate test words up to length L-3 (so V' · test word ≤ L)
        test_words = []
        for length in range(0, max(1, self.L - 3)):
            for combo in cart_product(range(2), repeat=length):
                test_words.append(combo)

        for w in test_words:
            for a in range(2):  # derivative w.r.t. M_0 or M_1
                b = 1 - a  # the other matrix

                # LHS: <tr[V'_a · w]>
                # V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a - 2 M_b M_a M_b)
                # for commutator-squared interaction

                # tr[M_a · w]
                word_a = (a,) + w
                lhs = self.moment(omega, word_a)

                if self.interaction == "commutator_squared":
                    # tr[(M_a M_b² + M_b² M_a - 2 M_b M_a M_b) · w]
                    word1 = (a, b, b) + w
                    word2 = (b, b, a) + w
                    word3 = (b, a, b) + w
                    lhs += (g2 / 2) * (
                        self.moment(omega, word1)
                        + self.moment(omega, word2)
                        - 2 * self.moment(omega, word3)
                    )

                elif self.interaction == "quartic_mixed":
                    # V'_a for S = g Tr(M_a M_b M_a M_b):
                    # ∂/∂M_a Tr(M_a M_b M_a M_b) = M_b M_a M_b + ... (cyclic)
                    word1 = (b, a, b) + w
                    lhs += self.g * 2 * self.moment(omega, word1)

                # RHS: splitting sum
                # When ∂/∂(M_a)_{ij} acts on tr[V'_a · M_{i₁}...M_{iₖ}],
                # the derivative on M_{i₁}...M_{iₖ} gives the splitting.
                rhs = 0.0
                full_word = w  # the test word (V' already accounted for)
                for m in range(len(full_word)):
                    if full_word[m] == a:
                        left = full_word[:m]
                        right = full_word[m + 1 :]
                        rhs += self.moment(omega, left) * self.moment(omega, right)

                residuals.append(lhs - rhs)

        return np.array(residuals[: self.n_vars])  # truncate to # of variables


class LoopMomentMatrix:
    """Construct the loop-space moment matrix Ω_{ij} = tr[w_i† w_j].

    This matrix must be positive semidefinite (PSD) — it encodes the fact
    that the underlying measure is a positive measure on matrices.

    For Hermitian matrices, w† = w reversed, so:
    Ω_{ij} = tr[M_{iₖ}...M_{i₁} M_{j₁}...M_{jₗ}]
    """

    def __init__(self, n_matrices: int, max_word_length: int):
        self.n = n_matrices
        self.L = max_word_length

        # Basis words for the moment matrix (up to length L/2)
        half_L = max_word_length // 2
        self.basis_words = [()]  # empty word
        for length in range(1, half_L + 1):
            for combo in cart_product(range(n_matrices), repeat=length):
                self.basis_words.append(combo)
        self.basis_dim = len(self.basis_words)

    def build_moment_matrix(self, moment_func: Callable) -> np.ndarray:
        """Build the moment matrix given a function that returns moments.

        Args:
            moment_func: word → float, returns tr[M_{i₁}...M_{iₖ}]
        """
        Omega = np.zeros((self.basis_dim, self.basis_dim))
        for i, wi in enumerate(self.basis_words):
            for j, wj in enumerate(self.basis_words):
                # Ω_{ij} = tr[w_i† w_j] = tr[reverse(w_i) · w_j]
                combined = tuple(reversed(wi)) + wj
                Omega[i, j] = moment_func(combined)
        return Omega

    def check_psd(self, moment_func: Callable) -> tuple[bool, float]:
        """Check if moment matrix is PSD. Returns (is_psd, min_eigenvalue)."""
        Omega = self.build_moment_matrix(moment_func)
        eigvals = np.linalg.eigvalsh(Omega)
        min_eig = eigvals[0]
        return min_eig >= -1e-10, min_eig


if __name__ == "__main__":
    print("=" * 60)
    print("Schwinger-Dyson equations test")
    print("=" * 60)

    # Test: Gaussian one-matrix model
    # V(M) = M²/2 → V'(M) = M → potential_coeffs = [0, 1]
    sd = OneMatrixSD(potential_coeffs=[0, 1.0], max_word_length=10)
    print(f"\nGaussian model: {sd.n_vars} loop variables, L={sd.L}")

    # Use exact Gaussian moments
    from one_matrix import gaussian_moments

    m_exact = gaussian_moments(12)

    # Pack into omega vector (words are (0,), (0,0), (0,0,0), ...)
    omega = np.zeros(sd.n_vars)
    for i, w in enumerate(sd.words):
        k = len(w)
        if k <= 12:
            omega[i] = m_exact[k]

    res = sd.sd_residuals(omega)
    print(f"SD residuals (should be ~0): max|residual| = {np.max(np.abs(res)):.2e}")

    # Test: Quartic one-matrix model
    g = 0.5
    sd_q = OneMatrixSD(potential_coeffs=[0, 1.0, 0, g], max_word_length=10)
    from one_matrix import quartic_moments_from_sd

    m_q = quartic_moments_from_sd(g, max_power=14)

    omega_q = np.zeros(sd_q.n_vars)
    for i, w in enumerate(sd_q.words):
        k = len(w)
        if k <= 14:
            omega_q[i] = m_q[k]

    res_q = sd_q.sd_residuals(omega_q)
    print(f"\nQuartic (g={g}): max|residual| = {np.max(np.abs(res_q)):.2e}")
