"""M5a — truncated bosonic oscillator Fock space for single-particle QM.

H = p^2 + x^2 + g x^4, hbar=1, [x,p]=i, with x=(a+adag)/sqrt2, p=-i(a-adag)/sqrt2 and
[a,adag]=1 (so [x,p]=i on the interior). This is the BOSONIC Fock space, distinct from
the M1-M4 free Cuntz-Fock. See docs/superpowers/specs/2026-06-25-m5a-anharmonic-qm-design.md.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def ladder(K):
    """Truncated annihilation/creation matrices on levels 0..K (dim K+1).

    a|n> = sqrt(n)|n-1>, adag|n> = sqrt(n+1)|n+1>, truncated at level K. [a,adag]=I on
    the interior block (levels 0..K-1); the top level is broken by truncation.
    """
    dim = K + 1
    a = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(1, dim)
    a[idx - 1, idx] = np.sqrt(idx)  # a[n-1,n] = sqrt(n)
    return jnp.asarray(a), jnp.asarray(a.conj().T)


def xp_operators(K):
    """Position/momentum on the truncated Fock space: [x,p]=i on the interior."""
    a, adag = ladder(K)
    X = (a + adag) / jnp.sqrt(2.0)
    P = -1j * (a - adag) / jnp.sqrt(2.0)
    return X, P


def hamiltonian_anharmonic(K, g, pad=4):
    """H = P^2 + X^2 + g X^4 as the EXACT Galerkin compression onto levels 0..K.

    Build the operators on a padded space (X^4 connects |n> to |n +/- 4|) and restrict
    to the (K+1) block, so H_trunc = P_K H P_K exactly. Then lambda_min(H_trunc) is a
    rigorous variational upper bound to the true E0(g), monotone-decreasing in K.
    """
    X, P = xp_operators(K + pad)
    X2 = X @ X
    H = P @ P + X2 + g * (X2 @ X2)
    H = H[: K + 1, : K + 1]
    return 0.5 * (H + jnp.conj(H).T)  # symmetrize residual numerical noise


def ground_state(K, g):
    """Variational ground state: (E_var, |Omega>) = lowest eigenpair of the truncated H.

    E_var >= E0(g) (Rayleigh-Ritz), converging down to E0 as K -> inf.
    """
    H = hamiltonian_anharmonic(K, g)
    w, v = jnp.linalg.eigh(H)
    return float(jnp.real(w[0])), v[:, 0]


def moment(omega, k, pad=4):
    """<Omega| x^k |Omega> for state vector omega (length K+1), padded for accuracy."""
    K = omega.shape[0] - 1
    X, _ = xp_operators(K + pad)
    psi = jnp.zeros(X.shape[0], dtype=jnp.complex128).at[: K + 1].set(omega)
    Xk = jnp.linalg.matrix_power(X, k)
    return float(jnp.real(jnp.conj(psi) @ Xk @ psi))
