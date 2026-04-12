"""tek.py — Core TEK model: clock matrix, twist tensor, link matrices, action.

TEK reduction: D-dimensional SU(N) lattice YM at N=∞ equivalent to D unitary
N×N matrices at a single site with twisted plaquette action.

Sign convention (R3):
    Z = ∫ dU exp(−S)
    S = −(β/N) Σ_{μ<ν} Re Tr(z_μν U_μ U_ν U_μ† U_ν†)
    β = 2N/λ  (so λ = g²N is the 't Hooft coupling)

The loss we MINIMIZE in `tek_loss` is the normalized action per plaquette
(β and N absorbed — they only rescale the gradient):

    L(H) = −(1/N_pairs) Σ_{μ<ν} Re[z_μν · Tr(U_μ U_ν U_μ† U_ν†)] / N

Minimizing L maximizes the plaquette, i.e., drives the system toward the
ordered saddle at weak coupling. λ enters only through the coupling schedule
used in continuation; it does not appear inside this function.

Ansatz (R1): U_1 = Γ (gauge-fixed); U_μ = Ω_μ Γ Ω_μ† with Ω_μ = expm(i H_μ),
H_μ Hermitian, for μ = 2, …, D. Γ = diag(1, ω, ω², …, ω^{N-1}) is the clock
matrix (eigenvalues = exact N-th roots of unity). This assumes center symmetry
is unbroken at the saddle, which is safe for D=2, D=3 symmetric flux, and for
D=4 with a modified flux choice (see reference/tek_master_field.md).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax.scipy.linalg import expm  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# Clock matrix
# ═══════════════════════════════════════════════════════════════


def build_clock_matrix(N: int) -> jnp.ndarray:
    """TEK clock matrix  Γ = P_L ⊗ I_L  where N = L² and P_L is the L×L clock:
    P_L = diag(1, ω_L, ω_L², …, ω_L^{L-1}),  ω_L = exp(2πi/L).

    Eigenvalues of Γ are the **L-th roots of unity, each with multiplicity L**.
    Satisfies Γ^L = I (stronger than Γ^N = I), Γ Γ† = I. Trace(Γ) = 0.

    This matches the TEK twist-eater structure (arXiv:1708.00841 §2.2, eq.
    2.16). The coadjoint orbit {Ω Γ Ω†} CONTAINS the TEK classical saddle
    (Q_L ⊗ P_L and related twist eaters), so the orientation-only
    parametrization U_μ = Ω_μ Γ Ω_μ† can, in principle, reach it.

    Note: an earlier version used Γ = diag(N-th roots) (non-degenerate). That
    spectrum differs from any TEK saddle at finite N and is unreachable via
    unitary conjugation. The tensor-product form is the correct TEK choice
    (see reference/tek_master_field.md §"Ansatz Caveat (R5)").
    """
    L = int(round(N**0.5))
    if L * L != N:
        raise ValueError(f"N must be a perfect square (N = L²); got N={N}")
    jk = jnp.arange(L)
    phases = jnp.exp(2j * jnp.pi * jk / L)
    P_L = jnp.diag(phases).astype(jnp.complex128)
    I_L = jnp.eye(L, dtype=jnp.complex128)
    return jnp.kron(P_L, I_L)


# ═══════════════════════════════════════════════════════════════
# Twist tensor
# ═══════════════════════════════════════════════════════════════


def build_twist(D: int, N: int, L: int, k: int = 1) -> jnp.ndarray:
    """Symmetric-twist phase tensor z_μν = exp(2πi n_μν / N).

    N = L² with L prime. The flux integer k controls the magnetic flux through
    each twisted plane: n_μν = k·L on twisted planes, 0 otherwise.

    D=2: single plane, n₁₂ = k·L. Safe for any k coprime to L.
    D=3: single plane, n₁₂ = k·L. Safe.
    D=4: two planes, n₁₂ = n₃₄ = k·L. Center-symmetry caveat (R1):
        k=1 is known to break center symmetry at N ≥ 100 (hep-th/0612097).
        González-Arroyo–Okawa 2010 (arXiv:1005.1981) recommend k near L/2.

    Returns a (D,D) complex128 antisymmetric-phase tensor with z[μ,μ] = 1.
    """
    if N != L * L:
        raise ValueError(f"Expected N = L², got N={N}, L={L} (L²={L*L})")
    if D not in (2, 3, 4):
        raise ValueError(f"D must be 2, 3, or 4; got {D}")

    phase = jnp.exp(2j * jnp.pi * k / L)
    z = jnp.ones((D, D), dtype=jnp.complex128)

    if D == 2 or D == 3:
        # Single twisted plane: (1, 2)
        z = z.at[0, 1].set(phase)
        z = z.at[1, 0].set(jnp.conj(phase))
    elif D == 4:
        # Two twisted planes: (1, 2) and (3, 4)
        z = z.at[0, 1].set(phase)
        z = z.at[1, 0].set(jnp.conj(phase))
        z = z.at[2, 3].set(phase)
        z = z.at[3, 2].set(jnp.conj(phase))

    return z


# ═══════════════════════════════════════════════════════════════
# Hermitian projection
# ═══════════════════════════════════════════════════════════════


def hermitianize(H: jnp.ndarray) -> jnp.ndarray:
    """Project a complex matrix onto its Hermitian part: (H + H†)/2."""
    return 0.5 * (H + jnp.conj(H.T))


# ═══════════════════════════════════════════════════════════════
# Link matrices
# ═══════════════════════════════════════════════════════════════


def build_link_matrices(
    H_list: list[jnp.ndarray],
    Gamma: jnp.ndarray,
) -> list[jnp.ndarray]:
    """Build the D link matrices from (D-1) Hermitian generators.

    U_1 = Γ (gauge-fixed)
    U_μ = expm(i H_{μ-1}) · Γ · expm(-i H_{μ-1})   for μ = 2, …, D

    H_list must contain (D-1) Hermitian matrices; we do NOT re-Hermitianize
    here — the caller is responsible (project before each gradient step).

    The result is a list of D unitary matrices. Unitarity of each U_μ follows
    from Γ Γ† = I and from exp(iH) being unitary when H is Hermitian.
    """
    U: list[jnp.ndarray] = [Gamma]
    for H in H_list:
        Omega = expm(1j * H)
        Omega_dag = jnp.conj(Omega.T)
        U_mu = Omega @ Gamma @ Omega_dag
        U.append(U_mu)
    return U


# ═══════════════════════════════════════════════════════════════
# TEK action / loss
# ═══════════════════════════════════════════════════════════════


def _plaquette_traces(U: list[jnp.ndarray], z: jnp.ndarray, D: int) -> jnp.ndarray:
    """Compute the D(D-1)/2 plaquette values p_{μν} = Re[z_μν Tr(U U U† U†)] / N.

    Returns a 1-D array of real-valued plaquettes, one per ordered pair μ<ν.
    """
    N = U[0].shape[0]
    vals = []
    for mu in range(D):
        for nu in range(mu + 1, D):
            Umu = U[mu]
            Unu = U[nu]
            plaq = Umu @ Unu @ jnp.conj(Umu.T) @ jnp.conj(Unu.T)
            vals.append(jnp.real(z[mu, nu] * jnp.trace(plaq)) / N)
    return jnp.stack(vals)


def build_link_matrices_full(M_list: list[jnp.ndarray]) -> list[jnp.ndarray]:
    """Full U(N) parametrization:  U_μ = expm(i M_μ)  for μ = 1, …, D.

    No clock-matrix reference: each U_μ is an arbitrary unitary, determined by
    a Hermitian M_μ. Total parameters: D · N² real. The eigenvalues of U_μ are
    no longer locked to the L-th roots of unity, so this ansatz can represent
    center-symmetry-breaking configurations (e.g., fluxons) that the
    orientation-only ansatz cannot.

    No gauge fixing — the global gauge freedom U_μ → g U_μ g† leaves the loss
    invariant, producing a flat N² − 1-dimensional direction at every point.
    Adam tolerates this overparametrization; projection to a gauge slice (e.g.,
    M_1 = 0) would reduce params to (D−1) N² but is not required.
    """
    return [expm(1j * M) for M in M_list]


def tek_loss_full(
    M_list: list[jnp.ndarray],
    z: jnp.ndarray,
    D: int,
) -> jnp.ndarray:
    """TEK loss under the full U(N) ansatz. Same sign convention as tek_loss
    (negative of the average plaquette)."""
    U = build_link_matrices_full(M_list)
    plaqs = _plaquette_traces(U, z, D)
    n_pairs = plaqs.shape[0]
    return -jnp.sum(plaqs) / n_pairs


def plaquette_average_full(
    M_list: list[jnp.ndarray],
    z: jnp.ndarray,
    D: int,
) -> jnp.ndarray:
    """Mean plaquette under the full ansatz. Physical observable."""
    return -tek_loss_full(M_list, z, D)


def tek_loss(
    H_list: list[jnp.ndarray],
    Gamma: jnp.ndarray,
    z: jnp.ndarray,
    D: int,
) -> jnp.ndarray:
    """TEK loss = −(1/N_pairs) Σ_{μ<ν} Re[z_μν Tr(U_μ U_ν U_μ† U_ν†)] / N.

    Minimizing this maximizes the plaquette (correct saddle for exp(−S)).

    NOTE: λ does NOT appear here. The coupling enters the optimization only
    via the schedule in `coupling_continuation` (determining which λ-value we
    are currently extremizing at) and via the noise structure of the gradient
    landscape. Strictly, for a single λ the saddle is independent of λ in this
    functional form — the saddle IS the global maximizer of Σ Re plaquette.
    The coupling dependence emerges from the measure, which we do not sample
    here; we find the master-field saddle directly.
    """
    U = build_link_matrices(H_list, Gamma)
    plaqs = _plaquette_traces(U, z, D)
    n_pairs = plaqs.shape[0]
    return -jnp.sum(plaqs) / n_pairs


def plaquette_average(
    H_list: list[jnp.ndarray],
    Gamma: jnp.ndarray,
    z: jnp.ndarray,
    D: int,
) -> jnp.ndarray:
    """Mean plaquette value = (1/N_pairs) Σ_{μ<ν} Re[z_μν Tr(U U U† U†)] / N.

    This is the physical observable typically reported in MC. It equals −tek_loss.
    """
    return -tek_loss(H_list, Gamma, z, D)


# ═══════════════════════════════════════════════════════════════
# Initialization helpers
# ═══════════════════════════════════════════════════════════════


def init_H_list_zero(D: int, N: int) -> list[jnp.ndarray]:
    """Initialize H_μ = 0 for μ = 2, …, D (all U_μ = Γ, strong-coupling start).

    At H=0 every link equals Γ (block-diagonal kron(P_L, I_L)), and since all
    these Γ copies commute, U_μ U_ν U_μ† U_ν† = I, so plaquette_{μν} = Re(z_μν).
    The plaquette average is Σ_{μ<ν} Re(z_μν) / N_pairs — deterministic.
    """
    return [jnp.zeros((N, N), dtype=jnp.complex128) for _ in range(D - 1)]


def init_H_list_random(D: int, N: int, key, scale: float = 0.1) -> list[jnp.ndarray]:
    """Initialize H_μ as small random Hermitian matrices.

    Used to break symmetry if starting from H=0 gives a pathological gradient.
    """
    from jax import random

    out: list[jnp.ndarray] = []
    for _ in range(D - 1):
        key, subkey = random.split(key)
        re = random.normal(subkey, (N, N))
        key, subkey = random.split(key)
        im = random.normal(subkey, (N, N))
        H = (re + 1j * im).astype(jnp.complex128) * scale
        out.append(hermitianize(H))
    return out


def init_M_list_zero(D: int, N: int) -> list[jnp.ndarray]:
    """Initialize M_μ = 0 for μ = 1, …, D (full ansatz; all U_μ = I).

    At M = 0 every U_μ = I, so plaquettes are trivially I and
    plaquette_{μν} = Re(z_μν) (same initial loss as orientation H=0).
    """
    return [jnp.zeros((N, N), dtype=jnp.complex128) for _ in range(D)]


def init_M_list_random(D: int, N: int, key, scale: float = 0.1) -> list[jnp.ndarray]:
    """Initialize M_μ as small random Hermitian matrices (full ansatz).

    Used to break the zero-gradient symmetry at M=0 and to break center
    symmetry at initialization if desired.
    """
    from jax import random

    out: list[jnp.ndarray] = []
    for _ in range(D):
        key, subkey = random.split(key)
        re = random.normal(subkey, (N, N))
        key, subkey = random.split(key)
        im = random.normal(subkey, (N, N))
        M = (re + 1j * im).astype(jnp.complex128) * scale
        out.append(hermitianize(M))
    return out


# ═══════════════════════════════════════════════════════════════
# Smoke test entry point
# ═══════════════════════════════════════════════════════════════


def _smoke_test() -> None:
    """Smoke test: at H=0, plaquette = mean of Re(z_μν) over ordered pairs."""
    for D, N, L in [(2, 9, 3), (2, 49, 7), (3, 25, 5), (4, 49, 7)]:
        Gamma = build_clock_matrix(N)
        z = build_twist(D, N, L, k=1)
        H_list = init_H_list_zero(D, N)
        plaq = float(plaquette_average(H_list, Gamma, z, D))

        # Expected: at H=0, every U_μ = Γ, so U U U† U† = I, Tr = N, plaq_{μν} = Re(z_μν).
        z_vals = [float(jnp.real(z[mu, nu])) for mu in range(D) for nu in range(mu + 1, D)]
        expected = sum(z_vals) / len(z_vals)
        err = abs(plaq - expected)
        print(
            f"  D={D} N={N} L={L}: plaq={plaq:.10f}  expected={expected:.10f}  err={err:.2e}  "
            f"{'✓' if err < 1e-12 else '✗'}"
        )
        assert err < 1e-12, f"Smoke test failed for D={D} N={N}: err={err:.3e}"

    # Clock matrix: Γ^L = I (stronger than Γ^N = I; TEK-style tensor-product Γ)
    for N, L in ((9, 3), (49, 7), (121, 11)):
        Gamma = build_clock_matrix(N)
        Gamma_L = jnp.linalg.matrix_power(Gamma, L)
        eye = jnp.eye(N, dtype=jnp.complex128)
        err = float(jnp.linalg.norm(Gamma_L - eye))
        print(f"  Clock matrix N={N} L={L}: ||Γ^L − I||_F = {err:.2e}  {'✓' if err < 1e-10 else '✗'}")
        assert err < 1e-10, f"Clock matrix failed for N={N}"

    # Twist tensor antisymmetry: z[μ,ν] = conj(z[ν,μ])
    z = build_twist(D=4, N=49, L=7, k=1)
    for mu in range(4):
        for nu in range(4):
            err = float(abs(z[mu, nu] - jnp.conj(z[nu, mu])))
            assert err < 1e-14, f"Twist antisymmetry failed at ({mu},{nu}): err={err:.3e}"
    print("  Twist antisymmetry D=4: ✓")

    print("\nAll tek.py smoke tests passed.")


if __name__ == "__main__":
    print("=" * 60)
    print("  TEK core — smoke tests")
    print("=" * 60)
    _smoke_test()
