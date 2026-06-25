# matrix_master_field/tests/test_qm_master_field.py
"""M5c — Gaussian master field (C2) + free-Fisher operator field (C3)."""
import numpy as np

from matrix_master_field.qm_master_field import (
    gaussian_comm_moment,
    gaussian_master_field,
)


def test_gaussian_anchor_is_2m():
    for m in (0.5, 1.0, 2.0):
        r = gaussian_master_field(m, 0.0)
        assert abs(r["energy"] - 2.0 * m) < 1e-9       # λ=0 → exact ground state
        assert abs(r["omega"] - m) < 1e-9              # Ω=m
        assert abs(r["m2"] - 1.0 / (2.0 * m)) < 1e-9   # m[X̃²]=1/(2Ω)=1/(2m)


def test_gaussian_is_upper_bound_increasing_in_lambda():
    base = gaussian_master_field(1.0, 0.0)["energy"]
    prev = base
    for lam in (0.5, 1.0, 2.0):
        e = gaussian_master_field(1.0, lam)["energy"]
        assert e > base                 # confining term raises the energy
        assert e >= prev - 1e-12        # monotone in λ
        prev = e


def test_gaussian_satisfies_its_cubic():
    # Ω minimizes Ω+m²/Ω+λ/(2Ω²) ⇒ Ω³ − m²Ω − λ = 0.
    m, lam = 1.0, 1.5
    om = gaussian_master_field(m, lam)["omega"]
    assert abs(om**3 - m**2 * om - lam) < 1e-7


def test_lambda_normalization_nonzero():
    # F3: the λ>0 interaction normalization — the λ=0 anchor CANNOT test it. With X=√N X̃,
    # g²=λ/N, the energy shift −g²⟨Tr[X,Y]²⟩/N² → λ/(2Ω²) as N→∞ (pins coefficient + N-powers).
    omega, lam = 1.3, 0.7
    target = lam / (2.0 * omega**2)
    prev_err = None
    for N in (20, 80, 320):
        shift = -(lam / N) * gaussian_comm_moment(omega, N) / N**2
        err = abs(shift - target)
        if prev_err is not None:
            assert err < prev_err            # monotone convergence to λ/(2Ω²)
        prev_err = err
    assert prev_err < 1e-2                    # close at N=320 (err = target/N²)


def test_wick_commutator_moment_matches_sampling():
    # ⟨Tr[X,Y]²⟩ ≈ −N³/(2Ω²) for independent Gaussian Hermitian X,Y, ⟨X_ij X_kl⟩=δ_il δ_jk/(2Ω).
    rng = np.random.default_rng(0)
    N, omega, trials = 12, 1.0, 4000
    a = 1.0 / (2.0 * omega)
    vals = []
    for _ in range(trials):
        X = _gauss_herm(rng, N, a)
        Y = _gauss_herm(rng, N, a)
        C = X @ Y - Y @ X
        vals.append(np.trace(C @ C).real)
    assert abs(np.mean(vals) - gaussian_comm_moment(omega, N)) < 0.06 * abs(gaussian_comm_moment(omega, N))


def _gauss_herm(rng, N, a):
    # Hermitian with ⟨X_ij X_kl⟩ = a δ_il δ_jk (variance a per complex off-diag pair, a on diag).
    Re = rng.normal(size=(N, N))
    Im = rng.normal(size=(N, N))
    M = Re + 1j * Im
    H = (M + M.conj().T) / 2.0
    return np.sqrt(a) * H
