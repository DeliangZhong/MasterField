# M5a — Anharmonic-oscillator QM sandwich: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Squeeze the ground-state energy of `H = p² + x² + g x⁴` between a certified SDP lower bound and a variational (operator-field) upper bound, validated against exact diagonalization — building the reusable QM master-field machinery (oscillator Fock space, the stationarity recursion, the energy-`E` bootstrap, the fail-closed sandwich gate).

**Architecture:** Three layers, reusing the M1–M4 stack. (1) `qm_fock.py`: truncated **bosonic** oscillator Fock space (`[â,â†]=1` ⟹ `[x̂,p̂]=i`), giving the variational ground state (`λ_min` of the truncated `Ĥ`) and the exact-diagonalization referee. (2) `loss.py` + `bootstrap_sdp.py`: the re-derived stationarity recursion (HHK Eq 6) as a numeric residual (verification) and as cvxpy equality constraints; the fixed-`E` Hankel-PSD feasibility SDP and an energy-lower-bound bisection. (3) `train.py`: `solve_qm_anharmonic` assembling the sandwich behind a pure, fail-closed gate `_qm_gate`.

**Tech Stack:** JAX (float64), cvxpy + CLARABEL/MOSEK (reusing `bootstrap_sdp._solve`/`TRUSTED_SOLVERS`), numpy, pytest.

## Global Constraints

- **Conventions (verbatim, HHK arXiv:2004.10212):** `H=p²+x²+gx⁴`, `g≥0`, `ℏ=1`, `[x,p]=+i`. Oscillator rep `x̂=(â+â†)/√2`, `p̂=−i(â−â†)/√2`, `[â,â†]=1`. Moments `m_k=⟨x^k⟩`, `m_0=1`, odd moments `=0` (parity). Spec: `docs/superpowers/specs/2026-06-25-m5a-anharmonic-qm-design.md`.
- **Verified reference values** (from the spec's sanity check — use as test anchors, not guesses): `g=0`: `E₀=1` exact, `m₂=½`, `m₄=¾`. `g=1`: `E₀=1.392352`, `m₂=0.305814`. Recursion D3 residual on exact-diag moments `~1e-13`.
- **Stationarity recursion (D3, re-derived = HHK Eq 6):** `4tE·m_{t-1} + t(t-1)(t-2)·m_{t-3} − 4(t+1)·m_{t+1} − 4g(t+2)·m_{t+3} = 0`.
- **Float64 everywhere:** `jax.config.update("jax_enable_x64", True)` at the top of every JAX file.
- **Variational rigor:** build the truncated `Ĥ` from operators on a **padded** space (`x⁴` reaches `n±4`) then restrict to the `(K+1)` block, so `Ĥ_trunc = P_K H P_K` exactly ⟹ `E_var=λ_min` is a true upper bound, monotone in `K`.
- **Fail-closed:** `validated=True` only if a **certified** (trusted-solver, `optimal`) SDP lower bound and `⟨x²⟩` island bracket the exact-diag answer AND the variational bound is consistent. No certified solver ⟹ `validated=False`.
- **Test runner (portable):** `uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest <path> -v` (run from repo root). CLARABEL ships with cvxpy, so `has_trusted_solver()` is True under this runner.
- **Commit discipline:** commit only the M5a files listed per task; never `git add -A` (the repo carries unrelated untracked postponed-QCD WIP). On branch `matrix-master-field`. End commit messages with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

---

## File Structure

- **Create `matrix_master_field/qm_fock.py`** — bosonic oscillator Fock space; `ladder`, `xp_operators`, `hamiltonian_anharmonic`, `ground_state`, `moment`.
- **Modify `matrix_master_field/loss.py`** — add `qm_anharmonic_recursion_residual(m, E, g)` (numeric D3 residual vector, for verification).
- **Modify `matrix_master_field/bootstrap_sdp.py`** — add `_qm_anharmonic_sdp_constraints`, `qm_anharmonic_feasibility`, `bootstrap_qm_anharmonic`.
- **Modify `matrix_master_field/train.py`** — add `_qm_gate` (pure) and `solve_qm_anharmonic`.
- **Create `matrix_master_field/derivations/m5a-anharmonic-qm.md`** — full D0–D3 derivation + verification recipe.
- **Create tests:** `tests/test_qm_fock.py`, `tests/test_qm_recursion.py`, `tests/test_bootstrap_qm.py`, `tests/test_train_qm.py`.

---

### Task 1: Oscillator Fock space — operators

**Files:**
- Create: `matrix_master_field/qm_fock.py`
- Test: `matrix_master_field/tests/test_qm_fock.py`

**Interfaces:**
- Produces: `ladder(K) -> (a, adag)` and `xp_operators(K) -> (X, P)`, all `(K+1)×(K+1)` complex JAX arrays; `[X,P]=iI` on the interior `K×K` block.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_qm_fock.py
"""M5a — truncated bosonic oscillator Fock space (H=p^2+x^2+g x^4, [x,p]=i)."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from matrix_master_field.qm_fock import ladder, xp_operators


def test_ladder_canonical():
    a, adag = ladder(10)
    comm = np.asarray(a @ adag - adag @ a)
    # [a,adag]=I on the interior (levels 0..K-1); only the top corner is truncated.
    assert np.allclose(np.diag(comm)[:10], 1.0, atol=1e-12)


def test_xp_commutator_interior_is_i():
    K = 30
    X, P = xp_operators(K)
    comm = np.asarray(X @ P - P @ X)
    assert np.allclose(comm[:K, :K], 1j * np.eye(K), atol=1e-12)


def test_xp_hermitian():
    X, P = xp_operators(20)
    assert np.allclose(np.asarray(X), np.conj(np.asarray(X)).T, atol=1e-12)
    assert np.allclose(np.asarray(P), np.conj(np.asarray(P)).T, atol=1e-12)


def test_truncation_corner_is_minus_iK():
    K = 12
    X, P = xp_operators(K)
    comm = np.asarray(X @ P - P @ X)
    assert np.isclose(comm[K, K], -1j * K, atol=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with jax --with numpy --with pytest python -m pytest matrix_master_field/tests/test_qm_fock.py -v`
Expected: FAIL with `ModuleNotFoundError: matrix_master_field.qm_fock`.

- [ ] **Step 3: Write minimal implementation**

```python
# matrix_master_field/qm_fock.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with jax --with numpy --with pytest python -m pytest matrix_master_field/tests/test_qm_fock.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/qm_fock.py matrix_master_field/tests/test_qm_fock.py
git commit -m "feat(mmf): M5a oscillator Fock space — x,p operators with [x,p]=i

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Hamiltonian, variational ground state, moments

**Files:**
- Modify: `matrix_master_field/qm_fock.py`
- Test: `matrix_master_field/tests/test_qm_fock.py`

**Interfaces:**
- Consumes: `xp_operators` (Task 1).
- Produces: `hamiltonian_anharmonic(K, g) -> Ĥ` (Hermitian, exact Galerkin block); `ground_state(K, g) -> (E_var, omega)` (variational upper bound + state vector); `moment(omega, k) -> ⟨x^k⟩` (float).

- [ ] **Step 1: Write the failing test** (append to `test_qm_fock.py`)

```python
from matrix_master_field.qm_fock import ground_state, hamiltonian_anharmonic, moment


def test_g0_ground_state_exact():
    E0, omega = ground_state(40, 0.0)
    assert np.isclose(E0, 1.0, atol=1e-9)
    assert np.isclose(moment(omega, 2), 0.5, atol=1e-9)
    assert np.isclose(moment(omega, 4), 0.75, atol=1e-9)


def test_g0_low_spectrum():
    H = np.asarray(hamiltonian_anharmonic(40, 0.0))
    w = np.sort(np.linalg.eigvalsh(H))
    assert np.allclose(w[:4], [1.0, 3.0, 5.0, 7.0], atol=1e-7)


def test_g1_reference_values():
    E0, omega = ground_state(60, 1.0)
    assert np.isclose(E0, 1.392352, atol=1e-5)   # HHK reference; verified in the spec
    assert np.isclose(moment(omega, 2), 0.305814, atol=1e-5)


def test_variational_upper_bound_monotone():
    Es = [ground_state(K, 1.0)[0] for K in (8, 16, 32, 60)]
    for lo, hi in zip(Es[1:], Es[:-1]):
        assert lo <= hi + 1e-12               # non-increasing in K
    assert Es[-1] >= 1.392352 - 1e-6          # still an upper bound


def test_odd_moments_vanish():
    _, omega = ground_state(40, 1.0)
    assert abs(moment(omega, 1)) < 1e-9
    assert abs(moment(omega, 3)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with jax --with numpy --with pytest python -m pytest matrix_master_field/tests/test_qm_fock.py -k "ground_state or spectrum or reference or variational or odd" -v`
Expected: FAIL — `cannot import name 'hamiltonian_anharmonic'`.

- [ ] **Step 3: Write minimal implementation** (append to `qm_fock.py`)

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with jax --with numpy --with pytest python -m pytest matrix_master_field/tests/test_qm_fock.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/qm_fock.py matrix_master_field/tests/test_qm_fock.py
git commit -m "feat(mmf): M5a Hamiltonian + variational ground state + moments (exact-diag anchor)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Stationarity recursion — derivation + numeric residual + verification

**Files:**
- Create: `matrix_master_field/derivations/m5a-anharmonic-qm.md`
- Modify: `matrix_master_field/loss.py`
- Test: `matrix_master_field/tests/test_qm_recursion.py`

**Interfaces:**
- Consumes: `ground_state`, `moment` (Task 2), `xp_operators` (Task 1).
- Produces: `qm_anharmonic_recursion_residual(m, E, g) -> list[float]` where `m` is a sequence with `m[k]=⟨x^k⟩` (length ≥ 2K+1); returns D3 residuals for `t=1,3,…,len-4`.

- [ ] **Step 1: Write the derivation doc**

Create `matrix_master_field/derivations/m5a-anharmonic-qm.md` with the full D0–D3 derivation copied from the spec's "Derivations" section (D0 canonical commutator; D1 energy relation `⟨x^{t-1}p²⟩=E m_{t-1}−m_{t+1}−g m_{t+3}`; D2 building blocks `[p,x^t]=−it x^{t-1}`, `[H,p]=i(2x+4gx³)`; D3 the boxed recursion), plus the verification recipe: "plug exact-diag moments (`qm_fock.ground_state`/`moment`) into D1 (operator form) and D3 (numeric) — residuals must be `~1e-13` (see `tests/test_qm_recursion.py`)."

- [ ] **Step 2: Write the failing test**

```python
# matrix_master_field/tests/test_qm_recursion.py
"""M5a — the stationarity recursion (HHK Eq 6) holds on exact-diag moments."""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

from matrix_master_field.loss import qm_anharmonic_recursion_residual
from matrix_master_field.qm_fock import ground_state, moment, xp_operators


def _exact_moments(g, kmax, K=80):
    _, omega = ground_state(K, g)
    return [moment(omega, k) for k in range(kmax + 1)]


def test_recursion_residual_vanishes_on_exact_moments():
    for g in (0.0, 0.5, 1.0, 2.0):
        E0, omega = ground_state(80, g)
        m = [moment(omega, k) for k in range(13)]  # need up to m_{2K}, here K=5 -> 10, pad
        res = qm_anharmonic_recursion_residual(m, E0, g)
        assert max(abs(r) for r in res) < 1e-9


def test_recursion_t1_closed_form():
    # t=1: 4E - 8 m2 - 12 g m4 = 0  (m0=1).
    g = 1.0
    E0, omega = ground_state(80, g)
    m2, m4 = moment(omega, 2), moment(omega, 4)
    assert abs(4 * E0 - 8 * m2 - 12 * g * m4) < 1e-9


def test_energy_relation_D1_operator_form():
    # <x^{t-1} p^2> = E m_{t-1} - m_{t+1} - g m_{t+3}, checked with the operators directly.
    g, K = 1.0, 60
    E0, omega = ground_state(K, g)
    X, P = xp_operators(K + 6)
    psi = np.zeros(X.shape[0], dtype=complex)
    psi[: K + 1] = np.asarray(omega)
    P2 = np.asarray(P @ P)
    Xn = np.asarray(X)
    for t in (1, 3, 5):
        lhs = (psi.conj() @ np.linalg.matrix_power(Xn, t - 1) @ P2 @ psi)
        rhs = E0 * moment(omega, t - 1) - moment(omega, t + 1) - g * moment(omega, t + 3)
        assert abs(lhs.real - rhs) < 1e-6
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run --no-project --with jax --with numpy --with pytest python -m pytest matrix_master_field/tests/test_qm_recursion.py -v`
Expected: FAIL — `cannot import name 'qm_anharmonic_recursion_residual'`.

- [ ] **Step 4: Write minimal implementation** (append to `matrix_master_field/loss.py`)

```python
def qm_anharmonic_recursion_residual(m, E, g):
    """Stationarity recursion D3 (HHK Eq 6) for H=p^2+x^2+g x^4, evaluated numerically.

    m[k] = <x^k>, m[0]=1, odd moments 0. Returns the residuals
        4 t E m[t-1] + t(t-1)(t-2) m[t-3] - 4(t+1) m[t+1] - 4 g (t+2) m[t+3]
    for t = 1, 3, ..., len(m)-4 (the t with all indices in range). Zero on the true
    eigenstate moments. See derivations/m5a-anharmonic-qm.md.
    """
    res = []
    for t in range(1, len(m) - 3, 2):
        r = 4.0 * t * E * m[t - 1] - 4.0 * (t + 1) * m[t + 1] - 4.0 * g * (t + 2) * m[t + 3]
        if t >= 3:
            r += t * (t - 1) * (t - 2) * m[t - 3]
        res.append(float(r))
    return res
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run --no-project --with jax --with numpy --with pytest python -m pytest matrix_master_field/tests/test_qm_recursion.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add matrix_master_field/loss.py matrix_master_field/derivations/m5a-anharmonic-qm.md matrix_master_field/tests/test_qm_recursion.py
git commit -m "feat(mmf): M5a stationarity recursion (HHK Eq6) — derivation + verified residual

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Fixed-E feasibility SDP — the ⟨x²⟩ island

**Files:**
- Modify: `matrix_master_field/bootstrap_sdp.py`
- Test: `matrix_master_field/tests/test_bootstrap_qm.py`

**Interfaces:**
- Consumes: `_solve`, `_LAST_SOLVE`, `HAS_CVXPY`, `cp` (existing).
- Produces: `qm_anharmonic_feasibility(g, K, E, with_status=False)` → `(m2_lo, m2_hi)` (the `⟨x²⟩` interval at fixed `E`; each `None` if not optimal), or with status the `(lo_status_tuple, hi_status_tuple)` where each tuple is `(solver, status)`.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_bootstrap_qm.py
"""M5a — the QM bootstrap: fixed-E <x^2> island + certified energy lower bound."""
import os
import numpy as np
import pytest

from matrix_master_field.bootstrap_sdp import (
    HAS_CVXPY,
    bootstrap_qm_anharmonic,
    has_trusted_solver,
    qm_anharmonic_feasibility,
)
from matrix_master_field.qm_fock import ground_state, moment

pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


def test_feasibility_brackets_true_m2_at_exact_energy():
    g, K = 1.0, 6
    E0, omega = ground_state(80, g)
    m2_true = moment(omega, 2)
    m2_lo, m2_hi = qm_anharmonic_feasibility(g, K, E0)
    assert m2_lo is not None and m2_hi is not None
    assert m2_lo - 1e-4 <= m2_true <= m2_hi + 1e-4


def test_feasibility_infeasible_below_ground_energy():
    # E well below E0=1.39 has no positive-moment solution -> at least one edge None.
    m2_lo, m2_hi = qm_anharmonic_feasibility(1.0, 6, 0.5)
    assert m2_lo is None or m2_hi is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with jax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_bootstrap_qm.py -k feasibility -v`
Expected: FAIL — `cannot import name 'qm_anharmonic_feasibility'`.

- [ ] **Step 3: Write minimal implementation** (append to `matrix_master_field/bootstrap_sdp.py`)

```python
def _qm_anharmonic_sdp_constraints(m, E, g, K):
    """cvxpy constraints for H=p^2+x^2+g x^4 at fixed energy E: m_0=1, odd moments 0,
    the stationarity recursion D3 (linear in m for fixed E), and Hankel(m) ⪰ 0.

    `m` is a cvxpy Variable of length 2K+1 (m[k]=<x^k>). With m_0=1 and the K-1
    recursion equalities, the only free moment is m[2]=<x^2>.
    """
    cons = [m[0] == 1.0]
    for k in range(1, 2 * K + 1, 2):
        cons.append(m[k] == 0.0)  # parity: odd moments vanish
    for t in range(1, 2 * K - 1, 2):  # t = 1,3,...,2K-3 ; uses m up to index 2K
        expr = 4.0 * t * E * m[t - 1] - 4.0 * (t + 1) * m[t + 1] - 4.0 * g * (t + 2) * m[t + 3]
        if t >= 3:
            expr = expr + t * (t - 1) * (t - 2) * m[t - 3]
        cons.append(expr == 0.0)
    H = cp.bmat([[m[i + j] for j in range(K + 1)] for i in range(K + 1)])  # Hankel
    cons.append(H >> 0)
    return cons


def qm_anharmonic_feasibility(g, K, E, with_status=False):
    """At fixed energy E, the [min, max] of <x^2> over {recursion(E,g) + Hankel ⪰ 0}.

    Returns (m2_lo, m2_hi); an edge is None if that solve is not optimal. E is feasible
    iff both are not None. With with_status, also returns ((lo_solver, lo_status),
    (hi_solver, hi_status)) for certification.
    """
    if not HAS_CVXPY:
        return (None, None, (None, None), (None, None)) if with_status else (None, None)

    def _edge(maximize):
        m = cp.Variable(2 * K + 1)
        cons = _qm_anharmonic_sdp_constraints(m, E, g, K)
        obj = cp.Maximize(m[2]) if maximize else cp.Minimize(m[2])
        prob = cp.Problem(obj, cons)
        _solve(prob)
        ok = prob.status in ("optimal", "optimal_inaccurate")
        val = float(prob.value) if ok else None
        return val, (_LAST_SOLVE["solver"], _LAST_SOLVE["status"])

    lo, lo_st = _edge(False)
    hi, hi_st = _edge(True)
    if with_status:
        return lo, hi, lo_st, hi_st
    return lo, hi
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with jax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_bootstrap_qm.py -k feasibility -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/bootstrap_sdp.py matrix_master_field/tests/test_bootstrap_qm.py
git commit -m "feat(mmf): M5a fixed-E QM bootstrap — <x^2> island via recursion + Hankel PSD

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Certified energy lower bound — feasibility bisection

**Files:**
- Modify: `matrix_master_field/bootstrap_sdp.py`
- Test: `matrix_master_field/tests/test_bootstrap_qm.py`

**Interfaces:**
- Consumes: `qm_anharmonic_feasibility` (Task 4).
- Produces: `bootstrap_qm_anharmonic(g, K, e_min=0.0, e_max=4.0, tol=1e-5, with_status=False)` → `E_lo` (smallest feasible energy = certified ground-state lower bound), or `None`; with status `(E_lo, solver, status)` from the feasible-edge solve.

- [ ] **Step 1: Write the failing test** (append to `test_bootstrap_qm.py`)

```python
def test_energy_lower_bound_brackets_and_tightens():
    g = 1.0
    E0 = ground_state(80, g)[0]
    E_lo_coarse = bootstrap_qm_anharmonic(g, 3)
    E_lo_fine = bootstrap_qm_anharmonic(g, 7)
    assert E_lo_coarse is not None and E_lo_fine is not None
    assert E_lo_coarse <= E0 + 1e-4               # valid lower bound
    assert E_lo_fine <= E0 + 1e-4
    assert E_lo_fine >= E_lo_coarse - 1e-4        # tightens (rises toward E0) with K


@pytest.mark.skipif(not has_trusted_solver(),
                    reason="certified edge needs CLARABEL/MOSEK")
def test_energy_lower_bound_certified():
    E_lo, solver, status = bootstrap_qm_anharmonic(1.0, 6, with_status=True)
    assert E_lo is not None
    assert solver in ("MOSEK", "CLARABEL") and status == "optimal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with jax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_bootstrap_qm.py -k energy -v`
Expected: FAIL — `cannot import name 'bootstrap_qm_anharmonic'`.

- [ ] **Step 3: Write minimal implementation** (append to `matrix_master_field/bootstrap_sdp.py`)

```python
def bootstrap_qm_anharmonic(g, K, e_min=0.0, e_max=4.0, tol=1e-5, with_status=False):
    """Certified lower bound on the ground-state energy E0(g): the smallest E for which
    {recursion(E,g) + Hankel ⪰ 0} is feasible. Coarse upward scan to bracket the
    feasibility boundary, then bisection. Rises toward E0 as K grows.

    With with_status, returns (E_lo, solver, status) from the feasible-edge solve so the
    caller can require a trusted, 'optimal' certificate.
    """
    if not HAS_CVXPY:
        return (None, None, None) if with_status else None

    def feasible(E):
        lo, hi = qm_anharmonic_feasibility(g, K, E)
        return lo is not None and hi is not None

    import numpy as _np
    grid = list(_np.arange(e_min, e_max + 1e-9, 0.25))
    flags = [feasible(E) for E in grid]
    if not any(flags):
        return (None, None, None) if with_status else None
    first = next(i for i, f in enumerate(flags) if f)
    lo_E = grid[first - 1] if first > 0 else e_min - 0.25
    hi_E = grid[first]  # feasible
    while hi_E - lo_E > tol:
        mid = 0.5 * (lo_E + hi_E)
        if feasible(mid):
            hi_E = mid
        else:
            lo_E = mid
    E_lo = hi_E  # smallest feasible energy
    if with_status:
        _, _, lo_st, hi_st = qm_anharmonic_feasibility(g, K, E_lo, with_status=True)
        # report the weaker of the two edge certificates
        solver, status = lo_st if lo_st[1] != "optimal" else hi_st
        return E_lo, solver, status
    return E_lo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with jax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_bootstrap_qm.py -v`
Expected: 4 passed (2 from Task 4 + 2 here).

Note (solver hygiene): if CLARABEL flags `optimal_inaccurate` near the feasibility boundary, the existing `_CLARABEL_KW` (`max_iter`, `static_regularization_constant`) is the lever — mirror the M3 tuning. The bound stays rigorous regardless of tightness.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/bootstrap_sdp.py matrix_master_field/tests/test_bootstrap_qm.py
git commit -m "feat(mmf): M5a certified energy lower bound via feasibility bisection

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: The sandwich solve + fail-closed gate

**Files:**
- Modify: `matrix_master_field/train.py`
- Test: `matrix_master_field/tests/test_train_qm.py`

**Interfaces:**
- Consumes: `ground_state`, `moment` (qm_fock); `bootstrap_qm_anharmonic`, `qm_anharmonic_feasibility`, `TRUSTED_SOLVERS`, `HAS_CVXPY` (bootstrap_sdp).
- Produces: `_qm_gate(...)` (pure boolean gate) and `solve_qm_anharmonic(g, K, K_sdp=None, validate=True, e_tol=1e-3, m2_tol=1e-3)` → result dict with `E_var, m2_var, E_exact, m2_exact, validation, validated`.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_train_qm.py
"""M5a — the certified sandwich: E_lo <= E0 <= E_var, validated against exact diag."""
import os
import pytest

from matrix_master_field.bootstrap_sdp import HAS_CVXPY, has_trusted_solver
from matrix_master_field.train import _qm_gate, solve_qm_anharmonic


def test_qm_gate_logic_pure():
    base = dict(E_lo=1.30, E_lo_cert=True, m2_island=(0.28, 0.33), m2_island_cert=True,
                E_var=1.3924, E_exact=1.3924, m2_exact=0.3058, e_tol=1e-3, m2_tol=1e-3)
    _, ok = _qm_gate(**base); assert ok is True
    _, ok = _qm_gate(**{**base, "E_lo_cert": False}); assert ok is False     # uncertified
    _, ok = _qm_gate(**{**base, "E_lo": 1.50}); assert ok is False           # lower bound > truth
    _, ok = _qm_gate(**{**base, "E_var": 1.30}); assert ok is False          # bad upper bound
    _, ok = _qm_gate(**{**base, "m2_island": (0.10, 0.20)}); assert ok is False  # m2 outside


@pytest.mark.skipif(not (HAS_CVXPY and has_trusted_solver()),
                    reason="certified sandwich needs CLARABEL/MOSEK")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: bisection over SDP solves; set MMF_SLOW=1")
def test_solve_qm_validated_sandwich():
    r = solve_qm_anharmonic(1.0, K=24, K_sdp=6)
    assert r["validated"] is True
    v = r["validation"]
    assert v["E_lo"] <= r["E_exact"] <= r["E_var"] + 1e-3   # the squeeze
    assert v["certified"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_train_qm.py -k gate -v`
Expected: FAIL — `cannot import name '_qm_gate'`.

- [ ] **Step 3: Write minimal implementation** (append to `matrix_master_field/train.py`; add `from matrix_master_field.qm_fock import ground_state as _qm_ground_state, moment as _qm_moment` and `from matrix_master_field.bootstrap_sdp import bootstrap_qm_anharmonic, qm_anharmonic_feasibility` to the imports)

```python
def _qm_gate(*, E_lo, E_lo_cert, m2_island, m2_island_cert, E_var, E_exact, m2_exact,
             e_tol, m2_tol):
    """Pure fail-closed gate for the M5a sandwich. validated=True iff a CERTIFIED SDP
    lower bound and <x^2> island bracket the exact-diag answer AND the variational
    upper bound is consistent — together certifying E_lo <= E0 <= E_var.
    """
    m2_lo, m2_hi = m2_island
    certified = bool(E_lo_cert and m2_island_cert)
    lower_ok = E_lo is not None and (E_lo - e_tol) <= E_exact
    upper_ok = E_exact <= (E_var + e_tol)
    m2_ok = (m2_lo is not None and m2_hi is not None
             and (m2_lo - m2_tol) <= m2_exact <= (m2_hi + m2_tol))
    validation = {
        "E_lo": E_lo, "E_var": E_var, "E_exact": E_exact,
        "m2_island": m2_island, "m2_exact": m2_exact,
        "lower_ok": lower_ok, "upper_ok": upper_ok, "m2_ok": m2_ok,
        "certified": certified,
    }
    return validation, bool(certified and lower_ok and upper_ok and m2_ok)


def solve_qm_anharmonic(g, K, *, K_sdp=None, validate=True, e_tol=1e-3, m2_tol=1e-3):
    """M5a sandwich for H=p^2+x^2+g x^4: variational upper bound (operator field) +
    certified SDP lower bound, refereed by exact diagonalization. Returns a result dict;
    `validated` is set by the fail-closed `_qm_gate` (needs a trusted SDP solver).
    """
    K_sdp = K_sdp if K_sdp is not None else min(K, 7)
    E_var, omega = _qm_ground_state(K, g)
    m2_var = _qm_moment(omega, 2)
    E_exact, omega_x = _qm_ground_state(max(K, 80), g)   # high-K referee
    m2_exact = _qm_moment(omega_x, 2)
    result = {"g": g, "K": K, "K_sdp": K_sdp, "E_var": E_var, "m2_var": m2_var,
              "E_exact": E_exact, "m2_exact": m2_exact}
    if validate:
        E_lo = E_lo_cert = None
        m2_island = (None, None)
        m2_cert = False
        if HAS_CVXPY:
            E_lo, lo_solver, lo_status = bootstrap_qm_anharmonic(g, K_sdp, with_status=True)
            E_lo_cert = lo_status == "optimal" and lo_solver in TRUSTED_SOLVERS
            m2_lo, m2_hi, lo_st, hi_st = qm_anharmonic_feasibility(
                g, K_sdp, E_exact, with_status=True)
            m2_island = (m2_lo, m2_hi)
            m2_cert = (m2_lo is not None and m2_hi is not None
                       and lo_st[1] == "optimal" and lo_st[0] in TRUSTED_SOLVERS
                       and hi_st[1] == "optimal" and hi_st[0] in TRUSTED_SOLVERS)
        result["validation"], result["validated"] = _qm_gate(
            E_lo=E_lo, E_lo_cert=E_lo_cert, m2_island=m2_island, m2_island_cert=m2_cert,
            E_var=E_var, E_exact=E_exact, m2_exact=m2_exact, e_tol=e_tol, m2_tol=m2_tol)
    return result
```

- [ ] **Step 4: Run gate-logic test, then the slow integration test**

Run (fast gate logic): `uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_train_qm.py -k gate -v`
Expected: 1 passed.

Run (integration, slow): `MMF_SLOW=1 uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_train_qm.py -k sandwich -v`
Expected: 1 passed — `validated is True`, `E_lo ≤ E_exact ≤ E_var`.

- [ ] **Step 5: Run the FULL suite (no regressions)**

Run: `uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/ -q`
Expected: all prior tests still pass; the new M5a tests pass (the two `MMF_SLOW`/cert-gated ones skip without the flag/solver).

- [ ] **Step 6: Commit**

```bash
git add matrix_master_field/train.py matrix_master_field/tests/test_train_qm.py
git commit -m "feat(mmf): M5a sandwich solve + fail-closed gate (certified E_lo <= E0 <= E_var)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## After the plan

- **Update `matrix_master_field/CONVENTIONS.md`** with the QM block (model Eq 1, `ℏ=1`, `[x,p]=i`, oscillator rep, the D3 recursion, `g=0`/`g=1` anchors) — fold into Task 1's commit or a final docs commit.
- **Write the result doc** `docs/superpowers/results/2026-06-25-m5a-anharmonic-qm.md` (sandwich table: `E_lo`, `E_var`, `E_exact` vs `g`; the honest note that the operator field here is textbook variational — novelty is the large-N rungs). Final docs commit.
- **Next:** M5b (single-matrix QM, HHK Eq 8) — own spec, reuses `qm_fock` + the QM SDP extension; then M5c.

## Self-Review

- **Spec coverage:** V1 (interior `[x,p]=i`) → Task 1; V2 (`g=0` exact) → Task 2; V3 (recursion) → Task 3; V4 (D1) → Task 3; V5 (exact-diag convergence) → Task 2 (`test_variational_upper_bound_monotone` + `g1` reference); V6 (SDP brackets) → Task 5; V7 (variational bound) → Task 2; V8 (sandwich) → Task 6; V9 (`⟨x²⟩` island) → Task 4 + Task 6. All covered.
- **Placeholder scan:** none — every code step has complete code; every run step has a command + expected result.
- **Type consistency:** `qm_anharmonic_feasibility` returns `(m2_lo, m2_hi)` (2-tuple) or `(lo, hi, lo_st, hi_st)` (4-tuple, `with_status`); `bootstrap_qm_anharmonic` returns `E_lo` or `(E_lo, solver, status)`. `_qm_gate` consumes `m2_island=(lo,hi)` and the cert flags exactly as `solve_qm_anharmonic` produces them. `ground_state -> (float, vector)`, `moment -> float` used consistently. Consistent across tasks.
- **Risk:** the SDP bisection (Task 5) and CLARABEL accuracy at the feasibility boundary may need the `_CLARABEL_KW` lever (flagged in Task 5); bounds remain rigorous regardless of tightness.
