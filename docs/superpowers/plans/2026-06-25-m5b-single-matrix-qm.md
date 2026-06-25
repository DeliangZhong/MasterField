# M5b — Single-matrix QM sandwich: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline) or superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Squeeze `E/N²` of `H=Tr P²+Tr X²+(g/N)Tr X⁴` between a certified SDP lower bound and a collective-field variational upper bound, with the exact free-fermion solution as referee.

**Architecture:** (A) `qm_collective.py` — the collective-field master field (eigenvalue density `σ(y)`) + finite-N free-fermion referee. (B) derive+verify the stationarity loop equations on free-fermion moments (de-risks the SDP). (C) `bootstrap_sdp.py` — the single-matrix-QM SDP using the verified relations. (D) `train.py` — the sandwich + fail-closed gate. Reuses the M1–M5a stack.

**Tech Stack:** numpy/scipy (collective minimization, finite-N diag), JAX (variational density), cvxpy + MOSEK/CLARABEL (SDP), pytest.

## Global Constraints

- **Conventions:** see `docs/superpowers/specs/2026-06-25-m5b-single-matrix-qm-design.md`. `H=Tr P²+Tr X²+(g/N)Tr X⁴`, `ℏ=1`, `[X_ij,P_kl]=iδ_il δ_jk`, Gauss `⟨Tr XP⟩=iN²/2`, scaling `X=√N X̃`, `m[w]=(1/N)⟨Tr w⟩`, `E/N²=m[P̃²]+m[X̃²]+g·m[X̃⁴]`.
- **Verified anchors (test against these, not guesses):** g=0 → `E/N²=1`, `m[X̃²]=½`, `m[X̃⁴]=½` (exact). g=0.5/1/2 → `E/N²`=1.18049/1.30190/1.48047, `m[X̃²]`=0.37943/0.33143/0.28161, `m[X̃⁴]`=0.28110/0.21301/0.15288.
- **Collective functional:** `E/N²[σ]=∫[π²σ³/3+(y²+g y⁴)σ]dy`, `∫σ=1`, `σ≥0`; analytic minimizer `σ=(1/π)√(μ−y²−g y⁴)`.
- **Float64**; `jax.config.update("jax_enable_x64", True)` atop JAX files.
- **R1 (SDP risk):** derive every stationarity relation and **verify it against the exact free-fermion moments (Task 3) before building the SDP (Task 4)**. The collective/free-fermion half (Tasks 1–2) is verified and low-risk.
- **Runner:** `uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest <path> -v`. **Commit only listed M5b files** (never `-A`). Branch `matrix-master-field`. Trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure
- Create `matrix_master_field/qm_collective.py`; modify `bootstrap_sdp.py`, `train.py`; create `derivations/m5b-single-matrix-qm.md`; tests `test_qm_collective.py`, `test_bootstrap_single_matrix_qm.py`, `test_train_single_matrix_qm.py`.

---

### Task 1: Collective-field master field + free-fermion referee

**Files:** Create `matrix_master_field/qm_collective.py`; Test `tests/test_qm_collective.py`.
**Produces:** `collective_master_field(g) -> dict(mu, energy, m2, m4, density)`, `collective_energy_density(sigma, ys, g) -> float`, `free_fermion_energy(g, N) -> float` (E/N²).

- [ ] **Step 1: failing test**

```python
# tests/test_qm_collective.py
"""M5b — collective-field master field + free-fermion referee for single-matrix QM."""
import numpy as np
from matrix_master_field.qm_collective import (
    collective_master_field, collective_energy_density, free_fermion_energy)

EXACT = {0.0: (1.0, 0.5, 0.5), 0.5: (1.18049, 0.37943, 0.28110),
         1.0: (1.30190, 0.33143, 0.21301), 2.0: (1.48047, 0.28161, 0.15288)}

def test_collective_matches_exact_table():
    for g, (e, m2, m4) in EXACT.items():
        r = collective_master_field(g)
        assert abs(r["energy"] - e) < 1e-4
        assert abs(r["m2"] - m2) < 1e-4
        assert abs(r["m4"] - m4) < 1e-4

def test_g0_exact_energy_one():
    r = collective_master_field(0.0)
    assert abs(r["energy"] - 1.0) < 1e-6 and abs(r["m2"] - 0.5) < 1e-6

def test_free_fermion_converges_to_collective():
    # finite-N level-filling -> collective N=inf value as N grows
    for g in (0.0, 1.0):
        e_inf = collective_master_field(g)["energy"]
        e_N = [free_fermion_energy(g, N) for N in (20, 40, 80)]
        assert abs(e_N[-1] - e_inf) < abs(e_N[0] - e_inf) + 1e-9  # converging
        assert abs(e_N[-1] - e_inf) < 5e-3
    assert abs(free_fermion_energy(0.0, 50) - 1.0) < 1e-12  # g=0 exact at any N
```

- [ ] **Step 2: run → fail** (`ModuleNotFoundError: qm_collective`).

- [ ] **Step 3: implement** (`qm_collective.py`)

```python
"""M5b — collective-field master field of single-matrix QM H=TrP^2+TrX^2+(g/N)TrX^4.

Large-N singlet sector = N free fermions; the master field is the rescaled eigenvalue
density sigma(y) (y=lambda/sqrt N), minimizing E/N^2[sigma]=int[pi^2 sigma^3/3 +
(y^2+g y^4) sigma], int sigma=1. Analytic minimizer sigma=(1/pi)sqrt(mu-V). See
docs/superpowers/specs/2026-06-25-m5b-single-matrix-qm-design.md.
"""
import numpy as np
from scipy import integrate, optimize


def _support_umax(mu, g):  # largest u=y^2 with u + g u^2 = mu
    return (-1 + np.sqrt(1 + 4 * g * mu)) / (2 * g) if g > 0 else mu


def collective_master_field(g):
    """Analytic large-N ground state: sigma=(1/pi)sqrt(mu-y^2-g y^4), mu from int sigma=1."""
    def sig(y, mu):
        v = mu - y ** 2 - g * y ** 4
        return np.sqrt(np.maximum(v, 0.0)) / np.pi

    def norm(mu):
        ym = np.sqrt(_support_umax(mu, g))
        return integrate.quad(lambda y: sig(y, mu), -ym, ym)[0]

    mu = optimize.brentq(lambda m: norm(m) - 1.0, 1e-3, 100.0)
    ym = np.sqrt(_support_umax(mu, g))
    ekin = integrate.quad(lambda y: np.pi ** 2 * sig(y, mu) ** 3 / 3, -ym, ym)[0]
    epot = integrate.quad(lambda y: (y ** 2 + g * y ** 4) * sig(y, mu), -ym, ym)[0]
    m2 = integrate.quad(lambda y: y ** 2 * sig(y, mu), -ym, ym)[0]
    m4 = integrate.quad(lambda y: y ** 4 * sig(y, mu), -ym, ym)[0]
    ys = np.linspace(-ym, ym, 400)
    return {"mu": mu, "energy": ekin + epot, "m2": m2, "m4": m4,
            "ys": ys, "density": sig(ys, mu)}


def collective_energy_density(sigma, ys, g):
    """E/N^2[sigma] = int[pi^2 sigma^3/3 + (y^2+g y^4) sigma] dy (trapezoid)."""
    sigma = np.asarray(sigma); ys = np.asarray(ys)
    integrand = np.pi ** 2 * sigma ** 3 / 3 + (ys ** 2 + g * ys ** 4) * sigma
    return float(np.trapz(integrand, ys))


def free_fermion_energy(g, N, n_basis=None):
    """Finite-N referee: fill N lowest levels of the single-particle h=-d^2/dlam^2 +
    lam^2 + (g/N) lam^4 (oscillator basis), return E/N^2. At g=0, h has levels 2n+1 so
    E=sum_{0}^{N-1}(2n+1)=N^2 exactly -> E/N^2=1.
    """
    M = n_basis or (4 * N + 40)
    n = np.arange(M)
    # x=(a+adag)/sqrt2 in oscillator basis: x^2, x^4 exact matrix elements; h=p^2+x^2+(g/N)x^4
    a = np.zeros((M, M)); a[np.arange(M - 1), np.arange(1, M)] = np.sqrt(np.arange(1, M))
    X = (a + a.T) / np.sqrt(2.0); P = (a - a.T) / np.sqrt(2.0)  # P=-i(a-adag)/sqrt2 -> real antisym *i
    X2 = X @ X
    h = -(P @ P) + X2 + (g / N) * (X2 @ X2)   # -(P_real)^2 = p^2 since P here is the real antisym part of -i(...)
    w = np.sort(np.linalg.eigvalsh(0.5 * (h + h.T)))
    return float(np.sum(w[:N]) / N ** 2)
```

*Note (Step 3 caveat to verify at run):* the single-particle kinetic sign — `p̂=−i(â−â†)/√2` gives `p̂²` real PSD; the snippet builds the real antisymmetric `A=(â−â†)/√2` so `p̂²=−A²` (since `p̂=−iA`, `p̂²=−A²`, and `−A²` is PSD as `A` is antisymmetric). Confirm `free_fermion_energy(0,N)=1` to validate the sign before trusting g>0.

- [ ] **Step 4: run → pass.** `uv run … pytest tests/test_qm_collective.py -v` → 3 passed. (If `free_fermion_energy(0,N)≠1`, fix the kinetic sign per the caveat.)

- [ ] **Step 5: commit** (`qm_collective.py`, `test_qm_collective.py`): `feat(mmf): M5b collective-field master field + free-fermion referee`.

---

### Task 2: Collective variational upper bound (operator master field by minimization)

**Files:** Modify `qm_collective.py`; Test `test_qm_collective.py`.
**Produces:** `collective_variational(g, n_grid=200, steps=2000) -> dict(energy, density, ys)` — minimize `E/N²[σ_θ]` over a positive normalized density ansatz (JAX); `energy ≥` exact, → exact as grid refines.

- [ ] **Step 1: failing test**

```python
def test_collective_variational_upper_bound():
    from matrix_master_field.qm_collective import collective_variational, collective_master_field
    for g in (0.0, 1.0):
        exact = collective_master_field(g)["energy"]
        r = collective_variational(g)
        assert r["energy"] >= exact - 1e-4          # variational: never below the true min
        assert abs(r["energy"] - exact) < 5e-3      # and close
```

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement** — JAX: density on a fixed grid `ys` via `σ = softmax(θ)/Δy` (enforces `σ≥0`, `∫σ≈1`), minimize `collective_energy_density` with optax adam; return the energy + density. (Grid `ys` wide enough to cover the support, e.g. `[-ymax, ymax]` with `ymax` from the analytic support + margin.)

- [ ] **Step 4: run → pass.**

- [ ] **Step 5: commit:** `feat(mmf): M5b collective variational upper bound (density ansatz)`.

---

### Task 3: Derive + verify the stationarity loop equations (de-risk the SDP)

**Files:** Create `derivations/m5b-single-matrix-qm.md`; Modify `loss.py` (add `single_matrix_qm_stationarity_residual(m, g)` — numeric residual of the loop equations on a moment dict); Test `test_bootstrap_single_matrix_qm.py` (verification part).

**Method (in the derivation doc):** EOM `[H,X_ij]=−2iP_ij`, `[H,P_ij]=i(2X+(4g/N)X³)_ji`. For each single-trace `O=Tr(W)`, `⟨[H,Tr W]⟩=0` gives a relation among `m[w]`; reduce `P`-orderings with `[X̃,P̃]` and large-N factorization (HHK Eq 14). Start with: virial `m[P̃²]=m[X̃²]+2g m[X̃⁴]` (`O=Tr(XP)`), and the `O=Tr(X^{2k+1}P)` family closing the `X̃`-moments with `E`.

- [ ] **Step 1: failing test** — assert `single_matrix_qm_stationarity_residual(exact_moments(g), g) < 1e-6` for g∈{0,0.5,1}, where `exact_moments` builds `m[X̃^{2k}]` from `collective_master_field` and `m[P̃²]` from the virial. **This is the gate that the derived relations are correct.**

- [ ] **Step 2: run → fail.**

- [ ] **Step 3: implement** the derived residual; iterate the derivation until the residual vanishes on the exact moments (the doc records the final verified relations).

- [ ] **Step 4: run → pass** (residual < 1e-6 on exact moments).

- [ ] **Step 5: commit:** `feat(mmf): M5b stationarity loop equations — derived + verified on free-fermion moments`.

---

### Task 4: Single-matrix-QM SDP — certified `E/N²` lower bound

**Files:** Modify `bootstrap_sdp.py`; Test `test_bootstrap_single_matrix_qm.py`.
**Produces:** `bootstrap_single_matrix_qm(g, L, *, maximize=False, with_status=False) -> E/N² bound` — moments `m[w]` (words in X̃,P̃ to length L, hermiticity+parity+cyclicity), constraints = the **verified** Task-3 loop equations + Gauss law `m[X̃P̃]=i/2` + large-N factorization (product matrix, as in `_bootstrap_two_matrix`) + Hermitian Gram PSD; objective min/max `E/N²=m[P̃²]+m[X̃²]+g m[X̃⁴]`. Reuse `_solve`/`_LAST_SOLVE`/`TRUSTED_SOLVERS`; use the margin trick if feasibility-edge conditioning bites.

- [ ] **Step 1: failing test** — for g∈{0,0.5,1}: `lb=bootstrap_single_matrix_qm(g,L=4,maximize=False)`, `ub=…maximize=True`; assert `lb ≤ E_exact ≤ ub` (with `E_exact=collective_master_field(g)["energy"]`); island tightens L=3→4; edges certified (skip if no trusted solver). g=0 must bracket 1.0.

- [ ] **Step 2–4:** implement, iterate to green (start L=3 per HHK).

- [ ] **Step 5: commit:** `feat(mmf): M5b single-matrix-QM SDP — certified E/N^2 bounds`.

---

### Task 5: The sandwich solve + fail-closed gate

**Files:** Modify `train.py`; Test `test_train_single_matrix_qm.py`.
**Produces:** `_sm_qm_gate(...)` (pure) and `solve_single_matrix_qm(g, *, L=4, validate=True)` → dict with `E_lo, E_var, E_exact, m2, validated`. `validated` iff a certified SDP island brackets the free-fermion `E_exact` AND `E_var ≥ E_exact` (collective upper bound consistent).

- [ ] **Steps:** pure gate-logic test (synthetic, no solver) + `MMF_SLOW`/cert-gated integration test asserting `validated is True` and `E_lo ≤ E_exact ≤ E_var` at g=1. Implement; commit `feat(mmf): M5b sandwich solve + fail-closed gate`.

---

### Task 6: Result doc + conventions
- [ ] Write `docs/superpowers/results/2026-06-25-m5b-single-matrix-qm.md` (sandwich table; the collective density figure optional); append the M5b row to `CONVENTIONS.md`. Commit `docs(mmf): M5b result + conventions`. Run full suite (no regressions).

## Self-Review
- **Coverage:** V1→T1, V2→T1, V3→T3, V4→T4, V5→T3, V6→T4, V7→T2, V8→T5. All mapped.
- **Risk:** Task 3 gates Task 4 (no SDP until the loop equations verify on exact moments). Task 4 SDP conditioning → margin trick (M5a precedent). Collective/free-fermion (T1–T2) verified and independent.
- **Types:** `collective_master_field`→dict; `free_fermion_energy`→float(E/N²); `bootstrap_single_matrix_qm`→float bound (or `(val,solver,status)`); gate consumes them consistently.
