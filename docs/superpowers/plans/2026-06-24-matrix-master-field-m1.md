# Matrix Master Field — Milestone 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `matrix_master_field/` package and prove the operator (Cuntz–Fock) representation reproduces the known one-matrix master field to ≤1e-6, with rigorous SDP bounds as an independent check.

**Architecture:** Port the already-validated one-matrix reference code (exact resolvents, moments, free cumulants), the Cuntz–Fock operator space, and the SDP bootstrap into a clean new package. Then compose them: free cumulants → Voiculescu operator → vacuum moments, validated against the exact answer. No new physics here — this de-risks the foundation before the optimization engine (Milestone 2).

**Tech Stack:** Python, NumPy, SciPy, cvxpy (SDP). JAX/optax arrive in Milestone 2.

## Global Constraints

- **Run Python via the project uv environment:** `uv run python ...` (if a package is missing, `uv run --with numpy --with scipy --with cvxpy python ...`). Resolve the exact runner in Task 1, Step 1.
- **Float64 everywhere** in any JAX code (`jax.config.update("jax_enable_x64", True)` at file top). (No JAX in M1, but the rule stands.)
- **Hermitian** matrices; **m₀ = m_∅ = 1** is a hard constraint, never optimized.
- **Conventions:** moments `m_k = lim_{N→∞}(1/N)⟨tr Mᵏ⟩`; one-matrix potentials V=½M² and ½M²+(g/4)M⁴ so V′=M(+gM³). Full conventions in `docs/superpowers/specs/2026-06-24-matrix-master-field-design.md` §5.
- **Never guess physics conventions.** (The Kazakov–Zheng action is a Milestone-4 concern; not touched here.)
- Source of ports: the validated files in `master_field/` (no QCD coupling in the four files we port).

---

### Task 1: Package scaffold + port exact one-matrix reference

**Files:**
- Create: `matrix_master_field/__init__.py`
- Create: `matrix_master_field/one_matrix.py` (port of `master_field/one_matrix.py`)
- Create: `matrix_master_field/CONVENTIONS.md`
- Test: `matrix_master_field/tests/__init__.py`, `matrix_master_field/tests/test_one_matrix.py`

**Interfaces:**
- Produces: `one_matrix.gaussian_moments(max_power) -> np.ndarray`; `one_matrix.quartic_moments_from_sd(g, max_power) -> np.ndarray`; `one_matrix.quartic_eigenvalue_density(g, n_points) -> (x, rho)`; `one_matrix.moments_from_density(x, rho, max_power) -> np.ndarray`; `one_matrix.r_transform_from_moments(moments) -> kappa`; `one_matrix.voiculescu_coefficients(kappa) -> np.ndarray`.

- [ ] **Step 1: Resolve the test runner**

Run: `cd "/Users/deliangzhong/Documents/Working/Master Field" && uv run python -c "import numpy, scipy; print('ok')"`
Expected: `ok`. If it fails, use `uv run --with numpy --with scipy --with cvxpy python ...` for all subsequent commands and note that in the commit message.

- [ ] **Step 2: Create the package skeleton**

```bash
cd "/Users/deliangzhong/Documents/Working/Master Field"
mkdir -p matrix_master_field/tests
printf '"""matrix_master_field — operator master field for matrix models."""\n' > matrix_master_field/__init__.py
printf '' > matrix_master_field/tests/__init__.py
```

- [ ] **Step 3: Port the one-matrix reference verbatim**

Copy `master_field/one_matrix.py` to `matrix_master_field/one_matrix.py` unchanged (it imports only `numpy`, `scipy.optimize.brentq`, `math`; no intra-package or QCD imports). Keep the `if __name__ == "__main__"` block.

```bash
cp master_field/one_matrix.py matrix_master_field/one_matrix.py
```

- [ ] **Step 4: Write `CONVENTIONS.md`**

```markdown
# Conventions — matrix_master_field

- Hermitian N×N matrices; action S = N·Tr[…]; couplings are 't Hooft (fixed as N→∞).
- Moments m_w = lim_{N→∞}(1/N)⟨tr M_{w₁}…M_{w_k}⟩, m_∅ = 1 (hard).
- One-matrix: V=½M² (V′=M); V=½M²+(g/4)M⁴ (V′=M+gM³).
- Two-matrix (commutator+mass): S = N·tr[½(M₁²+M₂²) − (λ/4)[M₁,M₂]²], λ>0.
- Kazakov–Zheng action: TRANSCRIBE from arXiv:2108.04830 §2 before Milestone 4 — do not guess.
- Cuntz–Fock: a_i a†_j = δ_ij; vacuum |Ω⟩; tracial state τ=⟨Ω|·|Ω⟩ with cyclicity imposed.
- Float64 in all JAX code.
```

- [ ] **Step 5: Write the failing regression tests**

```python
# matrix_master_field/tests/test_one_matrix.py
import numpy as np
from matrix_master_field import one_matrix as om


def test_gaussian_moments_are_catalan():
    m = om.gaussian_moments(10)
    # m_{2k} = Catalan C_k: 1, 1, 2, 5, 14, 42
    assert np.isclose(m[0], 1.0)
    assert np.isclose(m[2], 1.0)
    assert np.isclose(m[4], 2.0)
    assert np.isclose(m[6], 5.0)
    assert np.isclose(m[8], 14.0)
    assert np.isclose(m[10], 42.0)
    # odd moments vanish
    assert np.allclose(m[1:11:2], 0.0)


def test_gaussian_free_cumulants():
    kappa = om.r_transform_from_moments(om.gaussian_moments(10))
    # Gaussian/semicircle: kappa_2 = 1, all others 0
    assert np.isclose(kappa[1], 0.0, atol=1e-9)  # kappa_1
    assert np.isclose(kappa[2], 1.0, atol=1e-9)  # kappa_2
    assert np.allclose(kappa[3:], 0.0, atol=1e-9)


def test_quartic_sd_moments_match_density_moments():
    g = 0.5
    m_sd = om.quartic_moments_from_sd(g, max_power=8)
    x, rho = om.quartic_eigenvalue_density(g, n_points=4000)
    m_rho = om.moments_from_density(x, rho, 8)
    for k in range(0, 9, 2):
        assert abs(m_sd[k] - m_rho[k]) < 1e-3, f"m_{k}: {m_sd[k]} vs {m_rho[k]}"
```

- [ ] **Step 6: Run the tests**

Run: `uv run python -m pytest matrix_master_field/tests/test_one_matrix.py -v`
Expected: all 3 PASS (the ported code is already validated; this pins it as regression truth). If any fail, the port was altered — re-copy in Step 3.

- [ ] **Step 7: Commit**

```bash
git add matrix_master_field/__init__.py matrix_master_field/one_matrix.py matrix_master_field/CONVENTIONS.md matrix_master_field/tests/
git commit -m "feat(mmf): scaffold package + port exact one-matrix reference with regression tests"
```

---

### Task 2: Port the Cuntz–Fock operator space

**Files:**
- Create: `matrix_master_field/cuntz_fock.py` (port of `master_field/cuntz_fock.py`)
- Test: `matrix_master_field/tests/test_cuntz_fock.py`

**Interfaces:**
- Produces: `cuntz_fock.CuntzFockSpace(n_matrices, max_length)` with `.x(i)`, `.a(i)`, `.adag(i)`, `.vev(op)`, `.verify_cuntz_relations(tol)`, `.build_master_field_voiculescu(coeffs, matrix_idx=0)`, `.compute_moments(M_hat, max_power)`, `.compute_mixed_moments(operators, word)`.

- [ ] **Step 1: Port verbatim**

Copy `master_field/cuntz_fock.py` to `matrix_master_field/cuntz_fock.py` unchanged (imports only `numpy`, `itertools`; no intra-package or QCD imports).

```bash
cp master_field/cuntz_fock.py matrix_master_field/cuntz_fock.py
```

- [ ] **Step 2: Write the failing tests**

```python
# matrix_master_field/tests/test_cuntz_fock.py
import numpy as np
from matrix_master_field.cuntz_fock import CuntzFockSpace


def test_cuntz_relations_hold_in_interior():
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    # verify_cuntz_relations prints and returns True if interior relations hold
    assert fock.verify_cuntz_relations(tol=1e-12) is True


def test_gaussian_operator_gives_catalan():
    fock = CuntzFockSpace(n_matrices=1, max_length=8)
    M = fock.x(0)  # a + a† = free semicircular
    m = fock.compute_moments(M, max_power=8)
    for k, cat in [(0, 1), (2, 1), (4, 2), (6, 5), (8, 14)]:
        assert np.isclose(m[k], cat), f"tr[M^{k}]={m[k]} expected {cat}"


def test_free_product_two_semicirculars():
    fock = CuntzFockSpace(n_matrices=2, max_length=4)
    M1, M2 = fock.x(0), fock.x(1)
    # For free semicirculars (unit variance): tr[M1 M2 M1 M2] = 1
    val = fock.vev(M1 @ M2 @ M1 @ M2)
    assert np.isclose(val, 1.0, atol=1e-10), f"got {val}"
```

- [ ] **Step 3: Run the tests**

Run: `uv run python -m pytest matrix_master_field/tests/test_cuntz_fock.py -v`
Expected: all 3 PASS (these mirror the file's own `__main__` validation).

- [ ] **Step 4: Commit**

```bash
git add matrix_master_field/cuntz_fock.py matrix_master_field/tests/test_cuntz_fock.py
git commit -m "feat(mmf): port Cuntz-Fock operator space with relation + free-product tests"
```

---

### Task 3: One-matrix operator master field reproduces exact moments (centerpiece)

**Files:**
- Create: `matrix_master_field/operator_field.py`
- Test: `matrix_master_field/tests/test_operator_field.py`

**Interfaces:**
- Consumes: `one_matrix.gaussian_moments`, `one_matrix.quartic_moments_from_sd`, `one_matrix.r_transform_from_moments`, `one_matrix.voiculescu_coefficients`; `CuntzFockSpace.build_master_field_voiculescu`, `.compute_moments`.
- Produces: `operator_field.one_matrix_master_field_from_moments(target_moments, fock_length) -> (M_hat, model_moments)` returning the assembled operator and its vacuum moments.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_operator_field.py
import numpy as np
from matrix_master_field import one_matrix as om
from matrix_master_field.operator_field import one_matrix_master_field_from_moments


def test_gaussian_master_field_operator_recovers_catalan():
    target = om.gaussian_moments(8)
    _, model = one_matrix_master_field_from_moments(target, fock_length=10)
    assert np.max(np.abs(model[:9] - target[:9])) < 1e-6


def test_quartic_master_field_operator_recovers_exact_moments():
    g = 0.5
    target = om.quartic_moments_from_sd(g, max_power=8)
    _, model = one_matrix_master_field_from_moments(target, fock_length=12)
    # operator vacuum moments must match the exact one-matrix moments
    assert np.max(np.abs(model[:9] - target[:9])) < 1e-6, (
        f"max err {np.max(np.abs(model[:9] - target[:9])):.2e}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest matrix_master_field/tests/test_operator_field.py -v`
Expected: FAIL with `ImportError`/`ModuleNotFoundError` (`operator_field` not created).

- [ ] **Step 3: Write the implementation**

```python
# matrix_master_field/operator_field.py
"""Operator (Cuntz-Fock) realization of the master field.

Milestone 1: the one-matrix master field in closed form. The master field
of a one-matrix model is fixed by its free cumulants (Voiculescu): build the
operator M̂ = â + Σ_n M_n (â†)^n on the truncated Fock space and read its
vacuum moments. This validates that the operator representation reproduces
the known one-matrix answer (no optimization needed at one matrix).
"""

import numpy as np

from matrix_master_field.cuntz_fock import CuntzFockSpace
from matrix_master_field.one_matrix import (
    r_transform_from_moments,
    voiculescu_coefficients,
)


def one_matrix_master_field_from_moments(target_moments, fock_length: int = 10):
    """Build the one-matrix master-field operator from target moments.

    Args:
        target_moments: array m_0, m_1, ..., m_K (m_0 must be 1).
        fock_length: truncation length L of the Cuntz-Fock space.

    Returns:
        (M_hat, model_moments): the assembled operator and its vacuum moments
        tr[M̂^p] = ⟨Ω|M̂^p|Ω⟩ for p = 0..K.
    """
    target_moments = np.asarray(target_moments, dtype=float)
    K = len(target_moments) - 1

    # Free cumulants κ_n, then Voiculescu coefficients M_n = κ_{n+1}.
    kappa = r_transform_from_moments(target_moments)
    v_coeffs = voiculescu_coefficients(kappa)

    fock = CuntzFockSpace(n_matrices=1, max_length=fock_length)
    n_coeffs = min(len(v_coeffs), fock_length)
    M_hat = fock.build_master_field_voiculescu(v_coeffs[:n_coeffs], matrix_idx=0)
    model_moments = fock.compute_moments(M_hat, max_power=K)
    return M_hat, model_moments
```

- [ ] **Step 4: Run the tests**

Run: `uv run python -m pytest matrix_master_field/tests/test_operator_field.py -v`
Expected: both PASS. If the quartic test misses 1e-6, raise `fock_length` (truncation error) and confirm convergence; if it plateaus far from 1e-6, STOP and report — that would indicate a real issue in the Voiculescu build, not a tolerance tweak.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/operator_field.py matrix_master_field/tests/test_operator_field.py
git commit -m "feat(mmf): one-matrix operator master field reproduces exact moments to 1e-6"
```

---

### Task 4: One-matrix SDP bootstrap bounds (independent validator)

**Files:**
- Create: `matrix_master_field/bootstrap_sdp.py` (port of `master_field/bootstrap_sdp.py`)
- Test: `matrix_master_field/tests/test_bootstrap_sdp.py`

**Interfaces:**
- Produces: `bootstrap_sdp.bootstrap_one_matrix(v_prime_coeffs, max_moment, target_moment, maximize) -> float | None`; `bootstrap_sdp.HAS_CVXPY: bool`.

- [ ] **Step 1: Port verbatim**

Copy `master_field/bootstrap_sdp.py` to `matrix_master_field/bootstrap_sdp.py`. In the `if __name__ == "__main__"` block only, change `from one_matrix import gaussian_moments` to `from matrix_master_field.one_matrix import gaussian_moments`.

```bash
cp master_field/bootstrap_sdp.py matrix_master_field/bootstrap_sdp.py
```
Then edit the `__main__` import as noted.

- [ ] **Step 2: Write the failing test**

```python
# matrix_master_field/tests/test_bootstrap_sdp.py
import numpy as np
import pytest
from matrix_master_field import bootstrap_sdp as bs
from matrix_master_field import one_matrix as om

pytestmark = pytest.mark.skipif(not bs.HAS_CVXPY, reason="cvxpy not installed")


def test_gaussian_bounds_bracket_exact_m2():
    exact = om.gaussian_moments(8)[2]  # = 1
    lb = bs.bootstrap_one_matrix([0.0, 1.0], max_moment=8, target_moment=2, maximize=False)
    ub = bs.bootstrap_one_matrix([0.0, 1.0], max_moment=8, target_moment=2, maximize=True)
    assert lb is not None and ub is not None
    assert lb - 1e-4 <= exact <= ub + 1e-4, f"exact {exact} not in [{lb}, {ub}]"


def test_quartic_bounds_bracket_exact_m2():
    exact = om.quartic_moments_from_sd(0.5, max_power=8)[2]
    lb = bs.bootstrap_one_matrix([0.0, 1.0, 0.0, 0.5], max_moment=8, target_moment=2, maximize=False)
    ub = bs.bootstrap_one_matrix([0.0, 1.0, 0.0, 0.5], max_moment=8, target_moment=2, maximize=True)
    assert lb is not None and ub is not None
    assert lb - 1e-3 <= exact <= ub + 1e-3, f"exact {exact} not in [{lb}, {ub}]"
```

- [ ] **Step 3: Run the tests**

Run: `uv run python -m pytest matrix_master_field/tests/test_bootstrap_sdp.py -v`
Expected: both PASS (bounds bracket the exact value). If cvxpy is absent, they SKIP — then `uv run --with cvxpy python -m pytest ...` to actually run them.

- [ ] **Step 4: Run the full Milestone-1 suite**

Run: `uv run python -m pytest matrix_master_field/tests/ -v`
Expected: all tests PASS (or SDP tests SKIP if cvxpy unavailable).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/bootstrap_sdp.py matrix_master_field/tests/test_bootstrap_sdp.py
git commit -m "feat(mmf): port one-matrix SDP bootstrap; bounds bracket exact moments"
```

---

## Self-Review

**Spec coverage (Milestone 1 = "Moments + ρ(x) to ≤1e-6 vs exact"):** Task 1 ports the exact reference (moments + density); Task 3 validates the operator master field reproduces exact *moments* to ≤1e-6 (Gaussian + quartic); Task 4 adds rigorous bounds. **Gap noted:** ρ(x) *via diagonalizing the operator* is not validated here — the Voiculescu-form operator reproduces moments but is not manifestly Hermitian, so spectral-density extraction is deferred to Milestone 2, where the Hermitian optimization ansatz is built and ρ(x) can be read off by `eigh`. Moment-level validation (≤1e-6) plus the exact `quartic_eigenvalue_density` reference covers the density at the reference level for M1. This deferral is intentional and recorded in the spec (§6, §8).

**Placeholder scan:** none — all steps contain runnable commands and complete code. The only "transcribe later" item (KZ action) is a Milestone-4 convention task, correctly out of M1 scope.

**Type consistency:** `one_matrix_master_field_from_moments(target_moments, fock_length)` returns `(M_hat, model_moments)` and is consumed consistently in `test_operator_field.py`. Ported function signatures match their `master_field/` originals (verified against the source files).
