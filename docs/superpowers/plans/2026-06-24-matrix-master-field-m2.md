# Matrix Master Field — Milestone 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax. NOTE: the JAX engine here is genuinely new code — build it strictly test-first; the code blocks below are concrete starting points, refine under TDD when a test demands it.

**Goal:** Build the JAX loop-equation optimization engine (positivity automatic) and prove it recovers the *known* one-matrix master field from random init; then compare three operator ansätze head-to-head on that solved case.

**Architecture:** A shared pipeline — JAX moment evaluator (vacuum expectations on the Cuntz–Fock space) → loop-equation residual loss + cyclicity + normalization → optax trainer with coupling-homotopy and multi-restart. A pluggable **ansatz interface** `params → {Hermitian M̂_i}` with three implementations. A comparison harness scores them on the 1-matrix ground truth. The 2-matrix application is Milestone 3.

**Tech Stack:** JAX + optax (float64), NumPy, SciPy. Reuses `matrix_master_field.cuntz_fock` (numpy constants) and `matrix_master_field.one_matrix` (truth) and `schwinger_dyson`-style loop equations.

## Global Constraints

- **Runner:** `uv run --no-project --with numpy --with scipy --with jax --with optax --with pytest python -m pytest …` (verify JAX installs in Task 1, Step 1; CPU jax is fine).
- **Float64:** `jax.config.update("jax_enable_x64", True)` at the top of every JAX file.
- **Hermitian** operators M̂_i = M̂_i† (every ansatz must guarantee this). **Positivity is automatic** (τ(·)=⟨Ω|·|Ω⟩ is a vacuum state) — never a loss term.
- **m₀ = 1** hard. **Cyclicity** imposed as a loss (Cuntz vacuum is not tracial).
- Conventions: `matrix_master_field/CONVENTIONS.md`. Validation truth: `matrix_master_field/one_matrix.py`.
- **Comparison is the deliverable:** every ansatz is scored on the same 1-matrix tasks; no ansatz is presumed best.

---

### Task 1: JAX moment evaluator on the Cuntz–Fock space

**Files:** Create `matrix_master_field/fock_jax.py`; Test `matrix_master_field/tests/test_fock_jax.py`

**Interfaces:**
- Produces: `fock_jax.FockOps(n_matrices, max_length)` holding jnp constants `a[i]`, `adag[i]` (shape D×D) and `D`, `vacuum` (e₀); `fock_jax.word_moment(ops_list, word) -> jnp scalar` = ⟨Ω| M̂_{w₁}…M̂_{w_k} |Ω⟩; `fock_jax.power_moments(M, K) -> jnp array` = [⟨Ω|Mᵖ|Ω⟩]_{p=0..K}.

- [ ] **Step 1: Verify the JAX runner**

Run: `uv run --no-project --with numpy --with jax --with optax python -c "import jax; jax.config.update('jax_enable_x64',True); import jax.numpy as jnp; print(jnp.ones(3).dtype)"`
Expected: `float64`. (If jax download is slow, allow up to 5 min once; it caches.)

- [ ] **Step 2: Write the failing test** (agreement with the validated numpy Fock space)

```python
# matrix_master_field/tests/test_fock_jax.py
import numpy as np
from matrix_master_field.cuntz_fock import CuntzFockSpace
from matrix_master_field.fock_jax import FockOps, power_moments, word_moment


def test_jax_matches_numpy_gaussian_catalan():
    ops = FockOps(n_matrices=1, max_length=8)
    M = ops.a[0] + ops.adag[0]  # x̂ = â + â†
    m = np.asarray(power_moments(M, 8))
    for k, cat in [(0, 1), (2, 1), (4, 2), (6, 5), (8, 14)]:
        assert np.isclose(m[k], cat), f"tr[M^{k}]={m[k]} expected {cat}"


def test_jax_word_moment_matches_numpy():
    npf = CuntzFockSpace(n_matrices=2, max_length=4)
    ops = FockOps(n_matrices=2, max_length=4)
    M1n, M2n = npf.x(0), npf.x(1)
    M1, M2 = ops.a[0] + ops.adag[0], ops.a[1] + ops.adag[1]
    for word, Mn_prod in [((0, 1, 0, 1), M1n @ M2n @ M1n @ M2n),
                          ((0, 0, 1, 1), M1n @ M1n @ M2n @ M2n)]:
        got = float(word_moment([M1, M2], word))
        assert np.isclose(got, npf.vev(Mn_prod), atol=1e-12), f"{word}: {got}"
```

- [ ] **Step 3: Run to verify failure** — `ImportError` (`fock_jax` absent).

- [ ] **Step 4: Implement** (starter; the a/adag come straight from the validated numpy builder)

```python
# matrix_master_field/fock_jax.py
"""JAX moment evaluation on the truncated Cuntz-Fock space."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from matrix_master_field.cuntz_fock import CuntzFockSpace


class FockOps:
    def __init__(self, n_matrices: int, max_length: int):
        base = CuntzFockSpace(n_matrices, max_length)
        self.n = n_matrices
        self.D = base.dim
        self.a = [jnp.asarray(base.a(i), dtype=jnp.float64) for i in range(n_matrices)]
        self.adag = [jnp.asarray(base.adag(i), dtype=jnp.float64) for i in range(n_matrices)]
        self.vacuum = jnp.asarray(base.vacuum_state(), dtype=jnp.float64)


def word_moment(ops_list, word):
    """⟨Ω| M̂_{w0} … M̂_{w_{k-1}} |Ω⟩ via right-to-left matvecs."""
    D = ops_list[0].shape[0]
    v = jnp.zeros(D).at[0].set(1.0)  # |Ω⟩
    for idx in reversed(word):
        v = ops_list[idx] @ v
    return v[0]  # ⟨Ω| · v


def power_moments(M, K):
    """[⟨Ω|Mᵖ|Ω⟩]_{p=0..K} for a single operator M."""
    D = M.shape[0]
    v = jnp.zeros(D).at[0].set(1.0)
    out = [v[0]]
    for _ in range(K):
        v = M @ v
        out.append(v[0])
    return jnp.stack(out)
```

- [ ] **Step 5: Run** — `uv run --no-project --with numpy --with scipy --with jax --with optax --with pytest python -m pytest matrix_master_field/tests/test_fock_jax.py -v` → PASS.

- [ ] **Step 6: Commit** — `feat(mmf): JAX moment evaluator on Cuntz-Fock space (matches numpy)`.

---

### Task 2: Ansatz interface + Ansatz 1 (low-degree monomial, Hermitian)

**Files:** Create `matrix_master_field/ansatz.py`; Test `matrix_master_field/tests/test_ansatz.py`

**Interfaces:**
- Produces: a common protocol — each ansatz exposes `init_params(key) -> pytree` and `build_operators(params) -> list[jnp D×D Hermitian]`. `ansatz.MonomialAnsatz(fock_ops, degree)` implements it; degree-1 must equal â+â†.

- [ ] **Step 1: Failing test**

```python
# matrix_master_field/tests/test_ansatz.py
import jax, numpy as np
from matrix_master_field.fock_jax import FockOps, power_moments
from matrix_master_field.ansatz import MonomialAnsatz


def test_monomial_operators_are_hermitian():
    ops = FockOps(1, 6)
    ans = MonomialAnsatz(ops, degree=3)
    params = ans.init_params(jax.random.PRNGKey(0))
    for M in ans.build_operators(params):
        assert np.allclose(np.asarray(M), np.asarray(M).T.conj(), atol=1e-12)


def test_degree1_can_represent_free_field():
    ops = FockOps(1, 8)
    ans = MonomialAnsatz(ops, degree=1)
    # the parameter vector that selects M = â + â† must give Catalan moments
    params = ans.params_for_free_field()
    M = ans.build_operators(params)[0]
    m = np.asarray(power_moments(M, 4))
    assert np.isclose(m[2], 1.0) and np.isclose(m[4], 2.0)
```

- [ ] **Step 2–5:** Run (fail) → implement `MonomialAnsatz` (real coefficients over monomials â†_u â_v with |u|+|v|≤degree; enforce Hermiticity by adding each monomial together with its adjoint â†_v â_u with the conjugate coefficient; `params_for_free_field` sets the â+â† coefficients to 1, rest 0) → run (pass) → commit `feat(mmf): ansatz interface + monomial Hermitian ansatz`.

  Build M̂ = Σ over unordered monomial-pairs of (real c)·(â†_u â_v + â†_v â_u) plus the free generators; precompute each monomial's constant matrix once from `FockOps.a/adag`. Hermiticity is then exact by construction. (Refine the exact coefficient bookkeeping under the Hermiticity test.)

---

### Task 3: One-matrix loop-equation residual loss

**Files:** Create `matrix_master_field/loss.py`; Test `matrix_master_field/tests/test_loss.py`

**Interfaces:**
- Produces: `loss.one_matrix_sd_residual(power_moments_vec, v_prime_coeffs) -> jnp scalar` (mean-squared SD residual, relative-scaled as in `neural_master_field.sd_loss_one_matrix`); `loss.cyclicity_is_trivial_one_matrix()` (note: 1-matrix powers are cyclically trivial, so cyclicity loss = 0 here; it returns 0 and exists for interface parity).

- [ ] **Steps:** Failing test asserting the SD residual ≈ 0 (≤1e-10) when fed the **exact** Gaussian and quartic `power_moments` (compute the operator via M1's `one_matrix_master_field_from_moments`, evaluate `power_moments`, then residual). Implement by porting the relative-residual SD loss from `master_field/neural_master_field.py::sd_loss_one_matrix` into jnp. Run → pass → commit.

---

### Task 4: Trainer + Ansatz-1 one-matrix validation (proves the engine)

**Files:** Create `matrix_master_field/train.py`; Test `matrix_master_field/tests/test_train_one_matrix.py`

**Interfaces:**
- Produces: `train.solve(ansatz, v_prime_coeffs, fock_ops, K, *, n_restarts, steps, seed) -> dict` with keys `moments` (best vacuum power-moments), `sd_loss`, `params`. Uses optax Adam + warmup-cosine, multi-restart (keep lowest loss), optional coupling-homotopy hook.

- [ ] **Step 1: Failing test** (the engine recovers the *known* answer by optimization, not by closed form)

```python
# matrix_master_field/tests/test_train_one_matrix.py
import numpy as np
from matrix_master_field import one_matrix as om
from matrix_master_field.fock_jax import FockOps
from matrix_master_field.ansatz import MonomialAnsatz
from matrix_master_field.train import solve


def test_engine_recovers_gaussian_by_optimization():
    ops = FockOps(1, 10)
    ans = MonomialAnsatz(ops, degree=3)
    res = solve(ans, [0.0, 1.0], ops, K=8, n_restarts=4, steps=3000, seed=0)
    target = om.gaussian_moments(8)
    assert np.max(np.abs(res["moments"][:9] - target[:9])) < 1e-4


def test_engine_recovers_quartic_by_optimization():
    ops = FockOps(1, 12)
    ans = MonomialAnsatz(ops, degree=3)
    res = solve(ans, [0.0, 1.0, 0.0, 0.5], ops, K=8, n_restarts=6, steps=5000, seed=0)
    target = om.quartic_moments_from_sd(0.5, max_power=8)
    assert np.max(np.abs(res["moments"][:9] - target[:9])) < 1e-3
```

- [ ] **Steps 2–6:** Run (fail) → implement `solve` (jit train step, value_and_grad on `sd_residual(power_moments(build_operators(params)), v_prime)`, multi-restart) → run; **if it misses tolerance, this is a real result to investigate (degree too low? optimizer? restarts?) — record findings, do not just loosen the tolerance** → on pass, add a ρ(x)-via-eigh check (diagonalize the best Hermitian M̂, compare its spectral density to `om.quartic_eigenvalue_density`) → commit `feat(mmf): optimization engine recovers 1-matrix master field from random init`.

---

### Task 5: Ansatz 2 (full Hermitian matrix) + same validation

**Files:** Modify `matrix_master_field/ansatz.py` (add `DenseHermitianAnsatz`); Test add to `test_train_one_matrix.py`.

- [ ] `DenseHermitianAnsatz(fock_ops)`: params = a real lower-triangular `L` and strictly-lower `S`; M̂ = (L + Lᵀ) + i(S − Sᵀ) → Hermitian (for a single real-spectrum operator, real-symmetric `M̂ = L + Lᵀ` suffices; keep it real-symmetric for the 1-matrix case). Same `solve` interface. Test: recovers Gaussian & quartic to the same tolerances; **record parameter count (≫ monomial) and restart-robustness.** Commit.

---

### Task 6: Ansatz 3 (amortized network) + λ-range validation

**Files:** Create `matrix_master_field/amortized.py`; Test `matrix_master_field/tests/test_amortized.py`.

- [ ] `AmortizedMonomial(fock_ops, degree, hidden)`: MLP g(coupling) → monomial coefficients (same assembly as Ansatz 1). Train across the quartic family g ∈ {0.1,…,1.0} on the summed SD residual. Test: at held-out g, recovered moments match `om.quartic_moments_from_sd(g)` to ≤1e-2; observables differentiable in g. Commit.

---

### Task 7: Comparison report

**Files:** Create `matrix_master_field/validate.py` (a `compare_ansatze()` routine) + `docs/superpowers/results/2026-06-24-ansatz-comparison.md`.

- [ ] For the quartic g=0.5 task, run all three ansätze with matched budget; tabulate: max moment error, ρ(x) error vs exact, **#parameters**, steps-to-tol, and **fraction of restarts that converge to the correct basin** (the spurious-solution metric). Write the table + a 1-paragraph verdict (which ansatz to carry into the 2-matrix Milestone 3, and why) to the results doc. Commit.

---

## Self-Review

- **Coverage:** engine (Tasks 1,3,4) + the "try 1,2,3" comparison (Tasks 2,5,6,7) + Hermitian ρ(x) (Task 4). 2-matrix deferred to M3 (explicit).
- **Placeholders:** the JAX assembly in Tasks 2/4/5/6 is flagged "refine under TDD" rather than fully spelled out — deliberate, because it's new research code best built test-first; every task has a concrete test contract that pins success. The starter code in Task 1 is complete.
- **Honesty gate:** Tasks 4/7 explicitly forbid loosening tolerances to force a pass — a miss is a recorded result (ansatz expressiveness / optimization), which is exactly the comparison's purpose.
- **Type consistency:** `solve(ansatz, v_prime, fock_ops, K, …)` and the `init_params`/`build_operators` ansatz protocol are used identically across Tasks 4–6.
