# M5c Exact-Diagonalization Ground Truth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute the *true* ground-state energy density `E/N²` of the two-matrix QM (HHK arXiv:2004.10212 Eq 17) by exact diagonalization at N=2 and N=3, an independent reference that adjudicates every M5c bound.

**Architecture:** Decompose `X,Y` into an orthonormal Hermitian mode basis `{T_a}` → `2N²` decoupled oscillators plus a positive quartic `g²Σ_c L_c²` (`L_c=Σ_{ab}f_{abc}x_a y_b`). Diagonalize over a **bosonic multi-mode Fock space with total-quanta truncation** `Σn_i ≤ K` (the `2(N²−1)` interacting modes; the 2 trace modes contribute `2m` analytically), assembled as a `scipy.sparse` matrix and solved with Lanczos (`eigsh`).

**Tech Stack:** Python, numpy, scipy (`scipy.sparse`, `scipy.sparse.linalg.eigsh`, `scipy.optimize.minimize_scalar`), pytest. **No JAX** (no autodiff needed; exact diagonalization only).

## Global Constraints

- **Model (pinned, HHK Eq 17):** `H = Tr(P_X²+P_Y²+m²(X²+Y²) − g²[X,Y]²)`, `X,Y` Hermitian `N×N` (**U(N)** — trace mode included), `ℏ=1`, `[X_{ij},P_{X,kl}]=iδ_{il}δ_{jk}`. 't Hooft coupling `λ=Ng²`; **report `E/N²`**. `m=1` in all numeric tests unless stated.
- **'t Hooft conversion (do NOT guess):** a target `λ` at fixed `N` means `g = sqrt(λ/N)`. `ground_energy` takes `g`; the deliverable converts `λ→g`.
- **Mode basis (pinned):** orthonormal Hermitian `{T_a}_{a=0}^{N²−1}`, `Tr(T_aT_b)=δ_{ab}`, **index 0 = `I_N/√N` (trace mode)**, indices `1..N²−1` traceless. `X=Σ_a x_a T_a`. Canonical modes `[x_a,p_{x,b}]=iδ_{ab}`; `Tr X²=Σx_a²`, `Tr P_X²=Σp_{x,a}²`.
- **Structure constants (pinned):** `[T_a,T_b]=iΣ_c f_{abc}T_c`, `f_{abc}=−i Tr([T_a,T_b]T_c)`, **real and totally antisymmetric** in the orthonormal Hermitian basis.
- **Interaction (derived, FD/exactly verified in Task 2):** `−g²Tr[X,Y]² = +g²Σ_c(Σ_{ab}f_{abc}x_a y_b)² ≥ 0` (confining, sum of squares).
- **Oscillator rep (pinned):** each mode at frequency `m`: `x_a=(â_a+â†_a)/√(2m)`, `p_{x,a}=−i√(m/2)(â_a−â†_a)`, `[â,â†]=1`, so `p_{x,a}²+m²x_a²=m(2â†_aâ_a+1)`. Separate ladder sets for x-modes (`â`) and y-modes (`b̂`).
- **Trace-mode reduction:** `T_0∝I` commutes with all, so `x_0,y_0` are free decoupled oscillators absent from the interaction; the diagonalization runs over the `2(N²−1)` traceless modes and the two trace modes add `2m` to `E` analytically.
- **`g=0` anchor (exact, all N, all K):** every mode in `n=0` ⇒ `E=2N²m` ⇒ **`E/N²=2m`**.
- **Float64 everywhere.** `eigsh(..., which='SA')` with a fixed `v0` for reproducibility.
- **Test runner (project convention — `matrix_master_field/CONVENTIONS.md`):**
  `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -v`
- **Do NOT conflate finite-N and large-N (spec R2):** the finite-N bracket is `[2m, ⟨H⟩_Gauss(N)]` (both apply directly to `E_exact(N)`). The large-N Gaussian `2.365` and the planar SDP are NOT finite-N bounds — they appear ONLY in the `N→∞` extrapolation (Task 9). Never assert `2.365` against a finite-N value.

---

### Task 1: Hermitian mode basis + structure constants

**Files:**
- Create: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: nothing (foundation).
- Produces:
  - `hermitian_basis(N: int) -> np.ndarray` — shape `(N**2, N, N)` complex128; index 0 = `I_N/√N`, indices `1..N²−1` traceless Hermitian; `Tr(T_a T_b)=δ_ab`.
  - `structure_constants(N: int) -> np.ndarray` — shape `(N**2, N**2, N**2)` float64; `f[a,b,c] = −i Tr([T_a,T_b] T_c)`.

- [ ] **Step 1: Write the failing tests**

```python
# matrix_master_field/tests/test_exact_diag.py
import numpy as np
import pytest

from matrix_master_field.exact_diag import hermitian_basis, structure_constants


def _commutator(A, B):
    return A @ B - B @ A


def test_basis_orthonormal_and_trace_mode():
    for N in (2, 3):
        T = hermitian_basis(N)
        assert T.shape == (N * N, N, N)
        # orthonormality Tr(T_a T_b) = delta_ab
        gram = np.einsum("aij,bji->ab", T, T)
        assert np.allclose(gram, np.eye(N * N), atol=1e-12)
        # index 0 is I/sqrt(N); the rest are traceless and Hermitian
        assert np.allclose(T[0], np.eye(N) / np.sqrt(N), atol=1e-12)
        for a in range(1, N * N):
            assert abs(np.trace(T[a])) < 1e-12
            assert np.allclose(T[a], T[a].conj().T, atol=1e-12)


def test_basis_n2_is_pauli():
    T = hermitian_basis(2)
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    assert np.allclose(T[1], sx / np.sqrt(2), atol=1e-12)
    assert np.allclose(T[2], sy / np.sqrt(2), atol=1e-12)
    assert np.allclose(T[3], sz / np.sqrt(2), atol=1e-12)


def test_structure_constants_real_and_antisymmetric():
    for N in (2, 3):
        f = structure_constants(N)
        assert f.shape == (N * N,) * 3
        assert f.dtype == np.float64
        # totally antisymmetric
        assert np.allclose(f, -np.transpose(f, (1, 0, 2)), atol=1e-10)
        assert np.allclose(f, -np.transpose(f, (0, 2, 1)), atol=1e-10)
        # trace mode commutes with everything -> any f with a 0 index vanishes
        assert np.allclose(f[0], 0.0, atol=1e-12)
        assert np.allclose(f[:, 0], 0.0, atol=1e-12)
        assert np.allclose(f[:, :, 0], 0.0, atol=1e-12)


def test_structure_constants_reconstruct_commutator():
    for N in (2, 3):
        T = hermitian_basis(N)
        f = structure_constants(N)
        for a in range(N * N):
            for b in range(N * N):
                lhs = _commutator(T[a], T[b])
                rhs = 1j * np.einsum("c,cij->ij", f[a, b], T)
                assert np.allclose(lhs, rhs, atol=1e-10)


def test_structure_constants_n2_value():
    f = structure_constants(2)
    # [T_1,T_2]=i sqrt(2) T_3 for Pauli/sqrt(2): f_{123}=sqrt(2)
    assert np.isclose(f[1, 2, 3], np.sqrt(2.0), atol=1e-10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -v`
Expected: FAIL with `ImportError: cannot import name 'hermitian_basis'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# matrix_master_field/exact_diag.py
"""M5c follow-up — exact diagonalization of the two-matrix QM (ground truth).

Model (HHK arXiv:2004.10212 Eq 17):
    H = Tr(P_X^2 + P_Y^2 + m^2 (X^2 + Y^2) - g^2 [X,Y]^2),
X,Y Hermitian NxN (U(N), trace mode included), hbar=1, [X_ij,P_X,kl]=i delta_il delta_jk.
't Hooft coupling lambda = N g^2; report E/N^2.

Method: orthonormal Hermitian mode basis {T_a} -> 2N^2 decoupled oscillators + positive
quartic g^2 sum_c L_c^2 with L_c = sum_ab f_abc x_a y_b. Diagonalize over the bosonic Fock
space of the 2(N^2-1) traceless modes with total-quanta truncation sum n_i <= K (the 2 trace
modes add 2m analytically). See docs/superpowers/specs/2026-06-26-m5c-exact-diag-ground-truth-design.md.

Conventions are pinned in the plan's Global Constraints and CONVENTIONS.md. Float64 throughout.
"""
import numpy as np


def hermitian_basis(N):
    """Orthonormal Hermitian basis {T_a}, Tr(T_a T_b)=delta_ab, index 0 = I/sqrt(N).

    Generalized Gell-Mann basis: trace I/sqrt(N), then symmetric off-diagonal,
    antisymmetric off-diagonal, and diagonal (Cartan) traceless generators.
    """
    mats = [np.eye(N, dtype=np.complex128) / np.sqrt(N)]
    # symmetric and antisymmetric off-diagonal
    for j in range(N):
        for k in range(j + 1, N):
            S = np.zeros((N, N), dtype=np.complex128)
            S[j, k] = S[k, j] = 1.0 / np.sqrt(2.0)
            mats.append(S)
            A = np.zeros((N, N), dtype=np.complex128)
            A[j, k] = -1j / np.sqrt(2.0)
            A[k, j] = 1j / np.sqrt(2.0)
            mats.append(A)
    # diagonal Cartan generators D_l, l=1..N-1
    for l in range(1, N):
        D = np.zeros((N, N), dtype=np.complex128)
        for j in range(l):
            D[j, j] = 1.0
        D[l, l] = -l
        D = D / np.sqrt(l * (l + 1))
        mats.append(D)
    return np.stack(mats, axis=0)


def structure_constants(N):
    """f[a,b,c] = -i Tr([T_a,T_b] T_c), real and totally antisymmetric."""
    T = hermitian_basis(N)
    n = N * N
    f = np.zeros((n, n, n), dtype=np.float64)
    for a in range(n):
        for b in range(n):
            comm = T[a] @ T[b] - T[b] @ T[a]
            for c in range(n):
                val = -1j * np.trace(comm @ T[c])
                assert abs(val.imag) < 1e-10, "f must be real"
                f[a, b, c] = val.real
    return f
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — Hermitian mode basis + structure constants (V1a)"
```

---

### Task 2: Quartic ↔ commutator identity (V1)

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `hermitian_basis`, `structure_constants` (Task 1).
- Produces:
  - `quartic_potential_value(N: int, x_vec: np.ndarray, y_vec: np.ndarray) -> float` — returns `Σ_c(Σ_{ab}f_{abc}x_a y_b)²` (= `−Tr[X,Y]²`). `x_vec,y_vec` length `N²` real.

- [ ] **Step 1: Write the failing test**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import quartic_potential_value


def test_quartic_matches_minus_tr_commutator_sq():
    rng = np.random.default_rng(0)
    for N in (2, 3):
        T = hermitian_basis(N)
        for _ in range(5):
            x = rng.standard_normal(N * N)
            y = rng.standard_normal(N * N)
            X = np.einsum("a,aij->ij", x, T)
            Y = np.einsum("a,aij->ij", y, T)
            comm = X @ Y - Y @ X
            ref = -np.trace(comm @ comm)  # = +sum_c L_c^2, real and >= 0
            assert abs(ref.imag) < 1e-10
            val = quartic_potential_value(N, x, y)
            assert np.isclose(val, ref.real, atol=1e-10)
            assert val >= -1e-12  # positive (confining)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py::test_quartic_matches_minus_tr_commutator_sq -v`
Expected: FAIL with `ImportError: cannot import name 'quartic_potential_value'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to matrix_master_field/exact_diag.py
def quartic_potential_value(N, x_vec, y_vec):
    """sum_c (sum_ab f_abc x_a y_b)^2  ==  -Tr[X,Y]^2 (classical, c-number x,y)."""
    f = structure_constants(N)
    L = np.einsum("abc,a,b->c", f, np.asarray(x_vec, float), np.asarray(y_vec, float))
    return float(np.dot(L, L))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py::test_quartic_matches_minus_tr_commutator_sq -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — quartic = -Tr[X,Y]^2 identity, verified (V1)"
```

---

### Task 3: Bosonic Fock basis (total-quanta truncation) + sparse ladder operators

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: nothing (independent foundation).
- Produces:
  - `occupation_basis(n_modes: int, K: int) -> np.ndarray` — shape `(D, n_modes)` int64, all occupations with `Σn_i ≤ K`; `D = C(K+n_modes, n_modes)`.
  - `fock_ladder_ops(n_modes: int, K: int) -> tuple[np.ndarray, list]` — returns `(occ, ops)` where `occ` is the basis array and `ops[i]` is the sparse `scipy.sparse.csr_matrix` annihilation operator `â_i` (`â†_i = ops[i].transpose()`).

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from math import comb

import scipy.sparse as sp

from matrix_master_field.exact_diag import occupation_basis, fock_ladder_ops


def test_occupation_basis_size_and_bound():
    for n_modes, K in [(6, 4), (3, 5), (2, 7)]:
        occ = occupation_basis(n_modes, K)
        assert occ.shape == (comb(K + n_modes, n_modes), n_modes)
        assert occ.sum(axis=1).max() <= K
        assert occ.min() >= 0
        # all rows distinct
        assert len({tuple(r) for r in occ}) == occ.shape[0]


def test_ladder_commutator_interior():
    n_modes, K = 3, 5
    occ, ops = fock_ladder_ops(n_modes, K)
    D = occ.shape[0]
    for i in range(n_modes):
        a = ops[i]
        adag = a.transpose()
        comm = (a @ adag - adag @ a).toarray()
        # [a_i, a_i^dag] = 1 on states with total quanta < K (interior, not truncated)
        for r in range(D):
            if occ[r].sum() < K:
                assert np.isclose(comm[r, r], 1.0, atol=1e-12)


def test_number_operator_eigenvalues():
    n_modes, K = 4, 4
    occ, ops = fock_ladder_ops(n_modes, K)
    for i in range(n_modes):
        num = (ops[i].transpose() @ ops[i]).diagonal()
        assert np.allclose(num, occ[:, i], atol=1e-12)


def test_ladder_lowers_one_quantum():
    n_modes, K = 2, 3
    occ, ops = fock_ladder_ops(n_modes, K)
    index = {tuple(occ[r]): r for r in range(occ.shape[0])}
    a0 = ops[0]
    for r in range(occ.shape[0]):
        if occ[r, 0] > 0:
            tgt = occ[r].copy(); tgt[0] -= 1
            rt = index[tuple(tgt)]
            assert np.isclose(a0[rt, r], np.sqrt(occ[r, 0]), atol=1e-12)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k "occupation or ladder or number" -v`
Expected: FAIL with `ImportError: cannot import name 'occupation_basis'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add near the top of matrix_master_field/exact_diag.py, after `import numpy as np`
import scipy.sparse as sp


def occupation_basis(n_modes, K):
    """All occupation tuples (n_0,...,n_{n_modes-1}), n_i>=0, sum n_i <= K."""
    rows = []

    def rec(prefix, remaining):
        if len(prefix) == n_modes:
            rows.append(prefix)
            return
        for v in range(remaining + 1):
            rec(prefix + (v,), remaining - v)

    rec((), K)
    return np.array(rows, dtype=np.int64)


def _radix_key(occ_row, base):
    """Mixed-radix integer encoding of an occupation row (base = K+1) for O(1) lookup."""
    key = 0
    for v in occ_row[::-1]:
        key = key * base + int(v)
    return key


def fock_ladder_ops(n_modes, K):
    """(occ, ops): occupation basis and sparse annihilation operators a_i (a_i^dag = ops[i].T)."""
    occ = occupation_basis(n_modes, K)
    D = occ.shape[0]
    base = K + 1
    index = {_radix_key(occ[r], base): r for r in range(D)}
    ops = []
    for i in range(n_modes):
        rows, cols, data = [], [], []
        for c in range(D):
            ni = occ[c, i]
            if ni > 0:
                tgt = occ[c].copy()
                tgt[i] -= 1
                rows.append(index[_radix_key(tgt, base)])
                cols.append(c)
                data.append(np.sqrt(ni))
        ops.append(sp.csr_matrix((data, (rows, cols)), shape=(D, D), dtype=np.float64))
    return occ, ops
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k "occupation or ladder or number" -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — bosonic Fock basis + sparse ladder ops (total-quanta truncation)"
```

---

### Task 4: Assemble the sparse two-matrix QM Hamiltonian

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `structure_constants` (Task 1), `fock_ladder_ops` (Task 3).
- Produces:
  - `build_two_matrix_qm_hamiltonian(N: int, m: float, g: float, K: int) -> scipy.sparse.csr_matrix` — the **interacting** Hamiltonian over the `2(N²−1)`-mode truncated Fock space (the `2m` trace contribution is added later by `ground_energy`, NOT here). Free diagonal `m·(2Σn_i + 2(N²−1))` + quartic `g²Σ_c L_c²`.

  Mode layout (pinned): x-modes are traceless indices `a=1..N²−1` → ladder slot `a−1`; y-modes `b=1..N²−1` → ladder slot `(N²−1)+(b−1)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import build_two_matrix_qm_hamiltonian


def test_hamiltonian_hermitian_and_dim():
    N, m, K = 2, 1.0, 3
    H = build_two_matrix_qm_hamiltonian(N, m, g=0.7, K=K)
    n_modes = 2 * (N * N - 1)
    assert H.shape[0] == comb(K + n_modes, n_modes)
    assert abs((H - H.transpose()).max()) < 1e-12  # real symmetric


def test_hamiltonian_g0_is_diagonal_free_spectrum():
    # g=0: H_interacting diagonal, lowest entry = m * 2(N^2-1) (all interacting modes n=0)
    N, m, K = 2, 1.0, 4
    H = build_two_matrix_qm_hamiltonian(N, m, g=0.0, K=K).toarray()
    n_int = 2 * (N * N - 1)
    assert np.allclose(H - np.diag(np.diag(H)), 0.0, atol=1e-12)
    assert np.isclose(np.min(np.diag(H)), m * n_int, atol=1e-12)


def test_hamiltonian_quartic_is_psd_shift():
    # the quartic g^2 sum_c L_c^2 is PSD: H(g) - H(0) has nonnegative eigenvalues
    N, m, K = 2, 1.0, 3
    H0 = build_two_matrix_qm_hamiltonian(N, m, 0.0, K).toarray()
    Hg = build_two_matrix_qm_hamiltonian(N, m, 0.9, K).toarray()
    w = np.linalg.eigvalsh(Hg - H0)
    assert w.min() > -1e-9
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k hamiltonian -v`
Expected: FAIL with `ImportError: cannot import name 'build_two_matrix_qm_hamiltonian'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to matrix_master_field/exact_diag.py
def build_two_matrix_qm_hamiltonian(N, m, g, K, pad=2):
    """Exact Galerkin projection P_K H P_K of the interacting H over the 2(N^2-1)-mode Fock
    space, truncated to total quanta <= K (the 2m trace contribution is added by ground_energy).

    CRITICAL — exact Galerkin via padding. L_c = sum_ab f_abc x_a y_b is Hermitian in the FULL
    space (x_a, y_b act on disjoint mode sets, so they commute). But building it directly from
    K-truncated operators gives (P x_a P)(P y_b P) — a spurious MIDDLE projector that makes L_c
    non-Hermitian at the K-quantum boundary and L_c^2 non-PSD. Fix (same trick as qm_fock.py):
    build on the padded (K+pad) basis so the quartic's intermediate states (total <= K+2) are
    represented, then RESTRICT to the canonical total<=K basis. The <=K block of L_c^2-built-on-
    (K+pad) equals P_K L_c^2 P_K EXACTLY (the boundary non-Hermiticity lives only in rows/cols
    > K, which are discarded), so the result is Hermitian, PSD, and a rigorous variational upper
    bound monotone in K. pad=2 suffices (each L_c shifts total quanta by <=2). The returned H is
    ordered to match occupation_basis(N's interacting modes, K) so ground_energy / casimir agree.
    """
    n_tl = N * N - 1            # traceless modes per matrix
    n_modes = 2 * n_tl
    Kp = K + pad

    occ_p, ops_p = fock_ladder_ops(n_modes, Kp)   # padded basis
    s2m = np.sqrt(2.0 * m)
    xs = [None] + [(ops_p[a - 1] + ops_p[a - 1].transpose()) / s2m for a in range(1, n_tl + 1)]
    ys = [None] + [(ops_p[n_tl + (b - 1)] + ops_p[n_tl + (b - 1)].transpose()) / s2m
                   for b in range(1, n_tl + 1)]

    total_p = occ_p.sum(axis=1)
    H = sp.diags(m * (2.0 * total_p + n_modes), format="csr", dtype=np.float64)

    if g != 0.0:
        f = structure_constants(N)
        for c in range(1, n_tl + 1):
            terms = []
            for a in range(1, n_tl + 1):
                for b in range(1, n_tl + 1):
                    fabc = f[a, b, c]
                    if fabc != 0.0:
                        terms.append(fabc * (xs[a] @ ys[b]))
            if terms:
                Lc = terms[0]
                for t in terms[1:]:
                    Lc = Lc + t
                H = H + (g * g) * (Lc @ Lc)   # <=K block = P_K L_c^2 P_K exactly (Hermitian, PSD)

    # restrict to the canonical total<=K basis, in occupation_basis(n_modes, K) order
    occ_K = occupation_basis(n_modes, K)
    base_p = Kp + 1
    idx_p = {_radix_key(occ_p[r], base_p): r for r in range(occ_p.shape[0])}
    keep = np.array([idx_p[_radix_key(occ_K[t], base_p)] for t in range(occ_K.shape[0])],
                    dtype=np.int64)
    H = H.tocsr()[keep][:, keep]
    H = 0.5 * (H + H.transpose())   # numerical hygiene only (the block is already symmetric)
    return H.tocsr()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k hamiltonian -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — sparse two-matrix QM Hamiltonian assembly"
```

---

### Task 5: `ground_energy` + the `g=0` anchor (V2)

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `build_two_matrix_qm_hamiltonian` (Task 4).
- Produces:
  - `ground_energy(N: int, m: float, g: float, K: int) -> dict` — keys: `E_over_N2`, `E`, `E_interacting`, `K`, `basis_dim`, `n_modes`, `ground_state` (np.ndarray). Uses `eigsh(k=1, which='SA')` (dense `eigvalsh` fallback when `D<50`); `E = E_interacting + 2m`.

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import ground_energy


def test_g0_anchor_exact_all_N_K():
    # g=0 => E/N^2 = 2m exactly, any N, any K (the hard check, V2)
    for N, K in [(2, 2), (2, 5), (3, 2), (3, 3)]:
        res = ground_energy(N, m=1.0, g=0.0, K=K)
        assert np.isclose(res["E_over_N2"], 2.0, atol=1e-9)
        assert res["basis_dim"] == comb(K + 2 * (N * N - 1), 2 * (N * N - 1))


def test_g0_anchor_scales_with_m():
    res = ground_energy(2, m=1.7, g=0.0, K=3)
    assert np.isclose(res["E_over_N2"], 2 * 1.7, atol=1e-9)


def test_ground_energy_above_2m_when_interacting():
    res = ground_energy(2, m=1.0, g=0.8, K=6)
    assert res["E_over_N2"] >= 2.0 - 1e-9  # Rayleigh-Ritz: E_trunc >= E_true >= 2m
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k "anchor or above_2m" -v`
Expected: FAIL with `ImportError: cannot import name 'ground_energy'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add the import near the top of matrix_master_field/exact_diag.py
import scipy.sparse.linalg as spla


# append to matrix_master_field/exact_diag.py
def ground_energy(N, m, g, K):
    """Ground-state energy density E/N^2 of the two-matrix QM by exact diagonalization."""
    H = build_two_matrix_qm_hamiltonian(N, m, g, K)
    D = H.shape[0]
    if D < 50:
        w, v = np.linalg.eigh(H.toarray())
        e_int = float(w[0])
        gs = v[:, 0]
    else:
        v0 = np.ones(D) / np.sqrt(D)
        vals, vecs = spla.eigsh(H, k=1, which="SA", v0=v0)
        e_int = float(vals[0])
        gs = vecs[:, 0]
    E = e_int + 2.0 * m   # two trace modes, each ground energy m
    return {
        "E_over_N2": E / (N * N),
        "E": E,
        "E_interacting": e_int,
        "K": K,
        "basis_dim": D,
        "n_modes": 2 * (N * N - 1),
        "ground_state": np.asarray(gs, dtype=np.float64),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k "anchor or above_2m" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — ground_energy + g=0 anchor E/N^2=2m (V2)"
```

---

### Task 6: K-convergence (V3) + N=2 deliverable numbers (V6 part)

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `ground_energy` (Task 5).
- Produces:
  - `converge_in_K(N: int, m: float, g: float, K_list) -> dict` — keys: `series` (list of `(K, E_over_N2)`), `value` (the largest-K `E_over_N2`), `tail` (last consecutive `|ΔE/N²|`). Monotone non-increasing in K (Rayleigh-Ritz).

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import converge_in_K


def test_K_convergence_monotone_and_settles():
    # variational in K: E/N^2(K) non-increasing (Rayleigh-Ritz, rigorous), and the
    # successive gaps shrink -- the signature of convergence (V3). N=2, lambda=1.
    N, m = 2, 1.0
    g = np.sqrt(1.0 / N)  # lambda = N g^2 = 1
    out = converge_in_K(N, m, g, K_list=[4, 6, 8, 10])
    vals = [e for _, e in out["series"]]
    for lo, hi in zip(vals[1:], vals[:-1]):
        assert lo <= hi + 1e-9          # non-increasing (guaranteed)
    assert vals[-1] >= 2.0 - 1e-9       # still >= 2m (guaranteed)
    gaps = [abs(b - a) for a, b in zip(vals[:-1], vals[1:])]
    assert gaps[-1] <= gaps[0] + 1e-12  # converging: last gap no larger than the first
    assert out["tail"] == pytest.approx(gaps[-1], abs=1e-12)


def test_K_convergence_g0_flat_at_2m():
    out = converge_in_K(2, 1.0, 0.0, K_list=[2, 4, 6])
    assert all(np.isclose(e, 2.0, atol=1e-9) for _, e in out["series"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k convergence -v`
Expected: FAIL with `ImportError: cannot import name 'converge_in_K'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to matrix_master_field/exact_diag.py
def converge_in_K(N, m, g, K_list):
    """E/N^2 vs truncation K (variational, non-increasing). Returns series, value, tail."""
    series = []
    for K in K_list:
        series.append((K, ground_energy(N, m, g, K)["E_over_N2"]))
    vals = [e for _, e in series]
    tail = abs(vals[-1] - vals[-2]) if len(vals) >= 2 else float("inf")
    return {"series": series, "value": vals[-1], "tail": tail}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k convergence -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — K-convergence (V3), monotone variational tail"
```

---

### Task 7: Same-N Gaussian upper bound + V4a bracket

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `structure_constants` (Task 1), `ground_energy`/`converge_in_K` (Tasks 5–6).
- Produces:
  - `gaussian_upper(N: int, m: float, g: float) -> dict` — keys: `E_over_N2`, `E`, `omega`. Frequency-optimized **diagonal same-N Gaussian** (rigorous upper bound): `E(ω)=2m + (N²−1)(ω+m²/ω) + g²·S_f/(4ω²)`, `S_f=Σ_{a,b,c≥1}f_{abc}²`, minimized over `ω>0`.

  **Note (spec R2):** this is a *finite-N* upper bound; it is NOT the large-N `2.365`. Do not compare it to `2.365`.

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import gaussian_upper


def test_gaussian_g0_is_2m():
    for N in (2, 3):
        out = gaussian_upper(N, m=1.0, g=0.0)
        assert np.isclose(out["E_over_N2"], 2.0, atol=1e-9)
        assert np.isclose(out["omega"], 1.0, atol=1e-6)  # optimal omega = m at g=0


def test_v4a_bracket_finite_N():
    # 2m <= E_exact(converged K) <= same-N Gaussian, for g>0, N=2, lambda=1 (V4a)
    N, m = 2, 1.0
    g = np.sqrt(1.0 / N)
    e_exact = converge_in_K(N, m, g, K_list=[8, 10, 12])["value"]
    e_gauss = gaussian_upper(N, m, g)["E_over_N2"]
    assert 2.0 - 1e-9 <= e_exact
    assert e_exact <= e_gauss + 1e-6   # converged exact sits below the same-N Gaussian
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k "gaussian or v4a" -v`
Expected: FAIL with `ImportError: cannot import name 'gaussian_upper'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# add the import near the top of matrix_master_field/exact_diag.py
from scipy.optimize import minimize_scalar


# append to matrix_master_field/exact_diag.py
def gaussian_upper(N, m, g):
    """Same-N frequency-optimized diagonal Gaussian: a rigorous finite-N upper bound on E/N^2.

    Trial = product of frequency-omega oscillator ground states on the 2(N^2-1) interacting
    modes (trace modes optimal at omega=m -> 2m). <p^2+m^2 x^2> = (omega+m^2/omega)/2 per mode;
    <L_c^2> = sum_ab f_abc^2 /(4 omega^2) so the interaction is g^2 S_f/(4 omega^2).
    """
    f = structure_constants(N)
    S_f = float(np.sum(f[1:, 1:, 1:] ** 2))
    n_tl = N * N - 1

    def E_of_omega(w):
        return 2.0 * m + n_tl * (w + m * m / w) + (g * g) * S_f / (4.0 * w * w)

    res = minimize_scalar(
        E_of_omega, bounds=(1e-6, 100.0 * max(m, 1.0)), method="bounded"
    )
    w = float(res.x)
    E = E_of_omega(w)
    return {"E_over_N2": E / (N * N), "E": E, "omega": w}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k "gaussian or v4a" -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — same-N Gaussian upper + V4a finite-N bracket"
```

---

### Task 8: Singlet check via the SU(N) Casimir (V5)

**Files:**
- Modify: `matrix_master_field/exact_diag.py`
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `structure_constants` (Task 1), `fock_ladder_ops` (Task 3), `ground_energy` (Task 5).
- Produces:
  - `casimir_of_ground_state(N: int, m: float, g: float, K: int, ground_state=None) -> float` — `Σ_{A=1}^{N²−1}⟨ψ|G_A²|ψ⟩` where `G_A=Σ_{bc}f_{Abc}(x_b p_{x,c}+y_b p_{y,c})` is the U(N) adjoint generator. `≈0 ⟺ singlet` (since `G_A` Hermitian, `⟨G_A²⟩=‖G_Aψ‖²`).

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import casimir_of_ground_state, fock_ladder_ops


def test_casimir_singlet_g0_vacuum():
    # g=0 ground state = full vacuum, annihilated by every generator => Casimir ~ 0
    c2 = casimir_of_ground_state(2, m=1.0, g=0.0, K=3)
    assert abs(c2) < 1e-9


def test_casimir_ground_state_is_singlet():
    # interacting ground state is a singlet (gauge-invariant H, invariant vacuum) -> ~0
    c2 = casimir_of_ground_state(2, m=1.0, g=0.8, K=6)
    assert abs(c2) < 1e-6


def test_casimir_nonzero_on_non_singlet():
    # a single-quantum state in one mode is NOT a singlet -> Casimir > 0 (sanity)
    N, m, K = 2, 1.0, 3
    occ, _ = fock_ladder_ops(2 * (N * N - 1), K)
    psi = np.zeros(occ.shape[0])
    one = np.zeros(2 * (N * N - 1), dtype=np.int64); one[0] = 1
    idx = {tuple(occ[r]): r for r in range(occ.shape[0])}[tuple(one)]
    psi[idx] = 1.0
    c2 = casimir_of_ground_state(N, m, g=0.0, K=K, ground_state=psi)
    assert c2 > 1e-3
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k casimir -v`
Expected: FAIL with `ImportError: cannot import name 'casimir_of_ground_state'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to matrix_master_field/exact_diag.py
def casimir_of_ground_state(N, m, g, K, ground_state=None):
    """sum_A <G_A^2> on the ground state (SU(N) Casimir); ~0 iff singlet (V5).

    G_A = -i sum_bc f_Abc (a_dag_b a_c + b_dag_b b_c) is the U(N) adjoint generator in
    NUMBER-CONSERVING ladder form (the a_b a_c and a_dag_b a_dag_c pieces of the equivalent
    x_b p_c form cancel against antisymmetric f). Because it conserves total quanta, it maps the
    total<=K space to itself with NO boundary truncation artifact (the x_b p_c form's a_b a_dag_c
    piece would be truncated at the boundary). G_A is Hermitian, so <G_A^2> = ||G_A psi||^2.
    Total-quanta truncation preserves SU(N) ([G_A, N_total]=0), so the truncated ground state is
    an exact SU(N) eigenstate and a singlet gives ~0 to eigensolver precision. m is unused by the
    generator (mode rotation is frequency-independent); it only feeds ground_energy when no state
    is supplied. x-sector ladders = slots 0..n_tl-1; y-sector = slots n_tl..2n_tl-1.
    """
    n_tl = N * N - 1
    n_modes = 2 * n_tl
    occ, ops = fock_ladder_ops(n_modes, K)
    D = occ.shape[0]
    f = structure_constants(N)

    if ground_state is None:
        ground_state = ground_energy(N, m, g, K)["ground_state"]
    psi = np.asarray(ground_state, dtype=np.complex128)

    c2 = 0.0
    for A in range(1, n_tl + 1):
        GA = sp.csr_matrix((D, D), dtype=np.complex128)
        for b in range(1, n_tl + 1):
            for c in range(1, n_tl + 1):
                fac = f[A, b, c]
                if fac != 0.0:
                    x_term = ops[b - 1].transpose() @ ops[c - 1]                    # a_dag_b a_c
                    y_term = ops[n_tl + (b - 1)].transpose() @ ops[n_tl + (c - 1)]  # b_dag_b b_c
                    GA = GA + (-1j * fac) * (x_term + y_term)
        v = GA @ psi
        c2 += float(np.real(np.vdot(v, v)))
    return c2
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k casimir -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py
git commit -m "feat(mmf): exact-diag — SU(N) Casimir singlet check (V5)"
```

---

### Task 9: N→∞ extrapolation + the deliverable (V4b, V6)

**Files:**
- Modify: `matrix_master_field/exact_diag.py` (add `extrapolate_large_N` + `__main__` deliverable)
- Create: `docs/superpowers/results/2026-06-26-m5c-exact-diag.md` (captured numbers)
- Test: `matrix_master_field/tests/test_exact_diag.py`

**Interfaces:**
- Consumes: `converge_in_K`, `gaussian_upper`, `casimir_of_ground_state` (Tasks 6–8).
- Produces:
  - `extrapolate_large_N(values: dict) -> dict` — input `{N: E_over_N2}` for `N∈{2,3}`; keys: `E_inf` (mean of the two models), `E_inf_1overN2`, `E_inf_1overN`, `uncertainty` (model spread). Honest 2-point extrapolation with *stated* assumptions.

- [ ] **Step 1: Write the failing tests**

```python
# append to matrix_master_field/tests/test_exact_diag.py
from matrix_master_field.exact_diag import extrapolate_large_N


def test_extrapolation_recovers_known_1overN2():
    # synthetic exact 1/N^2 data: e(N) = 2.2 + 0.5/N^2 -> 1/N^2 estimator recovers 2.2
    vals = {2: 2.2 + 0.5 / 4, 3: 2.2 + 0.5 / 9}
    out = extrapolate_large_N(vals)
    assert np.isclose(out["E_inf_1overN2"], 2.2, atol=1e-9)


def test_extrapolation_reports_spread_as_uncertainty():
    vals = {2: 2.30, 3: 2.28}
    out = extrapolate_large_N(vals)
    assert out["uncertainty"] == pytest.approx(
        abs(out["E_inf_1overN2"] - out["E_inf_1overN"]), abs=1e-12
    )
    assert min(out["E_inf_1overN2"], out["E_inf_1overN"]) <= out["E_inf"]
    assert out["E_inf"] <= max(out["E_inf_1overN2"], out["E_inf_1overN"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k extrapolation -v`
Expected: FAIL with `ImportError: cannot import name 'extrapolate_large_N'`.

- [ ] **Step 3: Write the implementation + deliverable driver**

```python
# append to matrix_master_field/exact_diag.py
def extrapolate_large_N(values):
    """2-point N->inf extrapolation from {N: E/N^2}, N in {2,3}.

    Reports BOTH a 1/N^2 model and a 1/N model; their spread is the (honest) uncertainty.
    Only here are the large-N reference numbers [2m, 2.365] relevant (V4b) -- never at finite N.
    """
    e2, e3 = values[2], values[3]
    # e(N) = E_inf + c / N^2
    c2 = (e2 - e3) / (1.0 / 4 - 1.0 / 9)
    einf_n2 = e2 - c2 / 4
    # e(N) = E_inf + c / N
    c1 = (e2 - e3) / (1.0 / 2 - 1.0 / 3)
    einf_n1 = e2 - c1 / 2
    return {
        "E_inf": 0.5 * (einf_n2 + einf_n1),
        "E_inf_1overN2": einf_n2,
        "E_inf_1overN": einf_n1,
        "uncertainty": abs(einf_n2 - einf_n1),
    }


def deliverable(m=1.0, lambdas=(0.0, 0.5, 1.0), K2=(8, 10, 12), K3=(4, 6)):
    """Compute the V6 table: E/N^2 at N=2,3 for each lambda, + the N->inf estimate.

    N=3 is capped at K=6 (~12s/point; padded basis ~7.4e5). Per R1 the heavier K=8 (padded
    ~5.3e6 states, minutes + several GB with the O(n_tl^3) assembly) is DEFERRED; N=3 therefore
    carries a residual K-tail (~6e-3 at lambda=1) reported honestly. N=2 is converged (tail ~1e-4
    at K=12). Probed N=3 lambda=1 anchors: K=2->2.3333, K=4->2.3133, K=6->2.3076 (monotone)."""
    report = {}
    for lam in lambdas:
        row = {}
        for N, Klist in [(2, K2), (3, K3)]:
            g = np.sqrt(lam / N) if lam > 0 else 0.0
            conv = converge_in_K(N, m, g, list(Klist))
            c2 = casimir_of_ground_state(N, m, g, Klist[-1])
            row[N] = {"E_over_N2": conv["value"], "tail": conv["tail"],
                      "series": conv["series"], "casimir": c2,
                      "gaussian": gaussian_upper(N, m, g)["E_over_N2"]}
        row["extrap"] = extrapolate_large_N({2: row[2]["E_over_N2"], 3: row[3]["E_over_N2"]})
        report[lam] = row
    return report


if __name__ == "__main__":
    rep = deliverable()
    for lam, row in rep.items():
        print(f"lambda={lam}")
        for N in (2, 3):
            r = row[N]
            print(f"  N={N}: E/N^2={r['E_over_N2']:.5f} (tail {r['tail']:.1e}, "
                  f"Casimir {r['casimir']:.1e}, Gauss {r['gaussian']:.5f})")
        ex = row["extrap"]
        print(f"  N->inf: {ex['E_inf']:.4f} +- {ex['uncertainty']:.4f} "
              f"[1/N^2 {ex['E_inf_1overN2']:.4f}, 1/N {ex['E_inf_1overN']:.4f}]")
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -k extrapolation -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full deliverable and capture numbers**

Run: `uv run --no-project --with numpy --with scipy python matrix_master_field/exact_diag.py`
Expected: a table of `E/N²` for `N=2,3` at `λ∈{0,0.5,1}` with K-tails, Casimirs `≈0`, and an `N→∞` estimate. **At `λ=0` every entry must be exactly `2.0`.** Record the stdout.

> **R1 honesty (spec):** if N=3 does not converge at feasible `K` (tail not small at `λ=1`), report the tail openly and downgrade the N=3 entry to "best K + trend"; the deliverable still adjudicates the artifact question via N=2 + the N=3 bound. Do NOT silently truncate.

- [ ] **Step 6: Write the results doc**

Create `docs/superpowers/results/2026-06-26-m5c-exact-diag.md` with: the captured table; the verification status of V1–V6; the `N→∞` estimate with its stated assumptions + uncertainty; and the one-line **adjudication** — where in `[2m, 2.365]` the truth sits and what that says about the M5c bounds and the (truncation-sensitive) free-Fisher master field. Cross-reference the spec and `[[m5-progress]]`.

- [ ] **Step 7: Run the FULL test file (all tasks together)**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_exact_diag.py -v`
Expected: PASS (all tests, ~22).

- [ ] **Step 8: Commit**

```bash
git add matrix_master_field/exact_diag.py matrix_master_field/tests/test_exact_diag.py docs/superpowers/results/2026-06-26-m5c-exact-diag.md
git commit -m "feat(mmf): exact-diag — N->inf extrapolation + deliverable (V4b, V6); ground truth complete"
```

---

## Validation obligations → task map (spec §"Validation obligations")

| Spec | Claim | Task |
|------|-------|------|
| V1 | structure constants + quartic correct | Tasks 1–2 |
| V2 | `g=0` anchor `E/N²=2m` | Task 5 |
| V3 | K-convergence | Task 6 |
| V4a | finite-N bracket `[2m, ⟨H⟩_Gauss(N)]` | Task 7 |
| V4b | large-N adjudication | Task 9 |
| V5 | singlet ground state (Casimir ≈0) | Task 8 |
| V6 | the deliverable table + N→∞ | Task 9 |

## Risk handling (spec §"Risks")
- **R1 (N=3 K-convergence):** Task 9 Step 5 reports the K-tail openly; degrades to "N=2 exact + N=3 bound/trend" if needed — never silent truncation. The N=2 path (Tasks 5–8) is the rigorously-tested spine.
- **R2 (finite-N vs large-N):** enforced structurally — `gaussian_upper` is same-N; `2.365`/planar SDP appear ONLY in `extrapolate_large_N` (Task 9). No test asserts a large-N number against a finite-N value.
- **R3 (singlet assumption):** Task 8 measures the Casimir directly; if a real ground state shows Casimir ≫0 (it should not, per the invariant-vacuum argument), the follow-up is a singlet-sector projector — out of scope here, flagged in the results doc.

## Out of scope (next sub-project, own spec)
The direct-momentum variational master field (explicit `P̂`, `[X,P]=i`, `⟨P²⟩` direct → a genuine non-exploitable upper bound), to be benchmarked against this ground truth.
