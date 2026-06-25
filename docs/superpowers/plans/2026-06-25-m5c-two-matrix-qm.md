# M5c — Two-matrix QM sandwich: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Squeeze `E/N²` of the *unsolvable* two-matrix QM `H = Tr(P_X²+P_Y²+m²(X²+Y²) − g²[X,Y]²)` (HHK Eq 17) between a **certified SDP lower bound** and a **rigorous Gaussian upper bound**, with a **novel free-Fisher operator master field** as the sharp estimate inside the bracket — anchored exactly at `g=0` (`E/N²=2m`) and cross-checked against HHK's bootstrap numbers.

**Architecture:** Seven tasks. (1) derivation file + anchor/`Φ*`-identity numerical checks (numpy). (2) Gaussian master field upper bound (closed-form Wick, numpy). (3) derive+verify the QM stationarity loop equations + SU(N) Gauss law on the `g=0` Gaussian moments — de-risks the SDP. (4) `bootstrap_two_matrix_qm` certified SDP lower bound (cvxpy). (5) free-Fisher operator master field (JAX Cuntz–Fock + conjugate-variable solve). (6) sandwich + fail-closed gate with the strengthened V6 convergence diagnostics. (7) result doc + memory + full suite. Reuses the M1–M5b stack (`bootstrap_sdp`, `sparse_fock`, `ansatz`, `loss`, `qm_collective`).

**Tech Stack:** numpy/scipy (anchor, Gaussian, `Φ*`), JAX (Cuntz–Fock operator field), cvxpy + MOSEK/CLARABEL (certified SDP), pytest.

## Global Constraints

- **Spec / conventions:** `docs/superpowers/specs/2026-06-25-m5c-two-matrix-qm-design.md`. Model (HHK arXiv:2004.10212 Eq 17): `H=Tr(P_X²+P_Y²+m²(X²+Y²)−g²[X,Y]²)`, `[X_ij,P_kl]=+iδ_il δ_jk`, `ℏ=1`, confining `−g²[X,Y]²`. Symmetries: O(2) in (X,Y) + Z₂×Z₂ (odd count of any generator vanishes).
- **'t Hooft scaling (PROVISIONAL — pinned/validated in Task 1):** `X=√N X̃`, `g²=λ/N` (`λ=Ng²` fixed). Normalized moments `m[w]=(1/N)⟨Tr w⟩=O(1)`, `m[∅]=1`. Energy density `E/N²=m[P̃_X²]+m[P̃_Y²]+m²(m[X̃²]+m[Ỹ²])−λ·m[[X̃,Ỹ]²]`. The *form* is fixed; the normalization is validated by the anchor cross-check (all three pieces must give `2m` at `λ=0`).
- **Anchor (convention-independent, test against this not guesses):** at `λ=0`, `E/N²=2m`, `m[X̃²]=m[Ỹ²]=1/(2m)`, `m[P̃_X²]=m[P̃_Y²]=m/2`, Gauss `m[X̃P̃_X]=i/2` per pair. (Physical ground energy `2N²m` of two free matrix oscillators, `/N²`.)
- **HARD GATE (audit-mandated):** no SDP coefficient, Gauss-law constant, energy normalization, or module API may be frozen until the derivations T1, T2, T4, T5 have passed their numerical checks (Tasks 1 and 3). Tasks 1–3 are that gate.
- **Free-Fisher kinetic identity (T4):** `m[P̃_X²]+m[P̃_Y²]=¼Φ*(X̃,Ỹ)`, `Φ*` = Voiculescu free Fisher information; `Φ*_X=bᵀG⁻¹b` (`G`=moment Gram, `b`=free-difference-quotient score). Reduces to M5b `∫π²σ³/3` at one matrix (semicircle variance ½ ⇒ `Φ*=2` ⇒ `¼Φ*=½`). Truncation makes `¼Φ*` a *lower* bound on KE ⇒ `E_MF` rises from below ⇒ report only with V6 diagnostics, never bracket-inclusion alone.
- **Float64:** `jax.config.update("jax_enable_x64", True)` atop every JAX file (and every JAX test before other imports, with `# noqa: E402`).
- **Runner:** `uv run --no-project --with jax --with optax --with scipy --with numpy --with cvxpy --with clarabel --with pytest python -m pytest <path> -v`. Slow pieces gated by `MMF_SLOW=1`; certification gated by `has_trusted_solver()`.
- **Git:** branch `matrix-master-field`. **Commit only the listed M5c files** (never `git add -A`). Messages `feat(mmf): …` / `docs(mmf): …`, trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---

### Task 1: Derivations file + anchor & free-Fisher identity verification (T1, T4)

The hard-gate foundation. Pure numpy/scipy — no SDP, no operators yet. Pins the anchor and the `¼Φ*` constant before anything depends on them.

**Files:**
- Create: `matrix_master_field/derivations/m5c-two-matrix-qm.md`
- Create: `matrix_master_field/free_fisher.py`
- Test: `matrix_master_field/tests/test_m5c_derivations.py`

**Interfaces:**
- Produces: `free_fisher.phi_star_density(sigma, ys) -> float` (the 1-D free Fisher information `Φ*[σ]=(4π²/3)∫σ³`); `free_fisher.free_oscillator_anchor(m) -> dict(energy, m2, p2)` (the `g=0` matrix-oscillator anchor, `E/N²=2m`, `m2=1/(2m)`, `p2=m/2`).

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_m5c_derivations.py
"""M5c — anchor (g=0 free matrix oscillator) + free-Fisher kinetic identity (T1, T4)."""
import numpy as np

from matrix_master_field.free_fisher import (
    free_oscillator_anchor,
    phi_star_density,
)


def test_g0_anchor_is_2m():
    # Two free matrix oscillators: spectrum (2n+1)m per mode, ground m, x2 matrices.
    for m in (0.5, 1.0, 2.0):
        a = free_oscillator_anchor(m)
        assert abs(a["energy"] - 2.0 * m) < 1e-12       # E/N² = 2m
        assert abs(a["m2"] - 1.0 / (2.0 * m)) < 1e-12   # m[X̃²] = 1/(2m)
        assert abs(a["p2"] - m / 2.0) < 1e-12           # m[P̃²] = m/2


def test_phi_star_semicircle_is_one():
    # Standard semicircle on [-2,2], variance 1: Φ* = 1 (Voiculescu).
    ys = np.linspace(-2.0, 2.0, 200001)
    sigma = np.sqrt(np.clip(4.0 - ys**2, 0.0, None)) / (2.0 * np.pi)
    assert abs(phi_star_density(sigma, ys) - 1.0) < 1e-4


def test_quarter_phi_star_reduces_to_m5b_kinetic():
    # M5b g=0 density σ=(1/π)√(2−y²) (variance ½): ¼Φ* = ∫π²σ³/3 = ½ = m[P̃²].
    ys = np.linspace(-np.sqrt(2.0), np.sqrt(2.0), 200001)
    sigma = np.sqrt(np.clip(2.0 - ys**2, 0.0, None)) / np.pi
    quarter_phi = 0.25 * phi_star_density(sigma, ys)
    _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    m5b_kinetic = _trapz(np.pi**2 * sigma**3 / 3.0, ys)
    assert abs(quarter_phi - m5b_kinetic) < 1e-4
    assert abs(quarter_phi - 0.5) < 1e-3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_m5c_derivations.py -v`
Expected: FAIL with `ModuleNotFoundError: matrix_master_field.free_fisher`.

- [ ] **Step 3: Write minimal implementation**

```python
# matrix_master_field/free_fisher.py
"""M5c — free Fisher information helpers + the g=0 two-matrix-QM anchor.

Φ*[σ] = (4π²/3)∫σ³ (Voiculescu free Fisher information of a 1-D density). The
two-matrix-QM ground-state kinetic energy is ¼Φ* of the joint distribution; this
module holds the 1-D building block + the exact g=0 anchor. See
`derivations/m5c-two-matrix-qm.md`.
"""
import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz  # numpy 2.0 renamed trapz


def phi_star_density(sigma, ys):
    """Free Fisher information of a 1-D probability density σ on grid ys: (4π²/3)∫σ³."""
    return float(4.0 * np.pi**2 / 3.0 * _trapz(np.asarray(sigma) ** 3, np.asarray(ys)))


def free_oscillator_anchor(m):
    """g=0 anchor: two free matrix oscillators H1=Tr P²+m²Tr X² (spectrum (2n+1)m).

    Returns the convention-independent large-N values: E/N²=2m, m[X̃²]=1/(2m),
    m[P̃²]=m/2 (per matrix m[X̃²], m[P̃²]; energy is the two-matrix total).
    """
    return {"energy": 2.0 * m, "m2": 1.0 / (2.0 * m), "p2": m / 2.0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_m5c_derivations.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Write the derivation file**

Create `matrix_master_field/derivations/m5c-two-matrix-qm.md` documenting, with the numeric checks above as the verification:
- **T1** — the pinned model (Eq 17), the `X=√N X̃`, `λ=Ng²` scaling, the energy density, and the `m²/g^{4/3}` dial; note the normalization is validated by the anchor cross-check (Tasks 2/4/5 all give `2m` at `λ=0`).
- **T4 (partial)** — `Φ*[σ]=(4π²/3)∫σ³`; the analytic semicircle check (`∫_{−2}^{2}(4−x²)^{3/2}=6π ⇒ Φ*=1`); `¼Φ*=∫π²σ³/3` (the M5b reduction); the anchor (variance ½ ⇒ `Φ*=2 ⇒ ¼Φ*=½`). (The general two-matrix `Φ*=bᵀG⁻¹b` and its `g=0` free-additivity check land in Task 5.)
- **Anchor** — the `(2n+1)m` matrix-oscillator spectrum derivation ⇒ `E/N²=2m`, `m[X̃²]=1/(2m)`.

- [ ] **Step 6: Commit**

```bash
git add matrix_master_field/free_fisher.py matrix_master_field/tests/test_m5c_derivations.py matrix_master_field/derivations/m5c-two-matrix-qm.md
git commit -m "feat(mmf): M5c anchor + free-Fisher identity (T1,T4 numeric checks)"
```

---

### Task 2: Gaussian master field — rigorous upper bound (C2)

The de-risking baseline: an explicit normalizable trial state, `⟨H⟩` by Wick, closed-form. Locks the sandwich before the research-grade pieces.

**Files:**
- Create: `matrix_master_field/qm_master_field.py`
- Test: `matrix_master_field/tests/test_qm_master_field.py`

**Interfaces:**
- Consumes: `free_fisher.free_oscillator_anchor` (anchor cross-check).
- Produces: `qm_master_field.gaussian_master_field(m, lam) -> dict(energy, omega, m2)` — minimizes `f(Ω)=Ω+m²/Ω+λ/(2Ω²)` over `Ω>0`; rigorous upper bound on `E/N²`. `qm_master_field.gaussian_comm_moment(omega, N) -> float` — the Wick value `⟨Tr[X,Y]²⟩` (for the cross-check test).

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_qm_master_field.py -v`
Expected: FAIL with `ModuleNotFoundError: matrix_master_field.qm_master_field`.

- [ ] **Step 3: Write minimal implementation**

```python
# matrix_master_field/qm_master_field.py
"""M5c — master fields for two-matrix QM (HHK Eq 17).

C2 gaussian_master_field: explicit Gaussian trial state |ψ_G(Ω)⟩, ⟨H⟩ by Wick →
rigorous upper bound (NOT via Φ*). C3 free-Fisher operator field is added in a later
task. See docs/superpowers/specs/2026-06-25-m5c-two-matrix-qm-design.md (C2/C3).
"""
import numpy as np
from scipy import optimize


def gaussian_comm_moment(omega, N):
    """Wick ⟨Tr[X,Y]²⟩ in the Gaussian ground state of Tr(P²+Ω²X²) (X,Y independent).

    ⟨X_ij X_kl⟩=a δ_il δ_jk, a=1/(2Ω): ⟨Tr XYXY⟩=a²N, ⟨Tr XYYX⟩=a²N³ ⇒
    ⟨Tr[X,Y]²⟩ = 2a²N − 2a²N³ = 2a²N(1−N²)  (leading large-N: −N³/(2Ω²)).
    """
    a = 1.0 / (2.0 * omega)
    return 2.0 * a**2 * N * (1.0 - N**2)


def gaussian_master_field(m, lam):
    """Variational E/N² over the Gaussian family: min_Ω [Ω + m²/Ω + λ/(2Ω²)].

    A rigorous upper bound (variational principle on the explicit state |ψ_G(Ω)⟩).
    Returns dict(energy, omega, m2=1/(2Ω)).
    """
    def f(omega):
        return omega + m**2 / omega + lam / (2.0 * omega**2)

    # Ω ∈ (0, ∞); minimum near m for small λ, grows like λ^{1/3} for large λ.
    res = optimize.minimize_scalar(f, bounds=(1e-6, 10.0 + 10.0 * (m + lam)), method="bounded")
    omega = float(res.x)
    return {"energy": float(f(omega)), "omega": omega, "m2": 1.0 / (2.0 * omega)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_qm_master_field.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Extend the derivation file**

Append to `derivations/m5c-two-matrix-qm.md` the C2 derivation: the Gaussian two-point functions, the Wick contractions giving `⟨Tr[X,Y]²⟩=2a²N(1−N²)`, and `E/N²(Ω)=Ω+m²/Ω+λ/(2Ω²)` with the cubic `Ω³−m²Ω−λ=0`; note `λ=0 ⇒ Ω=m ⇒ 2m` (matches the anchor). Reference the sampling cross-check test.

- [ ] **Step 6: Commit**

```bash
git add matrix_master_field/qm_master_field.py matrix_master_field/tests/test_qm_master_field.py matrix_master_field/derivations/m5c-two-matrix-qm.md
git commit -m "feat(mmf): M5c Gaussian master field — rigorous Wick upper bound (C2)"
```

---

### Task 3: QM stationarity loop equations + Gauss law — derive & verify on g=0 moments (T5, T2)

The de-risk-before-SDP step (mirrors M5b Task 3). The `g=0` ground state is Gaussian, so its ordered moments `m[w]` in `{X̃,Ỹ,P̃_X,P̃_Y}` are exact Wick values — the reference to verify every loop/Gauss relation against, before the SDP relies on them.

**Files:**
- Create: `matrix_master_field/tm_qm_relations.py`
- Test: `matrix_master_field/tests/test_m5c_loop_equations.py`

**Interfaces:**
- Produces: `tm_qm_relations.g0_moment(word, m) -> complex` — exact `g=0` ordered moment of a word (tuple of ints `0=X̃,1=Ỹ,2=P̃_X,3=P̃_Y`) via Gaussian Wick; `tm_qm_relations.stationarity_terms(word) -> list[(coeff, word)]` — the linear combination encoding `⟨[H,Tr word]⟩=0` (PROVISIONAL coefficients carrying `m²`,`λ` symbolically as callables); `tm_qm_relations.gauss_terms(O) -> list[(coeff, word)]` — `m[(2,0)+O]−m[(0,2)+O]=i·m[O]`-type relation per canonical pair.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_m5c_loop_equations.py
"""M5c — verify QM stationarity + SU(N) Gauss law on the exact g=0 Gaussian moments (T5,T2)."""
import numpy as np

from matrix_master_field.tm_qm_relations import (
    g0_moment,
    gauss_terms,
    stationarity_terms,
)

LETTERS = (0, 1, 2, 3)  # X̃, Ỹ, P̃_X, P̃_Y


def _words_upto(L):
    out = [()]
    cur = [()]
    for _ in range(L):
        cur = [w + (c,) for w in cur for c in LETTERS]
        out += cur
    return out


def test_g0_moments_match_anchor():
    m = 1.0
    assert abs(g0_moment((0, 0), m) - 1.0 / (2.0 * m)) < 1e-12   # m[X̃²]=1/(2m)
    assert abs(g0_moment((2, 2), m) - m / 2.0) < 1e-12           # m[P̃_X²]=m/2
    assert abs(g0_moment((0, 2), m) - 0.5j) < 1e-12              # Gauss: m[X̃P̃_X]=i/2
    assert abs(g0_moment((0, 1), m)) < 1e-12                     # X̃Ỹ independent → 0


def test_stationarity_residual_zero_on_g0_moments():
    m = 1.0
    for w in _words_upto(3):
        terms = stationarity_terms(w)               # at λ=0 (commutator term dropped)
        resid = sum(c(m, 0.0) * g0_moment(ww, m) for c, ww in terms)
        assert abs(resid) < 1e-10, (w, resid)


def test_gauss_law_residual_zero_on_g0_moments():
    m = 1.0
    for O in _words_upto(2):
        resid = sum(c * g0_moment(ww, m) for c, ww in gauss_terms(O))
        assert abs(resid) < 1e-10, (O, resid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_m5c_loop_equations.py -v`
Expected: FAIL with `ModuleNotFoundError: matrix_master_field.tm_qm_relations`.

- [ ] **Step 3: Write minimal implementation**

```python
# matrix_master_field/tm_qm_relations.py
"""M5c — two-matrix-QM stationarity loop equations + SU(N) Gauss law (T5, T2).

Words are tuples of ints: 0=X̃, 1=Ỹ, 2=P̃_X, 3=P̃_Y. The g=0 ground state is Gaussian,
so g0_moment(word) is an exact Wick value used to verify every relation BEFORE the SDP
(Task 4) depends on it. Heisenberg EOM (ℏ=1, [X,P]=i):
  [H,X̃]=−2i P̃_X,  [H,P̃_X]= i(2m² X̃ − 2λ·F_X),  F_X = [Ỹ,[X̃,Ỹ]] = 2ỸX̃Ỹ−Ỹ²X̃−X̃Ỹ²,
and X↔Y. ⟨[H,Tr w]⟩=0 ⇒ replace each letter by its [H,·] and sum (the QM loop equation).
"""
import numpy as np

POS = {0: 2, 1: 3}        # X̃→P̃_X, Ỹ→P̃_Y  (the conjugate momentum letter)
MOM = {2: 0, 3: 1}        # P̃_X→X̃, P̃_Y→Ỹ
OTHER = {0: 1, 1: 0}      # the other position letter


def g0_moment(word, m):
    """Exact g=0 ordered moment via Gaussian Wick on the free matrix oscillator.

    Two-point seeds (per matrix, leading large-N, normalized tr): X̃X̃=1/(2m), P̃P̃=m/2,
    X̃P̃=i/2, P̃X̃=−i/2; X and Y sectors independent. Wick-expand ordered words by summing
    over pairings with the planar (nearest-neighbour, non-crossing) contraction weights.
    """
    w = tuple(word)
    if len(w) % 2 == 1:
        return 0.0 + 0.0j
    if w == ():
        return 1.0 + 0.0j
    return _wick(w, m)


def _two_point(a, b, m):
    # sector check: X̃/P̃_X are 'X' (0,2); Ỹ/P̃_Y are 'Y' (1,3).
    sec = {0: "X", 2: "X", 1: "Y", 3: "Y"}
    if sec[a] != sec[b]:
        return 0.0 + 0.0j
    is_pos = {0: True, 1: True, 2: False, 3: False}
    pa, pb = is_pos[a], is_pos[b]
    if pa and pb:
        return 1.0 / (2.0 * m) + 0.0j      # ⟨X̃X̃⟩
    if (not pa) and (not pb):
        return m / 2.0 + 0.0j              # ⟨P̃P̃⟩
    if pa and not pb:
        return 0.5j                        # ⟨X̃P̃⟩ = i/2
    return -0.5j                           # ⟨P̃X̃⟩ = −i/2


def _wick(w, m):
    # Sum over non-crossing pairings with ordered two-point weights (planar large-N).
    n = len(w)
    if n == 0:
        return 1.0 + 0.0j
    total = 0.0 + 0.0j
    a = w[0]
    for k in range(1, n, 2):  # pair position 0 with k so [1..k-1] is closed (non-crossing)
        inner = _wick(w[1:k], m)
        outer = _wick(w[k + 1:], m)
        total += _two_point(a, w[k], m) * inner * outer
    return total


def stationarity_terms(word):
    """⟨[H,Tr word]⟩=0 as a list of (coeff(m,lam), substituted_word). At λ=0 the
    commutator force drops; the λ-terms carry F_X = 2ỸX̃Ỹ−Ỹ²X̃−X̃Ỹ² insertions."""
    w = tuple(word)
    terms = []
    for k, c in enumerate(w):
        pre, post = w[:k], w[k + 1:]
        if c in (0, 1):                       # position letter → [H,X̃]=−2i P̃_X
            terms.append((lambda m, lam, p=POS[c]: -2j, pre + (POS[c],) + post))
        else:                                  # momentum letter → [H,P̃]=i(2m² X̃ − 2λ F)
            x = MOM[c]                         # the position partner
            y = OTHER[x]
            terms.append((lambda m, lam, x=x: 2j * m**2, pre + (x,) + post))
            # −2iλ F_x with F_x = 2 y x y − y y x − x y y
            terms.append((lambda m, lam: -2j * lam * 2.0, pre + (y, x, y) + post))
            terms.append((lambda m, lam: +2j * lam, pre + (y, y, x) + post))
            terms.append((lambda m, lam: +2j * lam, pre + (x, y, y) + post))
    return terms


def gauss_terms(O):
    """SU(N) Gauss law per canonical pair: m[(2,0)+O]−m[(0,2)+O]=i·m[O] (X), and (3,1)/(1,3) (Y).
    Returned as one relation for the X pair (caller iterates pairs as needed)."""
    O = tuple(O)
    return [(1.0, (2, 0) + O), (-1.0, (0, 2) + O), (-1j, O)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with numpy --with scipy --with pytest python -m pytest matrix_master_field/tests/test_m5c_loop_equations.py -v`
Expected: PASS (3 tests). If a stationarity residual is nonzero, the EOM/sign is wrong — fix the derivation, not the test (the `g=0` moments are exact).

- [ ] **Step 5: Extend the derivation file**

Append to `derivations/m5c-two-matrix-qm.md` the T5 EOM derivation (`[H,X̃]=−2iP̃_X`, `[H,P̃_X]=i(2m²X̃−2λF_X)`, `F_X=[Ỹ,[X̃,Ỹ]]`), the resulting loop equations, and T2 (the two-pair Gauss law `m[(2,0)+O]−m[(0,2)+O]=i·m[O]`), each annotated "verified residual <1e-10 on the g=0 Gaussian moments."

- [ ] **Step 6: Commit**

```bash
git add matrix_master_field/tm_qm_relations.py matrix_master_field/tests/test_m5c_loop_equations.py matrix_master_field/derivations/m5c-two-matrix-qm.md
git commit -m "feat(mmf): M5c stationarity + Gauss law verified on g=0 moments (T5,T2)"
```

---

### Task 4: Certified SDP lower bound — `bootstrap_two_matrix_qm` (C1)

Build the SDP from the Task-3-verified relations. Extends `bootstrap_sdp.py`, reusing the single-matrix-QM scaffolding (`m(w)` accessor, real-embedded Gram, `_solve`/`with_status`).

**Files:**
- Modify: `matrix_master_field/bootstrap_sdp.py` (add `bootstrap_two_matrix_qm` + helper)
- Test: `matrix_master_field/tests/test_bootstrap_two_matrix_qm.py`

**Interfaces:**
- Consumes: `tm_qm_relations.stationarity_terms`, `tm_qm_relations.gauss_terms`; `bootstrap_sdp._solve`, `_LAST_SOLVE`, `TRUSTED_SOLVERS`, `HAS_CVXPY`, `has_trusted_solver`.
- Produces: `bootstrap_sdp.bootstrap_two_matrix_qm(m, lam, L=3, *, maximize=False, with_status=False)` → `E/N²` bound (float), or `(value, solver, status)` if `with_status`.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_bootstrap_two_matrix_qm.py
"""M5c — certified SDP lower bound for two-matrix QM (HHK Eq 17)."""
import pytest

from matrix_master_field.bootstrap_sdp import (
    HAS_CVXPY,
    bootstrap_two_matrix_qm,
    has_trusted_solver,
)
from matrix_master_field.qm_master_field import gaussian_master_field

pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


def test_lower_bound_anchor_lambda0():
    # λ=0 → E/N²=2m exactly; the SDP lower bound must not exceed it.
    lb = bootstrap_two_matrix_qm(1.0, 0.0, L=3, maximize=False)
    assert lb is not None
    assert lb <= 2.0 + 1e-4
    assert lb >= 2.0 - 0.5        # not a trivial ≥0 collapse


def test_lower_bound_brackets_gaussian():
    # E_lo (SDP) ≤ E/N² ≤ E_hi (Gaussian) for λ>0.
    for lam in (0.5, 1.0):
        lb = bootstrap_two_matrix_qm(1.0, lam, L=3, maximize=False)
        ub = gaussian_master_field(1.0, lam)["energy"]
        assert lb is not None
        assert lb <= ub + 1e-3


@pytest.mark.skipif(not has_trusted_solver(), reason="certification needs CLARABEL/MOSEK")
def test_sdp_certified():
    lb, solver, status = bootstrap_two_matrix_qm(1.0, 1.0, L=3, maximize=False, with_status=True)
    assert lb is not None
    assert solver in ("MOSEK", "CLARABEL") and status == "optimal"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_bootstrap_two_matrix_qm.py -v`
Expected: FAIL with `ImportError: cannot import name 'bootstrap_two_matrix_qm'`.

- [ ] **Step 3: Write minimal implementation**

Add to `matrix_master_field/bootstrap_sdp.py` (after `bootstrap_single_matrix_qm`):

```python
def _tm_qm_words_upto(L):
    out, cur = [()], [()]
    for _ in range(L):
        cur = [w + (c,) for w in cur for c in (0, 1, 2, 3)]  # X̃,Ỹ,P̃_X,P̃_Y
        out += cur
    return out


def bootstrap_two_matrix_qm(m, lam, L=3, *, maximize=False, with_status=False):
    """Certified bound on E/N² for HHK Eq 17 H=Tr(P_X²+P_Y²+m²(X²+Y²)−g²[X,Y]²).

    Ordered single-trace moments in {X̃,Ỹ,P̃_X,P̃_Y}; stationarity (T5) + SU(N) Gauss law
    (T2) + Gram PSD. Minimizing the relaxation gives a certified lower bound on E/N².
    Reuses the M5b single-matrix-QM machinery (real-embedded Hermitian Gram, _solve).
    """
    from matrix_master_field.tm_qm_relations import stationarity_terms
    if not HAS_CVXPY:
        return (None, None, None) if with_status else None

    allw = [w for w in _tm_qm_words_upto(L) if len(w) % 2 == 0]
    var = {w: cp.Variable(complex=True) for w in allw if w != ()}

    def mm(w):
        w = tuple(w)
        if len(w) % 2 == 1:
            return 0.0 + 0j
        if w == ():
            return 1.0 + 0j
        return var.get(w, None)

    cons = []
    # stationarity ⟨[H,Tr w]⟩=0
    for w in _tm_qm_words_upto(max(1, L - 2)):
        expr, ok = 0, True
        for coeff, ww in stationarity_terms(w):
            t = mm(ww)
            if t is None:
                ok = False
                break
            expr = expr + coeff(m, lam) * t
        if ok and not isinstance(expr, (int, float, complex)):
            cons += [cp.real(expr) == 0, cp.imag(expr) == 0]
    # SU(N) Gauss law, both canonical pairs
    for O in _tm_qm_words_upto(L - 2):
        for pair in ((2, 0), (3, 1)):
            rel = [(1.0, pair + O), (-1.0, (pair[1], pair[0]) + O), (-1j, O)]
            terms = [(c, mm(ww)) for c, ww in rel]
            if any(t is None for _, t in terms):
                continue
            e = sum(c * t for c, t in terms)
            cons += [cp.real(e) == 0, cp.imag(e) == 0]
    # Gram PSD (complex Hermitian via real embedding), basis = words up to L//2
    basis = _tm_qm_words_upto(L // 2)
    A = [[None] * len(basis) for _ in basis]
    B = [[None] * len(basis) for _ in basis]
    for i, u in enumerate(basis):
        for j, v in enumerate(basis):
            e = mm(tuple(reversed(u)) + v)
            if e is None:
                return (None, None, None) if with_status else None
            A[i][j], B[i][j] = cp.real(e), cp.imag(e)
    embed = cp.bmat([[cp.bmat(A), -cp.bmat(B)], [cp.bmat(B), cp.bmat(A)]])
    cons.append(embed >> 0)

    # E/N² = m[P̃_X²]+m[P̃_Y²]+m²(m[X̃²]+m[Ỹ²]) − λ·m[[X̃,Ỹ]²]
    comm2 = (mm((0, 1, 0, 1)) - mm((0, 1, 1, 0)) - mm((1, 0, 0, 1)) + mm((1, 0, 1, 0)))
    energy = (mm((2, 2)) + mm((3, 3)) + m**2 * (mm((0, 0)) + mm((1, 1))) - lam * comm2)
    obj = cp.Maximize(cp.real(energy)) if maximize else cp.Minimize(cp.real(energy))
    prob = _solve(cp.Problem(obj, cons))
    val = None if prob.status not in ("optimal", "optimal_inaccurate") else float(prob.value)
    if with_status:
        return val, _LAST_SOLVE["solver"], _LAST_SOLVE["status"]
    return val
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run --no-project --with numpy --with scipy --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_bootstrap_two_matrix_qm.py -v`
Expected: PASS (3 tests; the certified one runs only with CLARABEL/MOSEK). If `λ=0` lower bound exceeds `2m`, the stationarity/Gauss wiring is wrong — re-check against Task 3.

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/bootstrap_sdp.py matrix_master_field/tests/test_bootstrap_two_matrix_qm.py
git commit -m "feat(mmf): M5c certified SDP lower bound bootstrap_two_matrix_qm (C1)"
```

---

### Task 5: Free-Fisher operator master field (C3)

The novel core: two `X̃,Ỹ` as ML-optimized Cuntz–Fock operators; kinetic energy `¼Φ*` via the conjugate-variable solve `Φ*_X=bᵀG⁻¹b`. Minimize `¼Φ*+potential`. Validated against the now-existing `[E_lo, E_hi]` bracket.

**Files:**
- Modify: `matrix_master_field/qm_master_field.py` (add `free_fisher_information`, `fisher_master_field`)
- Modify: `matrix_master_field/tests/test_qm_master_field.py` (add C3 tests)

**Interfaces:**
- Consumes: `sparse_fock.SparseMonomialField`, `sparse_fock.SuffixSharedMoments`; `free_fisher.phi_star_density` (reduction check).
- Produces: `qm_master_field.free_fisher_information(moment, basis_words, n_matrices) -> (phi_star, cond)` — `Φ*=Σ_a bᵀG⁻¹b`, plus the Gram condition number (V6 diagnostic). `qm_master_field.fisher_master_field(m, lam, *, cutoff, degree, max_word_len, steps, lr, seed) -> dict(energy, m2, comm2, phi_cond, params)`.

- [ ] **Step 1: Write the failing test**

```python
# add to matrix_master_field/tests/test_qm_master_field.py
import math
import os
import pytest


def test_free_fisher_reduces_to_one_matrix_semicircle():
    # n=1 semicircular moments (variance ½): Φ*=2 ⇒ ¼Φ*=½ (the M5b kinetic anchor).
    from matrix_master_field.qm_master_field import free_fisher_information

    def semicircle_moment(word):  # word in (0,), variance v=½ semicircle: even moments = v^n C_n
        k = len(word)
        if k % 2 == 1:
            return 0.0
        n = k // 2
        catalan = math.comb(2 * n, n) // (n + 1)
        return (0.5 ** n) * catalan

    basis = [(), (0,), (0, 0), (0, 0, 0)]
    phi, cond = free_fisher_information(semicircle_moment, basis, n_matrices=1)
    assert abs(0.25 * phi - 0.5) < 1e-6


@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: Cuntz–Fock optimization; set MMF_SLOW=1")
def test_fisher_master_field_anchor_lambda0():
    from matrix_master_field.qm_master_field import fisher_master_field
    r = fisher_master_field(1.0, 0.0, cutoff=8, degree=2, max_word_len=3, steps=1500, lr=5e-3)
    assert abs(r["energy"] - 2.0) < 5e-2     # E/N² → 2m at λ=0
    assert abs(r["m2"] - 0.5) < 5e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with jax --with optax --with pytest python -m pytest matrix_master_field/tests/test_qm_master_field.py::test_free_fisher_reduces_to_one_matrix_semicircle -v`
Expected: FAIL with `ImportError: cannot import name 'free_fisher_information'`.

- [ ] **Step 3: Write minimal implementation**

Add to `matrix_master_field/qm_master_field.py` (top: `import jax; jax.config.update("jax_enable_x64", True); import jax.numpy as jnp`):

```python
def _free_diff_quotient_score(moment, basis_words, a):
    """b[c] = τ⊗τ(∂_a w_c) = Σ_{i: w_c[i]=a} moment(w_c[:i])·moment(w_c[i+1:])."""
    b = []
    for w in basis_words:
        s = 0.0
        for i, c in enumerate(w):
            if c == a:
                s = s + moment(tuple(w[:i])) * moment(tuple(w[i + 1:]))
        b.append(s)
    return jnp.asarray(b)


def _phi_star(moment, basis_words, n_matrices):
    """Φ* = Σ_a bᵀG⁻¹b, G[c,c']=moment(reverse(w_c)+w_c'). Differentiable hot path (no cond).

    `moment(word)->scalar` is the backend (numpy callable, or a JAX SuffixSharedMoments
    closure — both differentiable).
    """
    nb = len(basis_words)
    G = jnp.asarray([[moment(tuple(reversed(u)) + tuple(v)) for v in basis_words]
                     for u in basis_words]).reshape(nb, nb)
    phi = 0.0
    for a in range(n_matrices):
        b = _free_diff_quotient_score(moment, basis_words, a)
        phi = phi + jnp.real(b @ jnp.linalg.solve(G, b))
    return phi


def free_fisher_information(moment, basis_words, n_matrices):
    """(Φ*, cond(G)) — Φ* plus the Gram condition number (V6 diagnostic; call once, NOT in
    the optimization loop — the SVD in `cond` is unnecessary there). Truncating `basis_words`
    LOWERS Φ* (sup characterization), so ¼Φ* is a one-sided/from-below estimate of KE.
    """
    nb = len(basis_words)
    G = jnp.asarray([[moment(tuple(reversed(u)) + tuple(v)) for v in basis_words]
                     for u in basis_words]).reshape(nb, nb)
    cond = float(jnp.linalg.cond(jnp.real(G)))
    return _phi_star(moment, basis_words, n_matrices), cond


def fisher_master_field(m, lam, *, cutoff=8, degree=2, max_word_len=3,
                        steps=1500, lr=5e-3, seed=0):
    """Minimize ¼Φ*(X̃,Ỹ) + m²(m[X̃²]+m[Ỹ²]) − λ·m[[X̃,Ỹ]²] over Cuntz–Fock operators.

    Positivity automatic (tracial vacuum). Returns dict(energy, m2, comm2, phi_cond, params).
    """
    import optax
    from matrix_master_field.sparse_fock import SparseMonomialField, SuffixSharedMoments

    field = SparseMonomialField(n_matrices=2, cutoff=cutoff, degree=degree)
    basis = [w for w in _tm_position_words(max_word_len)]
    # words the loss reads: Gram (reverse(u)+v), score splits, energy words, comm2 words.
    needed = set()
    for u in basis:
        for v in basis:
            needed.add(tuple(reversed(u)) + tuple(v))
    for w in basis:
        for i in range(len(w)):
            needed.add(tuple(w[:i]))
            needed.add(tuple(w[i + 1:]))
    for w in [(0, 0), (1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        needed.add(w)
    shared = SuffixSharedMoments(field, sorted(needed, key=len))

    def energy_fn(params):
        moment = shared.moment_fn(params)
        phi = _phi_star(moment, basis, n_matrices=2)
        comm2 = (moment((0, 1, 0, 1)) - moment((0, 1, 1, 0))
                 - moment((1, 0, 0, 1)) + moment((1, 0, 1, 0)))
        pot = m**2 * (moment((0, 0)) + moment((1, 1))) - lam * comm2
        return 0.25 * phi + pot

    params = field.params_for_free_field()
    opt = optax.adam(lr)
    state = opt.init(params)
    val_and_grad = jax.value_and_grad(energy_fn)
    for _ in range(steps):
        _, g = val_and_grad(params)
        updates, state = opt.update(g, state)
        params = optax.apply_updates(params, updates)

    moment = shared.moment_fn(params)
    phi, cond = free_fisher_information(moment, basis, n_matrices=2)
    comm2 = float((moment((0, 1, 0, 1)) - moment((0, 1, 1, 0))
                   - moment((1, 0, 0, 1)) + moment((1, 0, 1, 0))).real)
    return {
        "energy": float(energy_fn(params)),
        "m2": float(moment((0, 0)).real),
        "comm2": comm2,
        "phi_cond": cond,
        "params": params,
    }


def _tm_position_words(L):
    out, cur = [()], [()]
    for _ in range(L):
        cur = [w + (c,) for w in cur for c in (0, 1)]  # only X̃,Ỹ in the Φ* basis
        out += cur
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (fast reduction): `uv run --no-project --with numpy --with scipy --with jax --with optax --with pytest python -m pytest matrix_master_field/tests/test_qm_master_field.py::test_free_fisher_reduces_to_one_matrix_semicircle -v`
Expected: PASS.
Run (slow anchor): `MMF_SLOW=1 uv run --no-project --with numpy --with scipy --with jax --with optax --with pytest python -m pytest matrix_master_field/tests/test_qm_master_field.py::test_fisher_master_field_anchor_lambda0 -v`
Expected: PASS (`E/N²≈2` at λ=0).

- [ ] **Step 5: Extend the derivation file**

Append the T4 completion: `Φ*_X=bᵀG⁻¹b` from the conjugate-variable relation `τ(ξ_X w)=τ⊗τ(∂_X w)`; the sup-characterization (⇒ truncation lower-bounds `Φ*`); the `g=0` free-additivity check (`Φ*(X̃,Ỹ)=Φ*(X̃)+Φ*(Ỹ)=4 ⇒ ¼Φ*=1`).

- [ ] **Step 6: Commit**

```bash
git add matrix_master_field/qm_master_field.py matrix_master_field/tests/test_qm_master_field.py matrix_master_field/derivations/m5c-two-matrix-qm.md
git commit -m "feat(mmf): M5c free-Fisher operator master field (C3, conjugate-variable Φ*)"
```

---

### Task 6: Sandwich + fail-closed gate with V6 diagnostics

Assemble `solve_two_matrix_qm` mirroring `solve_single_matrix_qm`; gate `_tm_qm_gate` enforces the audit-strengthened V6 (bracket inclusion necessary-not-sufficient; require convergence + conditioning + tolerances).

**Files:**
- Modify: `matrix_master_field/train.py` (add `solve_two_matrix_qm`, `_tm_qm_gate`)
- Test: `matrix_master_field/tests/test_train_two_matrix_qm.py`

**Interfaces:**
- Consumes: `bootstrap_two_matrix_qm`, `gaussian_master_field`, `fisher_master_field`, `TRUSTED_SOLVERS`, `HAS_CVXPY`, `has_trusted_solver`.
- Produces: `train.solve_two_matrix_qm(m, lam, *, L=3, cutoff=8, degree=2, max_word_len=3, validate=True, e_tol=1e-2) -> dict(m, lam, E_lo, E_hi, E_mf, m2, validation, validated)`.

- [ ] **Step 1: Write the failing test**

```python
# matrix_master_field/tests/test_train_two_matrix_qm.py
"""M5c — the full two-matrix-QM sandwich + fail-closed gate."""
import os
import pytest

from matrix_master_field.bootstrap_sdp import HAS_CVXPY, has_trusted_solver
from matrix_master_field.train import solve_two_matrix_qm


def test_gate_rejects_bare_bracket_inclusion():
    # V6: bracket inclusion alone must NOT validate (no convergence/conditioning evidence).
    from matrix_master_field.train import _tm_qm_gate
    val, ok = _tm_qm_gate(E_lo=1.0, E_lo_cert=True, E_hi=3.0, E_mf=2.0,
                          mf_converged=False, phi_cond=1e3, e_tol=1e-2)
    assert ok is False
    assert val["in_bracket"] is True        # inside the bracket…
    assert val["mf_converged"] is False     # …but not converged → not validated


@pytest.mark.skipif(not (HAS_CVXPY and has_trusted_solver()),
                    reason="certified sandwich needs CLARABEL/MOSEK")
@pytest.mark.skipif(not os.environ.get("MMF_SLOW"),
                    reason="slow: SDP + Cuntz–Fock optimization; set MMF_SLOW=1")
def test_solve_two_matrix_qm_validated_lambda0():
    r = solve_two_matrix_qm(1.0, 0.0, L=3)
    assert r["validated"] is True
    assert r["validation"]["E_lo"] <= 2.0 + 1e-2 <= r["E_hi"] + 1e-2   # the squeeze on 2m
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-project --with numpy --with scipy --with jax --with optax --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_train_two_matrix_qm.py::test_gate_rejects_bare_bracket_inclusion -v`
Expected: FAIL with `ImportError: cannot import name 'solve_two_matrix_qm'`.

- [ ] **Step 3: Write minimal implementation**

Add to `matrix_master_field/train.py` (import `bootstrap_two_matrix_qm` from bootstrap_sdp, `gaussian_master_field`/`fisher_master_field` from qm_master_field):

```python
def _tm_qm_gate(*, E_lo, E_lo_cert, E_hi, E_mf, mf_converged, phi_cond, e_tol,
                cond_max=1e8):
    """V6 (audit-strengthened): bracket inclusion is necessary, NOT sufficient.

    validated ⇔ certified lower bound AND E_lo≤E_hi bracket AND E_mf in-bracket AND
    E_mf converged vs Fisher basis degree AND Gram well-conditioned.
    """
    in_bracket = (E_lo is not None and E_hi is not None and E_mf is not None
                  and (E_lo - e_tol) <= E_mf <= (E_hi + e_tol))
    bracket_ok = E_lo is not None and E_hi is not None and E_lo <= E_hi + e_tol
    cond_ok = phi_cond is not None and phi_cond < cond_max
    validation = {
        "E_lo": E_lo, "E_hi": E_hi, "E_mf": E_mf,
        "in_bracket": in_bracket, "bracket_ok": bracket_ok,
        "mf_converged": bool(mf_converged), "phi_cond": phi_cond, "cond_ok": cond_ok,
        "certified": bool(E_lo_cert),
    }
    validated = bool(E_lo_cert and bracket_ok and in_bracket
                     and mf_converged and cond_ok)
    return validation, validated


def solve_two_matrix_qm(m, lam, *, L=3, cutoff=8, degree=2, max_word_len=3,
                        validate=True, e_tol=1e-2):
    """Sandwich for HHK Eq 17: SDP lower (C1) + Gaussian upper (C2) + free-Fisher (C3).

    Returns dict(m, lam, E_lo, E_hi, E_mf, m2, validation, validated). Fail-closed:
    `validated` requires a certified SDP bracket AND a converged, well-conditioned E_mf.
    """
    E_hi = gaussian_master_field(m, lam)["energy"]
    mf = fisher_master_field(m, lam, cutoff=cutoff, degree=degree, max_word_len=max_word_len)
    E_mf = mf["energy"]

    # convergence of E_mf vs the Fisher basis degree (must plateau, rising from below).
    mf_lo = fisher_master_field(m, lam, cutoff=cutoff, degree=degree,
                                max_word_len=max(1, max_word_len - 1))
    mf_converged = abs(E_mf - mf_lo["energy"]) < 5e-2

    out = {"m": m, "lam": lam, "E_hi": E_hi, "E_mf": E_mf, "m2": mf["m2"]}
    if validate and HAS_CVXPY:
        E_lo, solver, status = bootstrap_two_matrix_qm(m, lam, L=L, maximize=False,
                                                       with_status=True)
        cert = status == "optimal" and solver in TRUSTED_SOLVERS
        val, ok = _tm_qm_gate(E_lo=E_lo, E_lo_cert=cert, E_hi=E_hi, E_mf=E_mf,
                              mf_converged=mf_converged, phi_cond=mf["phi_cond"],
                              e_tol=e_tol)
        out.update(E_lo=E_lo, validation=val, validated=ok)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run (fast gate logic): `uv run --no-project --with numpy --with scipy --with jax --with optax --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_train_two_matrix_qm.py::test_gate_rejects_bare_bracket_inclusion -v`
Expected: PASS.
Run (slow validated sandwich): `MMF_SLOW=1 uv run --no-project --with numpy --with scipy --with jax --with optax --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_train_two_matrix_qm.py -v`
Expected: PASS (`validated=True` at λ=0).

- [ ] **Step 5: Commit**

```bash
git add matrix_master_field/train.py matrix_master_field/tests/test_train_two_matrix_qm.py
git commit -m "feat(mmf): M5c sandwich solve_two_matrix_qm + V6 fail-closed gate"
```

---

### Task 7: Result doc + full suite + memory

**Files:**
- Create: `docs/superpowers/results/2026-06-25-m5c-two-matrix-qm.md`
- Update: `/Users/deliangzhong/.claude/projects/-Users-deliangzhong-Documents-Working-Master-Field/memory/m5-progress.md` and `MEMORY.md`

- [ ] **Step 1: Run the full M5c suite (fast + slow)**

Run: `MMF_SLOW=1 uv run --no-project --with numpy --with scipy --with jax --with optax --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/test_m5c_derivations.py matrix_master_field/tests/test_qm_master_field.py matrix_master_field/tests/test_m5c_loop_equations.py matrix_master_field/tests/test_bootstrap_two_matrix_qm.py matrix_master_field/tests/test_train_two_matrix_qm.py -v`
Expected: all green (slow ones included).

- [ ] **Step 2: Run the whole package suite (no regressions)**

Run: `uv run --no-project --with numpy --with scipy --with jax --with optax --with cvxpy --with clarabel --with pytest python -m pytest matrix_master_field/tests/ -q`
Expected: prior count + the new M5c tests, all passing (slow gated off without `MMF_SLOW`).

- [ ] **Step 3: Write the result doc**

Create `docs/superpowers/results/2026-06-25-m5c-two-matrix-qm.md` (mirror the M5b result doc): the model, the staged sandwich, the `E_lo ≤ E_mf ≤ E_hi` table at `λ∈{0,0.5,1}`, the `λ=0` anchor agreement, the honest rigor note (C3 from below; C2 the rigorous cap), the HHK soft cross-check status, and what stays open.

- [ ] **Step 4: Update memory**

Append M5c status to `m5-progress.md` (done; the unsolvable rung; Gaussian rigorous cap + certified SDP + novel free-Fisher operator field; the `¼Φ*=bᵀG⁻¹b` recipe + the from-below one-sidedness lesson). Add the `MEMORY.md` index line.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/results/2026-06-25-m5c-two-matrix-qm.md
git commit -m "docs(mmf): M5c result — two-matrix QM sandwich (Gaussian cap + SDP + free-Fisher)"
```

(Memory files live outside the repo; write them but do not `git add`.)

---

## Notes for the implementer
- **Order matters:** Tasks 1→3 are the hard gate (anchor, identities, verified loop/Gauss relations) and MUST pass before Task 4 (SDP) and Task 6 (gate) rely on them. Task 5 (free-Fisher) needs only Task 1's identity but is *validated* against Tasks 2+4.
- **If a `g=0` check fails, the physics is wrong, not the test** — the `g=0` moments are exact Gaussian Wick values; fix the EOM/sign/normalization in the derivation.
- **`L=3` is HHK's convergence order**; grow `L` (and `max_word_len` for C3) if a bracket is loose. Watch the V6 convergence/conditioning diagnostics — never report `E_mf` on bracket inclusion alone.
- **Provisional normalization:** if the `λ=0` anchor (`2m`) is missed by any piece, re-pin the `X=√N X̃` / `λ=Ng²` scaling (T1) before trusting `λ>0` numbers.
