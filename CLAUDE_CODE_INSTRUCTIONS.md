# Master Field ML Computation — Claude Code Instructions

## Mission

Numerically construct the **master field** for large-$N$ matrix models using ML optimisation. The master field is the $N=\infty$ saddle point of the matrix path integral; all gauge-invariant observables are computable from it without any functional integration. An explicit construction in QCD₄ would be one of the most important results in theoretical physics.

The physics framework is from:
- Gopakumar & Gross, "Mastering the Master Field" (hep-th/9411021) — attached
- Jevicki, Sakita, Rodrigues — collective field / loop-space optimisation
- Lin, Kazakov, Zheng — bootstrap / SDP bounds

## Environment & Dependencies

```bash
pip install jax jaxlib numpy scipy matplotlib optax flax cvxpy torch --break-system-packages
```

All computation is in **Python** (JAX for autodiff + JIT, cvxpy for SDP validation). This is correct because:
1. JAX gives automatic differentiation through the non-convex loss landscape
2. JAX's JIT compiles the inner loop to XLA — competitive with C++ for this problem
3. The constraint structure (PSD matrices, polynomial equations) maps cleanly to JAX's pytree API
4. cvxpy interfaces directly with SDP solvers (SCS, MOSEK) for bootstrap bounds

## Project Structure

```
master_field/
├── INSTRUCTIONS.md            ← you are here
├── config.py                  # Hyperparameters, model registry
├── cuntz_fock.py              # Cuntz algebra, Boltzmann Fock space, operator reps
├── one_matrix.py              # Exact one-matrix solutions (resolvent, density, moments)
├── schwinger_dyson.py         # Loop equations for multi-matrix models
├── neural_master_field.py     # Neural ansätze + JAX training loops
├── bootstrap_sdp.py           # SDP bounds via cvxpy
├── train.py                   # CLI entry point
├── visualize.py               # Plots
└── results/                   # Output directory
```

---

## Physics Summary (what the code computes)

### The master field problem

For a matrix model with action $S[M_1, \ldots, M_n]$ invariant under $M_i \to U M_i U^\dagger$, the large-$N$ factorisation theorem says:

$$\langle \mathcal{O}_1 \mathcal{O}_2 \rangle = \langle \mathcal{O}_1 \rangle \langle \mathcal{O}_2 \rangle + O(1/N^2)$$

for gauge-invariant observables $\mathcal{O}_k = \frac{1}{N}\text{Tr}[\text{words in } M_i]$. This means the path integral measure concentrates on a single configuration — the **master field** $\bar{M}_i$ — and:

$$\langle \mathcal{O} \rangle = \text{tr}[\bar{M}_{i_1} \bar{M}_{i_2} \cdots \bar{M}_{i_k}]$$

where tr is a normalised trace on an infinite-dimensional operator algebra.

### What we optimise

The gauge-invariant data of the master field is the set of **loop moments**:

$$\Omega(C) = \text{tr}[\bar{M}_{i_1} \bar{M}_{i_2} \cdots \bar{M}_{i_k}], \qquad C = (i_1, i_2, \ldots, i_k)$$

These satisfy:
1. **Schwinger-Dyson (loop) equations** — exact recursion relations from the action
2. **Positive semidefiniteness** — the moment matrix $\Omega_{ij} = \text{tr}[w_i^\dagger w_j] \succeq 0$
3. **Cyclicity** — $\Omega(C) = \Omega(\text{cyclic permutations of } C)$
4. **Hermiticity** — $\Omega(C)^* = \Omega(C^{-1})$ (reversed word)
5. **Normalisation** — $\Omega(\emptyset) = 1$

The master field is the unique solution satisfying all of these simultaneously.

### The Schwinger-Dyson equations

For a single matrix with potential $V(M)$, the SD equation for test word $M^n$ is:

$$\sum_k v_k \, m_{n+k} = \sum_{j=0}^{n-1} m_j \, m_{n-j-1}$$

where $V'(M) = \sum_k v_k M^k$ and $m_k = \text{tr}[M^k]$.

**CRITICAL**: The LHS index is $n+k$, NOT $n+k-1$. This comes from $\text{tr}[V'(M) \cdot M^n] = \sum_k v_k \, \text{tr}[M^{k+n}]$.

For two matrices with $S = \text{Tr}[M_1^2/2 + M_2^2/2 - (g^2/4)[M_1, M_2]^2]$:

$$V'_a = M_a + \frac{g^2}{2}(M_a M_b^2 + M_b^2 M_a - 2 M_b M_a M_b)$$

The SD equation for derivative w.r.t. $M_a$ on test word $w = M_{i_1} \cdots M_{i_k}$:

$$\langle \text{tr}[V'_a \cdot w] \rangle = \sum_{\substack{m=1 \\ i_m = a}}^{k} \langle \text{tr}[w_{\text{left of } m}] \rangle \langle \text{tr}[w_{\text{right of } m}] \rangle$$

The RHS "splits" the trace at every position where the derivative variable $M_a$ appears.

---

## Computational Pipeline

### Stage 0: Sanity checks (run FIRST, must ALL pass)

```bash
python cuntz_fock.py
```

**Expected output:**
- `✓ Cuntz algebra verified (n=1, L=6, dim=7)`
- Gaussian moments = Catalan numbers: $m_0=1, m_2=1, m_4=2, m_6=5, m_8=14, m_{10}=42$
- All to machine precision ($< 10^{-12}$)

```bash
python one_matrix.py
```

**Expected output:**
- Gaussian free cumulants: $\kappa_1=0, \kappa_2=1, \kappa_{k\geq 3}=0$
- Quartic ($g=0.5$) moments: $m_2 \approx 0.5162, m_4 \approx 0.4838$
- Quartic free cumulants: $\kappa_2 > 0, \kappa_4 < 0$ (interaction shifts the distribution)

```bash
python schwinger_dyson.py
```

**Expected output:**
- Gaussian SD residuals: $\max|\text{residual}| < 10^{-12}$
- Quartic SD residuals: $\max|\text{residual}| < 10^{-6}$ (limited by numerical integration)

### Stage 1: One-matrix models (exactly solvable — MUST match)

#### 1a. Gaussian model

```bash
python train.py --model gaussian --validate --max_word_length 12 --n_epochs 3000 --lr 1e-2
```

**Success criterion:** All even moments match Catalan numbers to $< 10^{-6}$. This is a TRIVIAL test — if it fails, the SD equation implementation is wrong.

**What to check:**
- The loss should drop to $< 10^{-10}$ (SD equations are trivially satisfied)
- $m_2 = 1.0000, m_4 = 2.0000, m_6 = 5.0000$
- Free cumulants: $\kappa_2 = 1.0000$, all others $\approx 0$

#### 1b. Quartic model

```bash
python train.py --model quartic --coupling 0.5 --validate --max_word_length 14 --n_epochs 5000
python train.py --model quartic --coupling 1.0 --validate --max_word_length 14 --n_epochs 8000
python train.py --model quartic --coupling 5.0 --validate --max_word_length 14 --n_epochs 10000
```

**Success criterion:** Moments match exact values (from eigenvalue density integration) to $< 10^{-4}$.

**Interesting physics:** As $g$ increases, the eigenvalue density develops heavier tails and the free cumulants $\kappa_{2k}$ for $k \geq 2$ grow — the distribution departs from semicircular.

#### 1c. Scan over coupling

```bash
for g in 0.1 0.2 0.5 1.0 2.0 5.0 10.0; do
    python train.py --model quartic --coupling $g --validate --n_epochs 5000
done
```

This produces a family of master fields parametrised by $g$. The R-transform coefficients $\kappa_n(g)$ should be smooth functions of $g$.

### Stage 2: Bootstrap validation

```bash
python train.py --model quartic --coupling 0.5 --bootstrap --max_word_length 10
```

This runs the SDP bootstrap to produce rigorous upper/lower bounds on each moment. The ML solution must lie within these bounds. If it doesn't, the ML has found a spurious local minimum.

### Stage 3: Two coupled matrices (the real challenge)

```bash
python train.py --model two_matrix_coupled --coupling 1.0 --max_word_length 6 --n_epochs 20000 --lr 5e-4 --interaction commutator_squared
```

**What makes this hard:**
- The word space grows as $2^L$ (exponentially in truncation level)
- No closed-form solution exists
- The moment matrix is large: at $L=6$, the basis dimension is $\sim 100$, so Cholesky has $\sim 5000$ parameters
- The SD equations involve 4th-order interaction terms

**Success criteria:**
- Loss $< 10^{-4}$
- Moment matrix is PSD: $\min \text{eig}(\Omega) > -10^{-8}$
- Results are stable under increasing $L$ (convergence in truncation)

**Benchmark:** Compare against Rodrigues et al. JHEP 2024 for Yang-Mills matrix QM at the same coupling.

### Stage 4: Scaling study

Run at increasing truncation levels $L = 4, 6, 8, 10$ and check that moments converge:

```bash
for L in 4 6 8 10; do
    python train.py --model two_matrix_coupled --coupling 1.0 --max_word_length $L --n_epochs 20000
done
```

The first few moments ($m_2, m_4$) should stabilise quickly; higher moments need larger $L$.

---

## Known Bugs and Pitfalls (read before debugging)

### 1. SD equation indexing

The most common bug: the SD equation LHS involves $\sum_k v_k m_{n+k}$ (where $V'(M) = \sum_k v_k M^k$), NOT $m_{n+k-1}$. The factor comes from $\text{tr}[M^k \cdot M^n] = m_{k+n}$.

### 2. Cyclic equivalence

Loop moments are cyclic: $\text{tr}[M_1 M_2] = \text{tr}[M_2 M_1]$. The canonical representative is the lexicographically smallest rotation. If you don't reduce to canonical form, you'll have redundant variables and the optimisation will be ill-conditioned.

### 3. PSD enforcement

If using the Cholesky parametrisation $\Omega = L L^T$, the diagonal of $L$ must be positive. Use $L_{ii} = \exp(\ell_i)$ where $\ell_i$ is the raw parameter. If you forget the exponential, the diagonal can go negative and $\Omega$ is no longer PSD.

### 4. Normalisation

$m_0 = \text{tr}[I] = 1$ is a hard constraint, not something to optimise. Either fix it by construction or add a very large penalty ($\lambda \sim 100$).

### 5. Symmetry

For potentials symmetric under $M \to -M$ (e.g., $V = M^2/2 + g M^4/4$), all odd moments vanish. Enforce this by construction (parametrise only even moments). If you optimise over all moments, the odd moments will introduce spurious flat directions.

### 6. Large-$L$ divergence

At large truncation $L$, the moments $m_{2k}$ grow factorially (like $(2k)!/(k!(k+1)!)$ for the Gaussian). The loss function should use RELATIVE residuals or normalise each SD equation by its scale.

### 7. JAX issues

- Use `jax.config.update('jax_platform_name', 'cpu')` if no GPU
- JIT-compile the training step (already done via `@partial(jit, static_argnums=...)`)
- For Cholesky: JAX's `jnp.linalg.cholesky` requires strict PSD; use the parametric approach instead
- For large moment matrices: use `float64` precision (`jax.config.update("jax_enable_x64", True)`)

### 8. Moment matrix vs. moment vector

The moment MATRIX $\Omega_{ij} = \text{tr}[w_i^\dagger w_j]$ is an $N_\Omega \times N_\Omega$ matrix where $N_\Omega$ is the number of basis words up to length $L/2$. The moment VECTOR is the list of all distinct single-trace moments $\text{tr}[w]$ for $|w| \leq L$. These are related but different objects. The PSD constraint is on the MATRIX.

---

## Detailed Code Fixes Needed

The current code is a v1 scaffold. Here are the specific issues Claude Code should fix:

### Fix 1: `neural_master_field.py` — SD loss function

The `sd_loss_one_matrix` function has incorrect indexing. The correct SD equation is:

```python
# For V'(M) = Σ_k v_k M^k and test word M^n:
# LHS = Σ_k v_k * m_{n+k}     ← sum over V' coefficients
# RHS = Σ_{j=0}^{n-1} m_j * m_{n-j-1}   ← splitting/factorisation
```

The loop should be `for n in range(0, K - max_v_degree)` where `max_v_degree = len(v_prime_coeffs) - 1`.

### Fix 2: `neural_master_field.py` — Gaussian initialisation

The initial parameters should give the EXACT Gaussian solution:
```python
# m_{2k} = C_k (Catalan number) = binom(2k, k) / (k+1)
from math import comb
catalan = [comb(2*k, k) // (k+1) for k in range(K)]
```

For the quartic, initialise AT the Gaussian (g=0) solution and let the optimizer deform.

### Fix 3: `neural_master_field.py` — Multi-matrix Cholesky

The `MultiMatrixTrainer.moment_from_matrix` method currently does a Python loop to look up moments from the Cholesky matrix. This is:
1. Not JIT-compatible (Python control flow)
2. Incomplete (misses many decompositions)

**Fix**: Pre-compute a mapping `word → (i, j)` index pair at init time, then the lookup is a simple array index. Store this as a static array for JIT.

### Fix 4: `schwinger_dyson.py` — Two-matrix SD

The `TwoMatrixSD.sd_residuals` method generates test words and evaluates the SD equation, but the splitting sum only considers positions where the word letter matches the derivative variable. This is correct but the implementation doesn't account for the V' interaction terms properly.

**The correct structure for commutator-squared interaction**:

$\partial S / \partial (M_a)_{ij}$ acting on the exponential gives:
- From kinetic term $\text{Tr}[M_a^2/2]$: contribution $M_a$
- From interaction $-\frac{g^2}{4}\text{Tr}[M_a M_b - M_b M_a]^2$: 
  contribution $\frac{g^2}{2}(M_a M_b^2 + M_b^2 M_a - 2 M_b M_a M_b)$

The full $V'_a$ is then dotted into the test word, and the derivative on the test word produces the splitting.

### Fix 5: `bootstrap_sdp.py` — Linearisation

The current bootstrap has bilinear terms $m_j m_k$ in the SD equations, which makes it non-SDP. The proper approach:

1. Define the Hankel moment matrix $H_{ij} = m_{i+j}$
2. The SD equations become: $\sum_k v_k H_{0, n+k} = \sum_{j} H_{j, n-j-1}$ — these are LINEAR in $H$
3. Impose $H \succeq 0$ as an SDP constraint
4. The bilinear SD terms are automatically captured because $H_{ij} \geq m_i m_j$ entry-wise for PSD $H$

This makes the bootstrap a genuine SDP.

### Fix 6: Add `float64` throughout

Master field moments can span many orders of magnitude. Add at the top of every file:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Fix 7: Implement Voiculescu coefficient extraction

After training, extract the master field operator $\hat{M} = \hat{a} + \sum M_n (\hat{a}^\dagger)^n$ from the optimised moments:

```python
# 1. Compute free cumulants κ_n from moments (moment-cumulant formula)
# 2. M_n = κ_{n+1} (Voiculescu coefficients)  
# 3. Build operator in Cuntz-Fock space
# 4. Verify: <Ω|M̂^k|Ω> = m_k for all k
```

This gives the EXPLICIT master field, not just the moments.

---

## Suggested Tests (automated validation suite)

Create a file `test_master_field.py`:

```python
"""Automated test suite. Run: python test_master_field.py"""

def test_cuntz_algebra():
    """a_i a†_j = δ_ij in truncated Fock space."""
    from cuntz_fock import CuntzFockSpace
    for n in [1, 2, 3]:
        for L in [4, 6, 8]:
            fock = CuntzFockSpace(n, L)
            assert fock.verify_cuntz_relations(), f"Failed for n={n}, L={L}"
    print("✓ Cuntz algebra")

def test_gaussian_moments():
    """M̂ = â + â† reproduces Catalan numbers."""
    from cuntz_fock import CuntzFockSpace
    from math import comb
    fock = CuntzFockSpace(1, 10)
    M = fock.x(0)
    moments = fock.compute_moments(M, 10)
    for k in range(6):
        expected = comb(2*k, k) / (k + 1)
        assert abs(moments[2*k] - expected) < 1e-10, f"m_{2*k}: got {moments[2*k]}, expected {expected}"
    print("✓ Gaussian moments (Catalan)")

def test_gaussian_free_cumulants():
    """Gaussian has κ_2=1, all others zero."""
    from one_matrix import gaussian_moments, r_transform_from_moments
    m = gaussian_moments(10)
    kappa = r_transform_from_moments(m)
    assert abs(kappa[2] - 1.0) < 1e-10, f"κ_2 = {kappa[2]}"
    for k in [1, 3, 4, 5]:
        assert abs(kappa[k]) < 1e-8, f"κ_{k} = {kappa[k]} (should be 0)"
    print("✓ Gaussian free cumulants")

def test_quartic_sd_consistency():
    """Quartic moments satisfy SD equations."""
    from one_matrix import quartic_moments_from_sd
    from schwinger_dyson import OneMatrixSD
    for g in [0.1, 0.5, 1.0, 2.0]:
        m = quartic_moments_from_sd(g, 12)
        sd = OneMatrixSD([0, 1.0, 0, g], max_word_length=10)
        import numpy as np
        omega = np.zeros(sd.n_vars)
        for i, w in enumerate(sd.words):
            k = len(w)
            if k <= 12:
                omega[i] = m[k]
        res = sd.sd_residuals(omega)
        assert np.max(np.abs(res)) < 1e-4, f"g={g}: max residual = {np.max(np.abs(res))}"
    print("✓ Quartic SD consistency")

def test_sd_indexing():
    """Verify the SD equation LHS index is n+k, not n+k-1."""
    # For Gaussian V'(M) = M: tr[M · M^n] = m_{n+1}
    # SD: m_{n+1} = Σ_{j=0}^{n-1} m_j m_{n-j-1}
    # Check: m_1 = 0 (no terms), m_2 = m_0^2 = 1, m_3 = 0, m_4 = 2m_0 m_2 = 2
    from math import comb
    m = [comb(2*k, k) / (k+1) for k in range(8)]  # Catalan
    full_m = [0.0] * 15
    for k in range(8):
        full_m[2*k] = m[k]
    
    for n in range(7):
        lhs = full_m[n + 1]  # V'=M, so v_1=1, index = n+1
        rhs = sum(full_m[j] * full_m[n-j-1] for j in range(n))
        assert abs(lhs - rhs) < 1e-12, f"n={n}: LHS={lhs}, RHS={rhs}"
    print("✓ SD indexing verified")

def test_free_product():
    """tr[M1 M2 M1 M2] = 0 for free semicirculars with zero mean."""
    from cuntz_fock import CuntzFockSpace
    fock = CuntzFockSpace(2, 5)
    M1, M2 = fock.x(0), fock.x(1)
    val = fock.vev(M1 @ M2 @ M1 @ M2)
    # For free semicirculars: tr[ABAB] = tr[A]tr[B²A] + tr[B]tr[A²B] - tr[A]²tr[B]²
    # With tr[A]=tr[B]=0: result = 0
    assert abs(val) < 1e-12, f"tr[M1 M2 M1 M2] = {val}, expected 0"
    
    # But tr[M1² M2²] = tr[M1²]tr[M2²] = 1 (for free variables)
    val2 = fock.vev(M1 @ M1 @ M2 @ M2)
    assert abs(val2 - 1.0) < 1e-10, f"tr[M1² M2²] = {val2}, expected 1"
    print("✓ Free product relations")

def test_psd_constraint():
    """Gaussian moment matrix is PSD."""
    from schwinger_dyson import LoopMomentMatrix
    from one_matrix import gaussian_moments
    import numpy as np
    m = gaussian_moments(10)
    lmm = LoopMomentMatrix(1, 8)
    
    def moment_func(word):
        k = len(word)
        if k == 0: return 1.0
        if k <= 10: return m[k]
        return 0.0
    
    is_psd, min_eig = lmm.check_psd(moment_func)
    assert is_psd, f"Gaussian moment matrix not PSD: min_eig = {min_eig}"
    assert min_eig > -1e-10
    print(f"✓ PSD constraint (min eigenvalue = {min_eig:.6f})")

def test_voiculescu_roundtrip():
    """moments → free cumulants → Voiculescu coefficients → Fock VEVs → moments."""
    from cuntz_fock import CuntzFockSpace
    from one_matrix import gaussian_moments, r_transform_from_moments, voiculescu_coefficients
    import numpy as np
    
    m = gaussian_moments(8)
    kappa = r_transform_from_moments(m[:9])
    v_coeffs = voiculescu_coefficients(kappa)
    
    fock = CuntzFockSpace(1, 6)
    M_hat = fock.build_master_field_voiculescu(v_coeffs[:6])
    m_fock = fock.compute_moments(M_hat, 6)
    
    for k in range(0, 7, 2):
        assert abs(m_fock[k] - m[k]) < 1e-6, f"Roundtrip failed at m_{k}: {m_fock[k]} vs {m[k]}"
    print("✓ Voiculescu roundtrip")

if __name__ == "__main__":
    test_cuntz_algebra()
    test_gaussian_moments()
    test_gaussian_free_cumulants()
    test_sd_indexing()
    test_quartic_sd_consistency()
    test_free_product()
    test_psd_constraint()
    test_voiculescu_roundtrip()
    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
```

---

## Extending the Code

### Adding a new matrix model

1. Define $V'(M)$ coefficients in `config.py`
2. Add a case to `train.py` for the new model
3. If multi-matrix: define the interaction in `schwinger_dyson.py`
4. If exactly solvable: add exact solution to `one_matrix.py` for validation

### Scaling to larger truncation

The bottleneck is the moment matrix dimension $N_\Omega$. For $n$ matrices at truncation $L$:
$$N_\Omega \sim n^{L/2}$$

Strategies:
- **Symmetry reduction**: Use $Z_2$ symmetry ($M \to -M$) to halve the basis. Use exchange symmetry ($M_1 \leftrightarrow M_2$) to reduce further.
- **Sparse Cholesky**: Most entries of $L$ are near-zero for physical solutions. Use $L_1$ regularisation.
- **Neural parametrisation**: Replace the explicit Cholesky with a neural network $\theta \to L(\theta)$ that maps a low-dimensional latent code to the Cholesky factor. This gives implicit compression.
- **Stochastic SD**: Don't evaluate ALL SD equations; sample a random subset per epoch (SGD over the constraint space).

### Adding the Gross-Witten phase transition

The one-plaquette model $V(U) = -(1/2g^2)(U + U^\dagger)$ for unitary $U$ has:
- Weak coupling ($g^2 < 2$): single-cut eigenvalue density
- Strong coupling ($g^2 > 2$): two-cut (gapped) density
- 3rd-order phase transition at $g^2 = 2$

To implement: change from Hermitian to UNITARY matrices. The moments become $\text{tr}[U^n]$ and the SD equations change. The master field in the Cuntz-Fock space is now $\hat{U} = f(\hat{a}, \hat{a}^\dagger)$ with $\hat{U}\hat{U}^\dagger = 1$.

### Toward QCD₄

The ultimate target. On a lattice with $L_{\text{lat}}$ sites, the master field is $4 L_{\text{lat}}$ unitary matrices (link variables) satisfying the lattice Makeenko-Migdal equations. The SD equations are the lattice loop equations. The bootstrap approach of Kazakov-Zheng (2021) has already shown this is computationally feasible for small lattices.

---

## Output Specification

After a successful run, the `results/` directory should contain:

| File | Content |
|------|---------|
| `moments_{model}_g{g}.npy` | Array of optimised moments $m_0, m_1, \ldots, m_K$ |
| `free_cumulants_{model}_g{g}.npy` | Free cumulants $\kappa_1, \ldots, \kappa_K$ |
| `losses_{model}_g{g}.npy` | Training loss history |
| `moment_matrix_{model}_g{g}.npy` | (multi-matrix) The full moment matrix $\Omega$ |
| `convergence_{model}_g{g}.png` | Loss vs. epoch plot |
| `moments_{model}_g{g}.png` | ML moments vs. exact |
| `eigenvalue_density_{model}_g{g}.png` | Reconstructed $\rho(x)$ vs. exact |
| `sd_residuals_{model}_g{g}.txt` | Final SD equation residuals |

---

## References for Implementation Details

1. **SD equations**: Gopakumar-Gross §2.5 (eq. 2.30-2.35)
2. **Master field operator**: Gopakumar-Gross §2.3 (eq. 2.16-2.18)
3. **Hermitian representation**: Gopakumar-Gross §3.1 (eq. 3.46-3.47)
4. **Master field EOM**: Gopakumar-Gross §2.5 (eq. 2.33) and §4 (eq. 4.69-4.70)
5. **Cholesky / master variables**: Rodrigues et al. JHEP 2022, §2
6. **SDP bootstrap**: Anderson & Kruczenski, Nucl. Phys. B 921 (2017); Lin, JHEP 2020
7. **Free cumulants / R-transform**: Voiculescu, Dykema, Nica, "Free Random Variables" (1992)
