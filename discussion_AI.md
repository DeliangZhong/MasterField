# AI Development Discussion Log

<!-- INSTRUCTIONS FOR AI:
  - Reverse chronological order: newest entries on top.
  - Only read the TOP entry — everything below has already been discussed.
  - Naming convention:
      "Discussion-" = AI/human open-ended discussion and brainstorming
      "Implementation-" = AI's implementation results and findings
      "Feedback-"       = Human/AI feedback on the preceding implementation
  - When adding a new entry, prepend it above the previous top entry.
-->

## Discussion-10: Gross-Witten Model — Unitary Loop Equations (Apr 9, 2026)

### The problem

The Gross-Witten-Wadia (GWW) model is the simplest lattice gauge theory — a single-plaquette unitary matrix model:

Z = ∫ dU exp(N/(2t) Tr(U + U†))

with 3rd-order phase transition at t_c = 1. Exact solution known (Gross-Witten 1980, Wadia 1980). Stepping stone to QCD₂ → QCD₃ → QCD₄.

### Why U = exp(iM) does NOT simplify to Hermitian

Initial idea: write U = exp(iM), M Hermitian, giving V(M) = -(1/t)cos(M). Then V'(M) = (1/t)sin(M) and we reuse the Hermitian MasterFieldTrainer.

**This fails.** The path integral measures differ:
- Hermitian: Π(θ_j - θ_k)² (Vandermonde polynomial)
- Unitary (Haar): Π|2 sin((θ_j - θ_k)/2)|² (Vandermonde on circle)

The different Vandermonde gives **different saddle-point equations** and **different SD/loop equations**. Empirically verified: Hermitian SD equations give large residuals (O(1)) when evaluated on exact GWW moments. Tested 4 candidate unitary SD equations — all fail for the weak-coupling (gapped) phase at n ≥ 2.

### Exact solution (verified numerically)

**Strong coupling (t > 1, ungapped):**
ρ(θ) = (1/2π)(1 + (1/t)cos θ) on [-π, π]
w₁ = 1/(2t), wₙ = 0 for n ≥ 2.

**Weak coupling (t < 1, gapped):**
ρ(θ) = (1/(πt)) cos(θ/2) √(t - sin²(θ/2)) on [-θ_c, θ_c], sin²(θ_c/2) = t.
w₁ = 1 - t/2, w₂ = (1-t)².

### Saddle-point equation on the circle

The correct constraint is:
P.V. ∫ ρ(φ) cot((θ-φ)/2) dφ = (1/t) sin θ

Via Hilbert transform on the circle (ungapped phase):
2 Σ_{n≥1} wₙ sin(nθ) = (1/t) sin θ → w₁ = 1/(2t), wₙ = 0 for n ≥ 2.

For the gapped phase, the Hilbert transform doesn't directly give a simple Fourier relation because the density has compact support.

### Open question: the correct moment recursion

The standard Hermitian SD equation Σ_k v_k m_{n+k} = splitting does **not** apply. The unitary loop equation involves:
1. Resolvent on the unit circle (outer R⁺ for |z| > 1, inner R⁻ for |z| < 1)
2. The spectral curve relating R⁺ and R⁻
3. Moment recursion derived from the spectral curve

### Implementation paths

**Path A — Saddle-point as loss:** Parameterize ρ(θ) directly (Fourier coefficients or support endpoint + polynomial). Loss = saddle-point equation residual. Constraint: ∫ρ = 1, ρ ≥ 0. The PV integral is numerically tricky.

**Path B — Toeplitz moment matrix + resolvent:** Parameterize Wilson loops wₙ. Enforce Toeplitz PSD (T_{ij} = w_{|i-j|} ≽ 0) and the resolvent equation. Requires deriving the correct moment recursion.

**Path C — Direct density optimization:** Parameterize the support endpoint a = sin(θ_c/2) and use the known density form ρ ∝ cos(θ/2)√(a² - sin²(θ/2)). The only free parameter is a (determined by the coupling). This reduces to 1D optimization.

### Toward QCD₂

Once GWW is solved (1 unitary matrix), extending to QCD₂ on a small lattice = multiple coupled unitaries with plaquette interactions. The same Toeplitz moment matrix framework applies, just with more matrices and lattice Makeenko-Migdal equations as the loop constraints.

### Reference

See `reference/gross_witten_model.md` for detailed formulas and bibliography.

---

## Implementation-9: Two-Matrix Scaling Study with Symmetry (Apr 6, 2026)

### Symmetry constraints added

Three constraint types pre-computed at init and enforced in the loss:
- **Cyclicity**: tr[w] = tr[cyclic rotation of w] (5 pairs at L=6)
- **Exchange M₁↔M₂**: tr[w(M₁,M₂)] = tr[w(M₂,M₁)] (7 pairs at L=6)
- **Z₂ (M→-M)**: tr[odd-length words] = 0 (10 terms at L=6)

Verification at L=6: |tr[M₁²]-tr[M₂²]| = 0, |tr[M₁M₂]-tr[M₂M₁]| = 0, |tr[M₁]| = 3e-21.

### Scaling study results (g=1.0, commutator-squared, with symmetry)

| L | dim | loss | min eig | tr[M₁²] | tr[M₁M₂] | time |
|---|-----|------|---------|---------|----------|------|
| 4 | 7 | 6.4e-253 | 0.961 | -0.003 | -0.004 | 17s |
| 6 | 15 | 2.2e-34 | 0.572 | 0.179 | 0.335 | 21s |
| 8 | 31 | 6.5e-38 | 0.426 | 0.348 | 0.000 | 40s |

**Not yet converged**: tr[M₁²] changes 0.18→0.35 from L=6 to L=8. Scaling study extended to L=4..16 to find convergence. L=14 (dim=255) est. ~1-2 hr, L=16 (dim=511) est. ~6-12 hr.

### Cluster setup

Standalone `cluster/` package with PBS scripts. Uses EasyBuild modules (`jax/0.4.25-gfbf-2023a`, `SciPy-bundle/2023.07-gfbf-2023a`). Auto-installs `optax` to user site-packages.

---

## Implementation-8: One-Matrix Quartic via scipy SLSQP (Apr 6, 2026)

### Problem

Direct moment parametrization fails for quartic: SD equations are underdetermined (always 2 more unknowns than equations). Without PSD, optimizer finds spurious solutions (SD loss=0 but wrong moments). Continuation in g and relative residuals help stability but don't fix uniqueness.

### Solution

scipy.optimize.minimize with SLSQP + Hankel PSD constraint:
- Objective: SD residuals (relative)
- Constraint: min eigenvalue of even Hankel submatrix G_{ij}=m_{2(i+j)} ≥ 0
- Key fix: Hankel dim must be n_even//2+1 (not n_even) to avoid truncation artifacts making the matrix falsely non-PSD

### Results

All couplings g=0.1..5.0 converge (SD loss < 1e-15). Low moments within 3-5% of exact. Quartic g=0.5 at L=12: m₂=0.614 (exact 0.631), m₄=0.773 (exact 0.738). Remaining error from underdetermined high moments.

---

## Implementation-7: Code Fixes & Training Pipeline (Apr 6, 2026)

### Fixes applied (CLAUDE_CODE_INSTRUCTIONS.md)

All 7 listed fixes implemented. Additionally discovered and fixed:
- **schwinger_dyson.py SD indexing**: used `n+k-1` (wrong), fixed to `n+k`
- **Eigenvalue density formula**: code doubled g-dependent terms. Correct: P(x) = gx² + 1 + ga²/2
- **Free cumulant κ_6**: hardcoded formula was classical, not free. Replaced with general recursion via z·G = 1 + Σ κ_k G^k — works to arbitrary order

### Training results

**Gaussian (Stage 1a)**: Perfect. Loss = 0, all moments match Catalan numbers to machine precision.
**Quartic (Stage 1b)**: Solved via scipy SLSQP (see Implementation-8).
**Two-matrix (Stage 3)**: JIT-compiled trainer converges. Symmetry constraints added (see Implementation-9).

---

## Discussion-5: Scaling Strategies & Future Directions (Apr 6, 2026)

### Scaling to larger truncation

The bottleneck is moment matrix dimension N_Ω ~ n^(L/2) for n matrices at truncation L.

**Strategies:**
- **Symmetry reduction**: Z₂ symmetry (M → -M) halves basis. Exchange symmetry (M₁ ↔ M₂) reduces further.
- **Sparse Cholesky**: L₁ regularisation on near-zero entries.
- **Neural parametrisation**: Neural network θ → L(θ) maps low-dim latent code to Cholesky factor.
- **Stochastic SD**: Sample random SD equation subset per epoch (SGD over constraint space).

### Adding new matrix models

1. Define V'(M) coefficients in `config.py`
2. Add case to `train.py`
3. If multi-matrix: define interaction in `schwinger_dyson.py`
4. If exactly solvable: add exact solution to `one_matrix.py` for validation

### Gross-Witten phase transition

One-plaquette model V(U) = -(1/2g²)(U + U†) for unitary U:
- Weak coupling (g² < 2): single-cut eigenvalue density
- Strong coupling (g² > 2): two-cut (gapped) density
- 3rd-order phase transition at g² = 2

Requires changing from Hermitian to UNITARY matrices. Moments become tr[Uⁿ], SD equations change, master field in Cuntz-Fock space is Û = f(â, â†) with ÛÛ† = 1.

### Toward QCD₄

Ultimate target. On lattice with L_lat sites: 4·L_lat unitary matrices (link variables) satisfying lattice Makeenko-Migdal equations. Bootstrap approach of Kazakov-Zheng (2021) shown computationally feasible for small lattices.

---

## Discussion-4: Known Bugs, Pitfalls & Code Fixes Needed (Apr 6, 2026)

### Known pitfalls

1. **SD equation indexing**: LHS is Σ_k v_k m_{n+k}, NOT m_{n+k-1}. From tr[M^k · M^n] = m_{k+n}.
2. **Cyclic equivalence**: Canonical representative = lexicographically smallest rotation. Without this, redundant variables cause ill-conditioning.
3. **PSD enforcement**: Cholesky diagonal must use L_ii = exp(ℓ_i). Forgetting exponential → diagonal can go negative → Ω not PSD.
4. **Normalisation**: m₀ = tr[I] = 1 is hard constraint. Either fix by construction or add large penalty (λ ~ 100).
5. **Symmetry**: For V symmetric under M → -M, enforce odd-moment vanishing by construction. Otherwise spurious flat directions.
6. **Large-L divergence**: Moments m_{2k} grow factorially (~(2k)!/(k!(k+1)!) for Gaussian). Loss must use RELATIVE residuals.
7. **JAX**: Use `jax_platform_name='cpu'` if no GPU. JIT-compile training step. Use float64 (`jax_enable_x64=True`). Use parametric Cholesky, not jnp.linalg.cholesky.
8. **Moment matrix vs. vector**: Moment MATRIX Ω_{ij} = tr[w_i† w_j] is N_Ω × N_Ω. PSD constraint applies to MATRIX, not the moment vector.

### Code fixes needed

**Fix 1: `neural_master_field.py` — SD loss function**
`sd_loss_one_matrix` has incorrect indexing. Correct: LHS = Σ_k v_k * m_{n+k}, RHS = Σ_{j=0}^{n-1} m_j * m_{n-j-1}. Loop should be `for n in range(0, K - max_v_degree)`.

**Fix 2: `neural_master_field.py` — Gaussian initialisation**
Initial parameters should give exact Gaussian solution: m_{2k} = C_k (Catalan) = binom(2k,k)/(k+1). For quartic, initialise at Gaussian (g=0) and let optimizer deform.

**Fix 3: `neural_master_field.py` — Multi-matrix Cholesky**
`MultiMatrixTrainer.moment_from_matrix` uses Python loop (not JIT-compatible). Fix: pre-compute word → (i,j) index mapping at init, store as static array for JIT.

**Fix 4: `schwinger_dyson.py` — Two-matrix SD**
`TwoMatrixSD.sd_residuals` splitting sum is correct but doesn't properly handle V' interaction terms for commutator-squared. Full V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a - 2 M_b M_a M_b).

**Fix 5: `bootstrap_sdp.py` — Linearisation**
Current bootstrap has bilinear terms (not SDP). Proper approach: define Hankel moment matrix H_{ij} = m_{i+j}, SD equations become linear in H, impose H ≽ 0 as SDP constraint.

**Fix 6: Add float64 throughout**
Add `jax.config.update("jax_enable_x64", True)` at top of every JAX file.

**Fix 7: Implement Voiculescu coefficient extraction**
After training: moments → free cumulants κ_n (moment-cumulant formula) → Voiculescu coefficients M_n = κ_{n+1} → build operator in Cuntz-Fock space → verify ⟨Ω|M̂^k|Ω⟩ = m_k.

---

## Discussion-3: Computational Pipeline — Stages 0–4 (Apr 6, 2026)

### Stage 0: Sanity checks (run FIRST, must ALL pass)

```bash
python master_field/cuntz_fock.py
```
Expected: `✓ Cuntz algebra verified (n=1, L=6, dim=7)`, Gaussian moments = Catalan numbers (m₀=1, m₂=1, m₄=2, m₆=5, m₈=14, m₁₀=42), all to < 10⁻¹².

```bash
python master_field/one_matrix.py
```
Expected: Gaussian free cumulants κ₁=0, κ₂=1, κ_{k≥3}=0. Quartic (g=0.5) moments m₂≈0.5162, m₄≈0.4838.

```bash
python master_field/schwinger_dyson.py
```
Expected: Gaussian SD residuals max < 10⁻¹². Quartic SD residuals max < 10⁻⁶.

### Stage 1: One-matrix models (exactly solvable — MUST match)

**1a. Gaussian**: `python train.py --model gaussian --validate --max_word_length 12 --n_epochs 3000 --lr 1e-2`
Success: all even moments match Catalan numbers to < 10⁻⁶. Loss < 10⁻¹⁰. κ₂ = 1.0000, all others ≈ 0.

**1b. Quartic**: Run at g = 0.5, 1.0, 5.0 with --validate. Moments match exact values (eigenvalue density integration) to < 10⁻⁴.

**1c. Coupling scan**: g ∈ {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0}. R-transform coefficients κ_n(g) should be smooth in g.

### Stage 2: Bootstrap validation

`python train.py --model quartic --coupling 0.5 --bootstrap --max_word_length 10`
SDP bootstrap produces rigorous bounds. ML solution must lie within them.

### Stage 3: Two coupled matrices (the real challenge)

`python train.py --model two_matrix_coupled --coupling 1.0 --max_word_length 6 --n_epochs 20000 --lr 5e-4 --interaction commutator_squared`
Word space grows as 2^L. No closed-form solution. At L=6: basis dim ~100, Cholesky has ~5000 parameters.
Success: Loss < 10⁻⁴, min eig(Ω) > -10⁻⁸, stable under increasing L.
Benchmark: Rodrigues et al. JHEP 2024.

### Stage 4: Scaling study

L = 4, 6, 8, 10 at fixed coupling. First few moments (m₂, m₄) should stabilise quickly; higher moments need larger L.

### Output specification

Results directory: `results/`
- `moments_{model}_g{g}.npy` — optimised moments
- `free_cumulants_{model}_g{g}.npy` — free cumulants
- `losses_{model}_g{g}.npy` — training loss history
- `moment_matrix_{model}_g{g}.npy` — full moment matrix (multi-matrix)
- `convergence_{model}_g{g}.png` — loss vs epoch
- `moments_{model}_g{g}.png` — ML vs exact moments
- `eigenvalue_density_{model}_g{g}.png` — reconstructed ρ(x) vs exact
- `sd_residuals_{model}_g{g}.txt` — final SD residuals

---

## Discussion-2: Physics Framework & Schwinger-Dyson Equations (Apr 6, 2026)

### The master field problem

For matrix model with action S[M₁,...,Mₙ] invariant under M_i → U M_i U†, large-N factorisation gives:
⟨O₁ O₂⟩ = ⟨O₁⟩⟨O₂⟩ + O(1/N²)

The path integral concentrates on a single configuration — the master field M̄_i:
⟨O⟩ = tr[M̄_{i₁} M̄_{i₂} ⋯ M̄_{iₖ}]

### What we optimise

Loop moments Ω(C) = tr[M̄_{i₁} ⋯ M̄_{iₖ}] for word C = (i₁,...,iₖ). These satisfy:
1. **Schwinger-Dyson equations** — exact recursion from the action
2. **Positive semidefiniteness** — moment matrix Ω_{ij} = tr[w_i† w_j] ≽ 0
3. **Cyclicity** — Ω(C) = Ω(cyclic permutations of C)
4. **Hermiticity** — Ω(C)* = Ω(C⁻¹)
5. **Normalisation** — Ω(∅) = 1

### SD equations

Single matrix with potential V(M): for test word M^n:
LHS = Σ_k v_k m_{n+k}  (where V'(M) = Σ_k v_k M^k)
RHS = Σ_{j=0}^{n-1} m_j m_{n-j-1}

**CRITICAL**: LHS index is n+k, NOT n+k-1. From tr[V'(M)·M^n] = Σ_k v_k tr[M^{k+n}].

Two matrices with S = Tr[M₁²/2 + M₂²/2 - (g²/4)[M₁,M₂]²]:
V'_a = M_a + (g²/2)(M_a M_b² + M_b² M_a - 2 M_b M_a M_b)

SD for derivative w.r.t. M_a on test word w = M_{i₁}⋯M_{iₖ}:
⟨tr[V'_a · w]⟩ = Σ (over positions m where i_m = a) ⟨tr[w_left]⟩⟨tr[w_right]⟩

### Loss function

L = L_SD + λ_EOM · L_EOM + λ_reg · L_reg
- L_SD = Σ_n |SD residual_n|²
- L_EOM = Σ_ψ |⟨Ω|[V'(M̂) - 2Π̂]|ψ⟩|²
- L_reg: spectral regularisation for eigenvalue non-negativity

### PSD constraint via Cholesky

Ω = LL† with L lower-triangular. Neural network outputs L; Ω automatically PSD.

### References

1. SD equations: Gopakumar-Gross §2.5 (eq. 2.30-2.35)
2. Master field operator: Gopakumar-Gross §2.3 (eq. 2.16-2.18)
3. Hermitian representation: Gopakumar-Gross §3.1 (eq. 3.46-3.47)
4. Master field EOM: Gopakumar-Gross §2.5 (eq. 2.33) and §4 (eq. 4.69-4.70)
5. Cholesky/master variables: Rodrigues et al. JHEP 2022, §2
6. SDP bootstrap: Anderson & Kruczenski, Nucl. Phys. B 921 (2017); Lin, JHEP 2020
7. Free cumulants/R-transform: Voiculescu, Dykema, Nica, "Free Random Variables" (1992)

---

## Implementation-1: Project Setup, Structure & Dependencies (Apr 6, 2026)

### Mission

Numerically construct the master field for large-N matrix models using ML optimisation. The master field is the N=∞ saddle point of the matrix path integral; all gauge-invariant observables are computable from it without functional integration. An explicit construction in QCD₄ would be one of the most important results in theoretical physics.

### Dependencies

```bash
pip install jax jaxlib numpy scipy matplotlib optax flax cvxpy torch --break-system-packages
```

JAX for autodiff + JIT (compiles inner loop to XLA), cvxpy for SDP validation (SCS/MOSEK solvers), PyTorch available as alternative.

### Project structure

```
master_field/
├── config.py                  # Hyperparameters, model registry
├── cuntz_fock.py              # Cuntz algebra, Boltzmann Fock space, operator reps
├── one_matrix.py              # Exact one-matrix solutions (resolvent, density, moments)
├── schwinger_dyson.py         # Loop equations for multi-matrix models
├── neural_master_field.py     # Neural ansätze + JAX training loops
├── bootstrap_sdp.py           # SDP bounds via cvxpy
├── train.py                   # CLI entry point
├── visualize.py               # Plots
├── test_master_field.py       # Automated test suite
└── results/                   # Output directory
```

### Test suite

Run: `python master_field/test_master_field.py`
Tests: Cuntz algebra relations, Gaussian moments (Catalan), Gaussian free cumulants, SD indexing verification, quartic SD consistency, free product relations, PSD constraint, Voiculescu roundtrip.
