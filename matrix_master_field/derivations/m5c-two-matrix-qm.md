# M5c — Two-Matrix Quantum Mechanics: Derivations (T1, T4)

**Date:** 2026-06-25
**Branch:** `matrix-master-field`
**Verification:** `matrix_master_field/tests/test_m5c_derivations.py` (3 tests, all GREEN)

> Scope: this file discharges T1 (model, scaling, g=0 anchor) and the T4 1-D building
> block (free Fisher information of a density). The general two-matrix `Φ*=bᵀG⁻¹b` (T4 full),
> the stationarity/Gauss relations (T5/T2), and the nonzero-λ normalization check land in
> Tasks 5, 3, and 2 respectively. Citations are limited to results we can verify; analytic
> computations shown here stand on their own and are cited to no one.

---

## T1 — Pinned model, large-N scaling, and the g=0 anchor

### Model (HHK Eq 17)

The two-matrix quantum mechanics Hamiltonian is

```
H = Tr( P_X² + P_Y² + m²(X² + Y²) − g²[X,Y]² )
```

with `X, Y` Hermitian `N×N`, conjugate momenta `[X_ab, P_X,cd] = i δ_ad δ_bc` (and likewise
`Y`), `ℏ=1`. The commutator coupling is **negative**: `[X,Y]` is anti-Hermitian
(`[X,Y]† = [Y,X] = −[X,Y]`), so `Tr[X,Y]² ≤ 0` and `−g²Tr[X,Y]² ≥ 0` — the term is confining.

**Reference:** Han, Hartnoll, Kruthoff, *Bootstrapping Matrix Quantum Mechanics*,
arXiv:2004.10212, Eq. (17). ("HHK" throughout = Han–Hartnoll–Kruthoff.)

### Large-N 't Hooft scaling (X = √N X̃, g² = λ/N)

Define `X̃ = X/√N`, `Ỹ = Y/√N`, `P̃ = P/√N`, and the normalized trace
`m[w] ≡ (1/N)⟨Tr w⟩ = O(1)` in the large-N ground state. Term by term:

```
(1/N²) Tr X²      = (1/N²)·N·Tr X̃²        → m[X̃²]               (since Tr X² = N Tr X̃²)
(1/N²) Tr P_X²    = m[P̃_X²]               (same scaling for P)
```

Commutator term, with `g² = λ/N` (λ = Ng² fixed):

```
[X,Y] = [√N X̃, √N Ỹ] = N [X̃, Ỹ]
Tr[X,Y]² = N² Tr[X̃,Ỹ]²
g² Tr[X,Y]² = (λ/N)·N² Tr[X̃,Ỹ]² = λN Tr[X̃,Ỹ]²
(1/N²) g² Tr[X,Y]² = (λ/N) Tr[X̃,Ỹ]² = λ · (1/N)Tr[X̃,Ỹ]² = λ m[[X̃,Ỹ]²]
```

Hence the energy density (dividing `H` by `N²`):

```
E/N² = m[P̃_X²] + m[P̃_Y²] + m²( m[X̃²] + m[Ỹ²] ) − λ m[[X̃,Ỹ]²]
```

The interaction sign is `−λ m[[X̃,Ỹ]²]`; since `m[[X̃,Ỹ]²] ≤ 0`, this contribution is `≥ 0`
(confining), consistent with `−g²[X,Y]²` above. This is the energy functional used by all
three pieces (SDP, Gaussian, free-Fisher).

### Coupling dial

HHK control the model by the dimensionless ratio `m²/g^{4/3}` (arXiv:2004.10212). The g=0
anchor below sits at `λ = 0`, i.e. `m²/g^{4/3} → ∞` (free / decoupled limit).

### g=0 anchor: E/N² = 2m

At `λ = 0`, `X̃` and `Ỹ` decouple: `H = H_1[X] ⊕ H_1[Y]` with `H_1 = Tr(P² + m²X²)`.

**Single oscillator mode.** A Hermitian `N×N` matrix has `N²` independent real components; in an
orthonormal basis `H_1 = Σ_{a=1}^{N²} (p_a² + m² x_a²)`, a sum of decoupled modes
`h = p² + m²x²` with `[x,p]=i`. Writing `h = 2(½p² + ½m²x²)`, the rescaled oscillator
`½p² + ½ω²x²` with frequency `ω = m` has spectrum `(n+½)m` and ground-state variance
`⟨x²⟩ = 1/(2ω) = 1/(2m)`. Therefore `h` has spectrum `(2n+1)m`, ground energy `m`, and
`⟨x²⟩ = 1/(2m)`.

**Aggregate (large N).** All `N²` modes in their ground state give `E_1 = N² m`, so
`E_1/N² = m`. For the position moment, `Tr X² = Σ_a x_a²` gives `⟨Tr X²⟩ = N²·(1/(2m))`, and
with `Tr X̃² = Tr X²/N`,

```
m[X̃²] = (1/N)⟨Tr X̃²⟩ = (1/N)·(N²/(2m))/N = 1/(2m).
```

The virial relation for `h = p² + m²x²` is `⟨p²⟩ = m²⟨x²⟩`, so `m[P̃²] = m²·1/(2m) = m/2`.

**Two matrices:**

| Quantity | Value | Derivation |
|----------|-------|------------|
| `m[X̃²] = m[Ỹ²]` | `1/(2m)` | ground variance `1/(2m)` of `h=p²+m²x²`, summed over `N²` modes |
| `m[P̃_X²] = m[P̃_Y²]` | `m/2` | virial `⟨p²⟩ = m²⟨x²⟩` |
| `E_1/N²` (per matrix) | `m` | `m[P̃²] + m² m[X̃²] = m/2 + m/2 = m` |
| `E/N²` (two matrices) | **`2m`** | two decoupled matrix oscillators |

This anchor value is convention-independent: it is the physical ground energy `2N²m` of Eq 17
at `g=0`, divided by `N²`. **Numeric check:** `test_g0_anchor_is_2m` verifies `E/N²=2m` and
`m[X̃²]=1/(2m)` to `1e-12` for `m ∈ {0.5, 1.0, 2.0}`.

---

## T4 (partial) — Free Fisher information of a 1-D density

### Formula

For a probability density `σ` on ℝ, Voiculescu's free Fisher information evaluates to

```
Φ*[σ] = (4π²/3) ∫ σ(x)³ dx,
```

implemented in `free_fisher.phi_star_density`. We cite the free-Fisher-information / free-entropy
framework to Voiculescu, *The analogues of entropy and of Fisher's information measure in free
probability theory, V*, Invent. Math. **132** (1998) 189–227, and verify the `(4π²/3)`
normalization analytically below (the standard semicircle must give `Φ*=1`).

### Analytic semicircle check (Φ* = 1)

Standard semicircle on `[−2,2]`, variance 1: `σ(x) = √(4−x²)/(2π)`. Then

```
∫_{−2}^{2} σ³ dx = ∫_{−2}^{2} (4−x²)^{3/2}/(8π³) dx        [x = 2 sinθ]
                = (1/(8π³)) ∫_{−π/2}^{π/2} (4cos²θ)^{3/2}·2cosθ dθ
                = (1/(8π³)) ∫_{−π/2}^{π/2} 16 cos⁴θ dθ
                = (16/(8π³))·(3π/8) = 3/(4π²),
```

using `∫_{−π/2}^{π/2} cos⁴θ dθ = 3π/8`. Hence `Φ*[σ] = (4π²/3)·3/(4π²) = 1`.
**Numeric check:** `test_phi_star_semicircle_is_one` (200001-point grid, `<1e-4`).

### ¼Φ* = the M5b kinetic term

The M5b g=0 collective density is `σ(y) = √(2−y²)/π` on `[−√2,√2]` (semicircle of variance ½).

```
∫ σ³ dy = ∫_{−√2}^{√2} (2−y²)^{3/2}/π³ dy   [y = √2 sinθ]
        = (1/π³)·4·(3π/8) = 3/(2π²),
```

so `Φ*[σ] = (4π²/3)·3/(2π²) = 2` and `¼Φ*[σ] = ½`. The M5b collective kinetic functional gives
the same number,

```
∫ (π²/3) σ³ dy = (π²/3)·3/(2π²) = ½,
```

and the two agree identically because `(4π²/3)·¼ = π²/3` — i.e. `¼Φ*[σ] = ∫(π²/3)σ³` term by
term. At the anchor (`m=1`), `½ = m[P̃²]`, confirming the kinetic identity `m[P̃²] = ¼Φ*` at the
free point. **Numeric check:** `test_quarter_phi_star_reduces_to_m5b_kinetic`
(`quarter_phi ≈ m5b_kinetic < 1e-4` and `≈ 0.5 < 1e-3`).

---

## What lands in later tasks

- **Task 2:** *(done — see C2 section below)* the nonzero-λ finite-N Wick → large-N check pinning
  the interaction coefficient `+λ/(2Ω²)` and its N-powers.
- **Task 5 (T4 full):** *(done — see T4 full section below)* the general two-matrix
  `Φ*(X̃,Ỹ) = bᵀG⁻¹b` from the conjugate-variable relation `τ(ξ_X w)=τ⊗τ(∂_X w)` (which
  requires `τ` tracial), and the g=0 free-additivity check `Φ*(X̃,Ỹ)=Φ*(X̃)+Φ*(Ỹ)=4 ⇒ ¼Φ*=1`.
- **Task 3 (T5/T2):** the stationarity loop equations `⟨[H,Tr w]⟩=0` and the SU(N) Gauss law
  `m[(0,2)+O]−m[(2,0)+O]=i·m[O]`, each verified against the exact g=0 Gaussian moments.
  (Done — see T5 and T2 sections below.)

---

## C2 — Gaussian master field: variational upper bound

**Implementation:** `matrix_master_field/qm_master_field.py`
**Tests:** `matrix_master_field/tests/test_qm_master_field.py` (5 tests, all GREEN)

### Trial state and two-point functions

The Gaussian trial state `|ψ_G(Ω)⟩` is the ground state of the free Hamiltonian
`Tr(P_X² + Ω²X²) + Tr(P_Y² + Ω²Y²)` with frequency `Ω > 0`. In this state `X` and `Y`
are independent, each a Gaussian Hermitian matrix with two-point function

```
⟨X_ij X_kl⟩ = a δ_il δ_jk,    a = 1/(2Ω).
```

The momentum two-point function follows from the virial relation: `⟨(P_X)_ij (P_X)_kl⟩ = Ω² a δ_il δ_jk`.

### Wick contraction of ⟨Tr[X,Y]²⟩

Expanding the commutator:

```
Tr[X,Y]² = Tr(XYXY) − Tr(XYYX) − Tr(YXXY) + Tr(YXYX)
          = 2 Tr(XYXY) − 2 Tr(XY²X)       (using cyclicity of trace).
```

Wick contractions (X,Y independent, so only same-letter pairs survive):

```
⟨Tr XYXY⟩ = ΣΣ ⟨X_ij X_kl⟩ ⟨Y_jk Y_li⟩ = Σ a δ_il δ_jk · a δ_ji δ_lk = a² Σ δ_ii δ_jj... 
```

Working through the index sums explicitly:

```
⟨Tr XYXY⟩ = Σ_{i,j,k,l} ⟨X_ij X_kl⟩ ⟨Y_jk Y_li⟩
           = Σ a δ_il δ_jk · a δ_ji δ_lk = a² Σ_i δ_ii = a² N.

⟨Tr XYYX⟩ = Σ_{i,j,k,l} ⟨X_ij X_lk⟩ ⟨Y_jk Y_kl⟩... = a² N³.
```

The second contraction picks up a factor N² from the two free index loops. Hence:

```
⟨Tr[X,Y]²⟩ = 2 a²N − 2 a²N³ = 2a²N(1 − N²).
```

For large N, the leading term is `−2a²N³ = −N³/(2Ω²)`.

### Variational energy E/N²(Ω)

With `X = √N X̃`, `Y = √N Ỹ` (so `a = 1/(2Ω)` applies to the unscaled matrices), and
`g² = λ/N`:

```
(1/N²)⟨H⟩ = m[P̃_X²] + m[P̃_Y²] + m²(m[X̃²] + m[Ỹ²]) − (λ/N)·⟨Tr[X,Y]²⟩/N²
```

Using `m[P̃²] = Ω/2`, `m[X̃²] = 1/(2Ω)` (from the Gaussian ground state at frequency Ω), and
the large-N limit of the commutator moment `⟨Tr[X,Y]²⟩/N³ → −1/(2Ω²)`:

```
E/N²(Ω) = Ω + m²/Ω + λ/(2Ω²).
```

The sign of the interaction contribution is positive because `−g²Tr[X,Y]² ≥ 0` (confining):
`−(λ/N)·(−N³/(2Ω²))/N² = +λ/(2Ω²)`.

### Minimization over Ω: the cubic equation

The stationarity condition `∂(E/N²)/∂Ω = 0` gives:

```
1 − m²/Ω² − λ/Ω³ = 0   ⟹   Ω³ − m²Ω − λ = 0.
```

At `λ = 0`: `Ω³ = m²Ω ⟹ Ω = m` (the unique positive root), giving `E/N² = m + m + 0 = 2m`,
which matches the g=0 anchor from T1. For `λ > 0`, the root is found numerically via `brentq`.

### Sampling cross-check (test_wick_commutator_moment_matches_sampling)

The Wick formula `⟨Tr[X,Y]²⟩ = 2a²N(1−N²)` is cross-checked against direct Monte Carlo sampling
of independent Gaussian Hermitian matrices at N=12, Ω=1, 4000 trials (fixed seed 0). The
sampled mean agrees with the analytic formula to within 6% relative tolerance.

### Normalization cross-check (test_lambda_normalization_nonzero)

For N=20, 80, 320 with Ω=1.3, λ=0.7, the finite-N energy shift `−(λ/N)·⟨Tr[X,Y]²⟩/N²`
converges monotonically to the large-N target `λ/(2Ω²)`, reaching relative error O(1/N²).
This pins both the overall coefficient and the N-power counting in the interaction term.

---

## T5 — Stationarity loop equations ⟨[H, Tr w]⟩ = 0

**Implementation:** `matrix_master_field/tm_qm_relations.py` — `stationarity_terms(word)`
**Verification:** `test_stationarity_residual_zero_on_g0_moments` — residual `<1e-10` on all words up to length 3 (verified residual <1e-10 on the g=0 Gaussian moments).

### Heisenberg equations of motion

From `H = Tr(P_X² + P_Y² + m²X² + m²Y² − λ[X̃,Ỹ]²)` (with scaled variables), the commutators with each letter follow from canonical commutation `[X̃_ab, P̃_X,cd] = i δ_ad δ_bc / N` and the large-N planar limit. For a single letter in a word, the Leibniz rule gives the replacement:

**Position letters:**

```
[H, X̃] = −2i P̃_X
[H, Ỹ] = −2i P̃_Y
```

**Momentum letters:**

```
[H, P̃_X] = i(2m² X̃ − 2λ F_X),    F_X = [Ỹ,[X̃,Ỹ]] = 2ỸX̃Ỹ − Ỹ²X̃ − X̃Ỹ²
[H, P̃_Y] = i(2m² Ỹ − 2λ F_Y),    F_Y = [X̃,[Ỹ,X̃]] = 2X̃ỸX̃ − X̃²Ỹ − ỸX̃²
```

The sign of the EOM for position letters follows from `[P²,X] = P[P,X]+[P,X]P = −2iP`; for momentum from `[m²X²,P] = m²X[X,P]+m²[X,P]X = 2im²X` and the interaction from the large-N commutator expansion of `[[X̃,Ỹ]²,P̃_X]`.

### Loop equations

For a word `w = (c_0, c_1, ..., c_{n-1})`, the loop equation is

```
⟨[H, Tr w]⟩ = Σ_k ⟨Tr(c_0···[H,c_k]···c_{n-1})⟩ = 0,
```

where each letter `c_k` is replaced by its `[H,·]` commutator from the EOM above. At `λ=0` the force terms drop, and the relation becomes a recursion among Gaussian moments.

### Verification at g=0

The g=0 ground state is Gaussian with exact moments from the Wick contraction of `_wick(w, m)` (non-crossing planar pairings). The stationarity residual vanishes identically at `λ=0` because the Gaussian distribution is the exact ground state of `H_{λ=0}`. The code in `stationarity_terms` encodes this replacement; `test_stationarity_residual_zero_on_g0_moments` checks all 341 words up to length 3 and finds residual `<1e-10` (verified residual <1e-10 on the g=0 Gaussian moments).

---

## T2 — SU(N) Gauss law

**Implementation:** `matrix_master_field/tm_qm_relations.py` — `gauss_terms(O)`
**Verification:** `test_gauss_law_residual_zero_on_g0_moments` — residual `<1e-10` on all operator insertions O up to length 2 (verified residual <1e-10 on the g=0 Gaussian moments).

### Gauss law relation

The SU(N) Gauss law constraint on physical states gives, for each canonical pair and any operator insertion `O`:

```
m[(X̃, P̃_X) + O] − m[(P̃_X, X̃) + O] = i · m[O]
```

In the word encoding (0=X̃, 2=P̃_X):

```
m[(0, 2) + O] − m[(2, 0) + O] − i · m[O] = 0
```

The sign convention (position-first minus momentum-first) is anchored by the two-point functions:

```
m[(0,2)] = m[X̃P̃_X] = +i/2
m[(2,0)] = m[P̃_X X̃] = −i/2
```

so `m[(0,2)] − m[(2,0)] = i = i · m[()] = i · 1`, which holds. The analogous relation for the Y pair (letters 1=Ỹ, 3=P̃_Y) follows by symmetry.

### Verification at g=0

The Gauss law is a kinematic identity (it holds for any state satisfying canonical commutation, not just the ground state). At g=0 the moments are Gaussian Wick values. `gauss_terms(O)` returns the three-term combination `[(1.0, (0,2)+O), (−1.0, (2,0)+O), (−i, O)]`. `test_gauss_law_residual_zero_on_g0_moments` checks all 85 insertions O up to length 2 and finds residual `<1e-10` (verified residual <1e-10 on the g=0 Gaussian moments).

---

## T4 (full) — Free Fisher information of the two-matrix operator field

**Implementation:** `matrix_master_field/qm_master_field.py` — `free_fisher_information`, `_phi_star`, `_free_diff_quotient_score`, `fisher_master_field`, `_tm_position_words`
**Tests:** `matrix_master_field/tests/test_qm_master_field.py` — `test_free_fisher_reduces_to_one_matrix_semicircle` (GREEN), `test_fisher_master_field_anchor_lambda0` (GREEN with `MMF_SLOW=1`)

### Conjugate-variable formula for Φ*

For a tracial noncommutative probability space `(A, τ)` with a self-adjoint element `X`, the free
score `ξ_X` is defined by the duality relation

```
τ(ξ_X · w) = τ⊗τ(∂_X w),    ∂_X w = Σ_{i: w_i=X} w_{<i} ⊗ w_{>i}
```

for all noncommutative polynomials `w`. The free Fisher information is `Φ*(X) = τ(ξ_X²)`.

**Why traciality is required.** The identity `τ(ξ_X · w) = τ⊗τ(∂_X w)` is a tensor identity that
holds only when `τ` is a trace (i.e., tracial: `τ(ab)=τ(ba)` for all `a,b`). The Cuntz vacuum
state `ω(·) = ⟨Ω|·|Ω⟩` on the Cuntz–Fock space is NOT tracial: `ω(a_i a†_j) ≠ ω(a†_j a_i)` in
general. Therefore, the training loss for `fisher_master_field` includes a cyclicity penalty
`w_sym · (cyclicity + exchange + Z₂)` to push the Cuntz vacuum toward a tracial state; the
energy is evaluated only after this penalty has been enforced.

### Gram matrix and finite-basis formula

Fix a finite basis of words `{w_c}`. The Gram matrix is

```
G[c, c'] = τ(w_c* w_{c'}) = τ(reverse(w_c) · w_{c'}) = moment(reverse(w_c) + w_{c'}).
```

The free difference-quotient score vector has components

```
b[c] = τ⊗τ(∂_X w_c) = Σ_{i: (w_c)_i = X} moment(w_c[:i]) · moment(w_c[i+1:]).
```

Solving `G ξ = b` gives the Gram representation of the score, and

```
Φ*(X) = bᵀ G⁻¹ b.
```

For two matrices `X̃, Ỹ`, free additivity gives `Φ*(X̃, Ỹ) = Φ*(X̃) + Φ*(Ỹ)`, computed as the
sum of two `bᵀG⁻¹b` terms, one for each generator.

### Sup-characterization and truncation lower-bound

The free Fisher information admits the sup-characterization

```
Φ*(X) = sup_b { 2 τ⊗τ(∂_X b) − τ(b²) : b ∈ A tracial },
```

from which it follows that restricting the sup to a finite-dimensional subspace spanned by
`{w_c}` yields a lower bound: truncating `basis_words` LOWERS Φ*. Therefore `¼Φ*` (from
the finite basis) is a one-sided, from-below estimate of the kinetic energy. Larger bases give
tighter lower bounds.

### n=1 semicircle check: Φ*=2 ⇒ ¼Φ*=½

For the semicircle law of variance `½` (moments `m_{2k} = (½)^k C_k` where `C_k = C(2k,k)/(k+1)`
is the Catalan number), the free Fisher information formula gives `Φ*=2`. This is consistent with
the density-level computation from T4 (partial): the variance-½ semicircle has density
`σ(y) = √(2−y²)/π` on `[−√2,√2]`, and `Φ*[σ] = (4π²/3) ∫ σ³ = 2`, so `¼Φ*=½`.

The test `test_free_fisher_reduces_to_one_matrix_semicircle` constructs the semicircle moments
directly and verifies `abs(0.25 * phi - 0.5) < 1e-6` using basis `[(), (0,), (0,0), (0,0,0)]`.

### g=0 free-additivity check: ¼Φ*(X̃,Ỹ)=1

At `λ=0`, the two matrices `X̃, Ỹ` are free (independent semicircles of variance `½`).
Free-additivity of Φ* gives

```
Φ*(X̃, Ỹ) = Φ*(X̃) + Φ*(Ỹ) = 2 + 2 = 4   ⟹   ¼Φ* = 1.
```

The full energy at `λ=0` is then

```
E/N² = ¼Φ* + m² (m[X̃²] + m[Ỹ²]) = 1 + 1²·(½ + ½) = 2 = 2m,
```

matching the T1 anchor. The test `test_fisher_master_field_anchor_lambda0` verifies this
numerically with 1500 Adam steps on the Cuntz–Fock operator field (`cutoff=8, degree=2,
max_word_len=3`), asserting `|energy − 2.0| < 5e-2` and `|m2 − 0.5| < 5e-2`.
