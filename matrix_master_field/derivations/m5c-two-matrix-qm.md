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

- **Task 2:** the nonzero-λ finite-N Wick → large-N check pinning the interaction coefficient
  `−λ/(2Ω²)` and its N-powers (the anchor alone, with λ=0, cannot test the interaction term).
- **Task 5 (T4 full):** the general two-matrix `Φ*(X̃,Ỹ) = bᵀG⁻¹b` from the conjugate-variable
  relation `τ(ξ_X w)=τ⊗τ(∂_X w)` (which requires `τ` tracial), and the g=0 free-additivity check
  `Φ*(X̃,Ỹ)=Φ*(X̃)+Φ*(Ỹ)=4 ⇒ ¼Φ*=1`.
- **Task 3 (T5/T2):** the stationarity loop equations `⟨[H,Tr w]⟩=0` and the SU(N) Gauss law
  `m[(0,2)+O]−m[(2,0)+O]=i·m[O]`, each verified against the exact g=0 Gaussian moments.
