# Cuntz-Fock Coefficient Bootstrap — Reference (Phase 4)

## Problem

Construct the master field for large-N lattice Yang-Mills (or any large-N
matrix model) by parametrising the Gopakumar-Gross Cuntz-Fock operators
Û_μ directly and enforcing the N = ∞ loop equations + physical-state
constraints as loss terms. Invert GG's original setup: instead of computing
the Cuntz-Fock coefficients from connected Green's functions (which
presumes the theory is solved), make them the UNKNOWNS.

Central advantage: works at N = ∞ by construction. No Haar measure, no
center symmetry to break, no finite-N corrections. Phase 3's blockers
(R6 classical saddle, R9 multi-matrix correlation drift at W[2×2] to 900×)
do not arise in this formulation.

## Parametrisation

Take n = 2D Cuntz creation/annihilation operators â_j, â_j† on a truncated
Fock space of word length L_trunc. Label indices j ∈ {0, …, 2D−1} encode
signed lattice directions: `direction_to_label(+μ, D) = 2(μ−1)`,
`direction_to_label(−μ, D) = 2(μ−1) + 1`.

For each forward direction μ ∈ {1, …, D}, define a Hermitian polynomial

    Ĥ_μ = Σ_{|w|≤L_poly} h_{μ,w} · (â†_{w_1} … â†_{w_k}) + h.c.

with complex coefficients h_{μ,w}. The empty-word coefficient h_{μ,∅}
contributes Re(h_{μ,∅}) · I to Ĥ_μ (imaginary part cancels).

The master link operator is

    Û_μ = expm(i · Ĥ_μ)           (unitary, automatic)
    Û_{-μ} = Û_μ†                  (orientation reversal)

Wilson loops via vector sweep:

    W[C] = ⟨Ω| Û_{μ_1} Û_{μ_2} … Û_{μ_k} |Ω⟩
         = e_0ᵀ · (product of Û matrices) · e_0

## Parameter counts

Real DOFs per Hermitian generator = 2 d_L − 1 where d_L = |Fock basis|.

| D | n = 2D | L_poly | d_L | real DOFs / matrix | total real DOFs |
|---|--------|--------|-----|---------------------|-----------------|
| 1 (GW) | 1 | 6 | 7 | 13 | 13 |
| 2 (QCD₂) | 4 | 3 | 85 | 169 | 338 |
| 3 | 6 | 2 | 43 | 85 | 255 |
| 4 (QCD) | 8 | 2 | 73 | 145 | 580 |

Two orders of magnitude below Phase 3 (TEK at N=49 had 9604 parameters).

## Loss components

Minimise

    L = w_MM · L_MM + w_cyc · L_cyc + w_RP · L_RP + w_sym · L_sym  [+ w_sup · L_sup]

### L_MM — Makeenko-Migdal (direct)

Reuse Kazakov-Zheng candidate D from `master_field/lattice.LoopSystem`.
Per equation:

    res(C, e) = (1/λ) Σ_{P ∋ e} W[staple(C, e, P)]
              − c_self · W[C] − Σ_{splits} W[C_1] · W[C_2]

with c_self = 2. L_MM = Σ |res|². Indirect equations (Qiao-Zheng 2601.04316)
deferred.

### L_cyc — cyclicity / traciality

For each loop C and each cyclic rotation C_i:

    L_cyc = Σ_{C, i} |W[C_i] − W[C_0]|²

Enforces the planar trace property ⟨Ω|·|Ω⟩ = N = ∞ normalised trace.
Not automatic — must be imposed.

### L_RP — reflection positivity

Pick a reflection plane (default: μ = D). For open paths p in the
"positive half" (no step in −D direction), the K × K overlap matrix

    R_{ij} = ⟨Ω| Û_{θ(p_i)}† Û_{p_j} |Ω⟩         θ = reverse + time-axis flip

is PSD at the physical master field. L_RP = Σ_{λ_k(R) < 0} λ_k(R)².

### L_sym — lattice symmetries

For each σ ∈ B_D (axis permutations and sign flips):

    L_sym = Σ_{C, σ} |W[σ(C)] − W[C]|²

Uses B_D generators (D axis flips + D−1 adjacent swaps) for efficiency.

### L_sup — supervised anchor (optional)

    L_sup = Σ_C |W[C] − W_target[C]|²

Default target: GW lattice area law W[C] = w_+^{Area(C)} with
w_+ = `gw_w_plus(λ)` (D = 2 only). OFF by default; enable only if
unsupervised losses underdetermine.

## Phase priorities

1. **Phase A — Gross-Witten (D=1)**: single-mode Cuntz-Fock, L_poly = 6.
   Calibration: pure supervised moment-matching.
   Gate: max |w_k − w_k^GW| < 1e-2 at strong coupling (λ ≥ 1).
2. **Phase B — QCD₂ (D=2)**: L_poly = 3 (d_L = 85), 338 DOFs. Unsupervised
   L = L_MM + L_cyc + L_RP + L_sym.
   Gate at λ = 5: W[□], W[2×1], W[2×2], figure-8 factorisation, cyclicity
   residuals < 1e-6, symmetry residuals < 1e-6. All within 1%.
3. **Phase C — D=3**: L_poly = 2 (d_L = 43), 255 DOFs. Compare to
   Kazakov-Zheng bounds (arXiv:2203.11360) and published MC.
4. **Phase D — D=4**: L_poly = 2 (d_L = 73), 580 DOFs. The target —
   first explicit SU(∞) master-field construction for 4D lattice YM.

## Coupling continuation

Strong → weak with warm-start:

    h^{(0)} = init_hermitian_params(scale = 0.02)     (at λ = λ_start)
    for λ in schedule [λ_start → λ_end]:
        h^{(new)} = optimize_cuntz(L, h^{(old)}, λ, n_steps)
        h^{(old)} = h^{(new)}

Initialising at strictly h = 0 is a saddle point: at the identity, the
gradient of Re[W[C]] with respect to h vanishes by symmetry. Use
`scale = 0.02` non-zero random init to break this symmetry.

## Known caveats

- **MM is underdetermined in isolation.** Phase 1b (neural W[C] +
  MM-only, D = 2) converged to spurious solutions (W[plaq] tracked the
  candidate-D self-consistent value w_+^MM = λ − √(λ² − 1), which is
  the same as w_+^GW to leading order in 1/λ but wrong at subleading
  order). L_cyc, L_sym, L_RP are intended to break the degeneracy;
  supervised anchor is the fallback.
- **Fock truncation vs polynomial degree.** L_trunc must be ≥ L_poly
  so Ĥ fits in the basis. For Wilson loops, L_trunc should be ≥ |C|
  so the path stays in-basis; in practice L_trunc ≥ max(L_poly, 3)
  suffices at strong coupling, where amplitude decay is rapid.
- **Memory.** The word-operator cache `_build_word_operators` stores
  d_L matrices of size d_L × d_L. For d_L = 85, this is 10 MB; for
  d_L = 341 (L_trunc = 4, n = 4), 640 MB; for d_L = 1365 (L_trunc = 5),
  40 GB (infeasible). Refactor to sparse or on-demand for larger truncations.
- **eigh gradient is unstable.** `jax.scipy.linalg.expm` (Padé) is used
  instead of an eigh-based expm to avoid the nearly-degenerate-eigenvalue
  NaN in the eigh adjoint.

## Bibliography

- **Gopakumar, Gross** (hep-th/9411021). Original Cuntz-Fock master-field
  construction. Chapter 2 defines operators; §5 handles QCD₂.
- **Rodrigues et al.** (JHEP 2024). Collective-field / master-variable
  program for Hermitian matrix QM. Template for the SD-equation-as-loss
  approach.
- **Kazakov, Zheng** (arXiv:2203.11360). Bootstrap bounds via Makeenko-
  Migdal equations + SDP positivity for lattice YM in D = 2, 3, 4.
- **Qiao, Zheng** (arXiv:2601.04316). Systematic construction of MM loop
  equations (direct + indirect), extending Kazakov-Zheng.
- **Anderson, Kruczenski** (Nucl. Phys. B 921, 2017). SDP for lattice YM
  Wilson loops; introduces the Toeplitz-PSD constraint used in Kazakov-
  Zheng.
- **Han, Hartnoll** (Phys. Rev. X 10, 2020). Neural-network master-field
  construction for matrix models.
- **Local references**: `reference/qcd2_master_field.md` (Phase 0,
  single-α Gaussian ansatz); `reference/gross_witten_model.md` (exact
  GW moments); `reference/tek_master_field.md` (Phase 3, finite-N TEK).
- **Repo**: `master_field/lattice.py` (LoopSystem); `master_field/mm_equations.py`
  (candidate-D MM residuals); `master_field/cuntz_fock.py` (numpy Phase 0
  infrastructure); `cuntz_bootstrap/` (this project).
