# Bosonic matrix QM (Lin–Zheng 2507.21007): setup understood + build plan

**Date:** 2026-07-01 · **Branch:** `matrix-master-field` · **Status:** setup locked (E39 reproduced); build starting

**Why.** The user picked "dig into the appendices before we build." Target = validate our
master-field+continuation vehicle (proven on the KZ two-matrix *integral*,
`2026-07-01-two-matrix-validation.md`) in the **matrix-QM ground-state** (BFSS-class) setting,
then push to D=9 bosonic BFSS. This note records the verified setup and the concrete plan.

## The model (their Eq. 1; App-E normalization)

D traceless-Hermitian `X_I` (adjoint of SU(N)) + conjugate `P_I`, `[(X_I)_{ij},(P_J)_{kl}]=iδ_IJδ_{il}δ_{jk}`:

    H = ½ Σ_I ( Tr P_I² + M² Tr X_I² ) − (g²_YM/4) Σ_{I,J} Tr[X_I,X_J]²

App-E uses `Π_I = −i P_I` (⇒ all correlators **real**) and normalized single trace (factor N stripped):

    H = ½ Σ_I ( −Tr Π_I Π_I + M² Tr X_I X_I ) − ¼ Σ_{I,J} Tr[X_I,X_J]²       (E1)
    Gauss (singlet g.s. C|Ω⟩=0):  C = Σ_I ( [X_I,Π_I] − 1 ) = 0             (E2)

- `M²=0` = dimensional reduction of pure YM = **bosonic BFSS**; D=9,M²=0 = the prize target.
- Normalization: `𝓔 ≡ E₀/N² (g²_YM N)^{−1/3}`, `⟨tr X²⟩ ≡ (1/D)⟨tr X_I X_I⟩`.
- Published islands (Table II): D=2,M²=1: 𝓔∈[1.172098376,1.172098408] (**8 digits**, lvl14);
  D=2,M²=0: 𝓔∈[0.707832,0.707868] (lvl14); **D=9,M²=0: 𝓔∈[6.69946,6.69968]** (lvl11).

## What distinguishes this from the tracial KZ integral (the crux)

This is a genuine **quantum-mechanical ground state** — there are momenta `P_I` and a state `|Ω⟩`
that minimizes ⟨H⟩. The master field is a functional `τ(word in X_I,Π_I)` with:
- large-N factorization (single traces, cyclic) — as in KZ;
- **canonical structure**: `[X,P]=i` enters the moment tower via the Gauss law
  `⟨tr Σ_I[X_I,Π_I]⟩ = D` (E2), which time-reversal (`⟨tr XΠ⟩ = −⟨tr ΠX⟩`, sign `(−1)^{#Π}`)
  splits into `⟨tr X_I Π_I⟩ = +D/2`, `⟨tr Π_I X_I⟩ = −D/2` (E34). **This is the M5 "[X,P]≠0
  under the trace" lesson made concrete** — `tr[X,P]` is NOT zero here because the entries are
  operators.
- **EOM / stationarity**: `⟨[H,O]⟩ = 0` for every single-trace O (holds for any eigenstate).
  Linear up to level 5; first **quadratic** (factorized double-trace) at level 7 (E33):
  `0 = ⟨tr X_IX_JX_JΠ_KX_KX_K⟩ − ⟨tr X_IX_IΠ_JX_KX_KX_J⟩ + D⟨tr X_IX_I⟩²`.
- Two positivity families: **inner-product** `M_ij=⟨tr Ō_iO_j⟩⪰0` (E5) and **ground-state**
  `N_ij=⟨tr Ō_i[H,O_j]⟩⪰0` (E6, from ⟨O|(H−E₀)|O⟩≥0; "dramatically boosts precision").

## Verified: level-5 reduces to 3 variables, bracket E39 reproduced

Independent level-5 singlets: `v=⟨tr X_IX_I⟩`, `p=⟨tr P_IP_I⟩ (=−⟨tr Π_IΠ_I⟩)`, and a quartic-X.
Virial (E40): **`𝓔 = ¾ p + ¼ M² v`** (from `−2K+2M+4V=0`, `K+M+V=E₀`). Reduced positivity:

    E41:  [[v, D/2],[D/2, p]] ⪰ 0   and   p ≥ M² v
    E42:  [[D/2, p],[p, (D−1)v + (D/2)M²]] ⪰ 0

`lz_level5.py` reproduces **E39 exactly**, analytically and numerically:

    𝓔/D ≥ max{ 3/(16⟨trX²⟩) + M²⟨trX²⟩/4 ,  M²⟨trX²⟩ }         (lower; from E41)
    𝓔/D ≤ (1/8)[ 2M²⟨trX²⟩ + 3(2(D−1)⟨trX²⟩ + M²)^{1/2} ]        (upper; from E42)

Best level-5 lower bound (minimize branch-1 over ⟨trX²⟩ at `⟨trX²⟩=√3/(2M)`): **`𝓔 ≥ D·M·√3/4`**
(D=2,M²=1 ⇒ 0.86603; true 1.1721). All three published islands sit inside the level-5 band. ✓
**⇒ my encoding of Gauss/CCR + virial + both positivities is correct.**

## Leading master field = free/semicircle state (verified, `lz_gaussian_mf.py`)

The N=∞ ground state at leading order **is** the free (semicircular) state — a genuine
master field, computable analytically. `X_I` free semicircular with `τ(X_I X_J)=s δ_IJ`;
conjugate `P` saturates the oscillator uncertainty `τ(P_IP_J)=(1/4s)δ_IJ` (= saturates E41).
Energy in their units (`λ=g²_YM N=1`):

    𝓔(s) = D/(8s) + ½M²D s + (λ/2)D(D−1)s²        [kinetic + mass + commutator]

using the free-probability identity `Σ_{I,J} τ(tr[X_I,X_J]²) = −2D(D−1)s²` (I≠J:
`τ(X_I²X_J²)=s²`, `τ(X_IX_JX_IX_J)=0`) — **verified against random matrices** (D=2:
−0.640 exact vs −0.6398±0.0006; D=9: −23.04 vs −23.039). Minimizing over `s`:

| model | 𝓔_Gauss | published island | Δ𝓔 |
|---|---|---|---|
| D=2, M²=1 | 1.18226 (s=0.3774) | [1.172098376, 1.172098408] | **+0.87%** |
| D=2, M²=0 | 0.75000 (s=0.5) | [0.707832, 0.707868] | +5.95% |
| **D=9, M²=0 (bosonic BFSS)** | **6.75000 (s=0.25)** | [6.69946, 6.69968] | **+0.75%** |

A rigorous variational **upper** bound (a real state), **tight at large D / massive** (~0.8%,
exactly the BFSS regime), loose for D=2 massless (the flat-direction "peninsula"). `⟨trX²⟩=s`
lands just below the true island. **This is the leading point; it does not yet beat the
bootstrap island** — sharpening (below) is the value-add.

## Continuation anchor (verified)

g=0 ⇒ D decoupled matrix harmonic oscillators (freq √M²). Ground state saturates `p=M²v`
(equipartition) and `vp=D²/4` (coherent) ⇒ **`v=D/(2M)`, `p=DM/2`, `⟨trX²⟩=1/(2M)`**.
`𝓔` itself is singular at g=0 (the `(g²N)^{−1/3}` prefactor), so **continuation runs in the
dimensionless `λ_eff = g²N/M³` from 0 → target**, tracking the ground-state branch, exactly as
the KZ h-ramp — energy read off at the end via the virial E40.

## Engine: faithful port of their loop-equation generator (VALIDATED)

Found their Mathematica notebook (`github.com/Canonical111/O2massiveBootstrap`, the D=2 "O(2)
massive" model) and ported it to Python (`lz_port.py`). Complex letters `Z,z̄,P,q̄` for D=2.
The double-trace/factorization rule (their `cycZP`, the piece I couldn't derive by hand):

    tr[l, tail] − tr[tail, l] = (−1)^(#P in l) · Σ_i  tr[tail[:i]] · tr[tail[i+1:]]

over positions `i` where `tail[i]=conj(l)` (canonical conjugate) **and** the accumulated charge is
neutral there. Full constraint set = `cycZP` ∪ `gauge`(Gauss) ∪ `mirror`(T-rev) ∪ `reflect`(O(2))
∪ `commH`(EOM). **My moment counts match their Table I exactly** (single-trace vars: level 4→14,
6→94, 8→614; free vars after `tr(1)=1`: 3, 8 — level 8 off by 2 only from the D=2 ε-identities
E43). The EOM layer independently reproduces E31. The port is correct.

## Vehicle A result: exact-factorization + continuation does NOT pin the BFSS point (KEY FINDING)

Continue g:0→1 (g on the commutator; g=0 = Gaussian anchor), factorize every double-trace
`tr[a]tr[b]→⟨tr a⟩⟨tr b⟩`, min-change branch-tracking from the exact Gaussian. Result for D=2,M²=1
(true ⟨trX²⟩=0.389):

| level | ⟨trX²⟩ | \|residual\| |
|---|---|---|
| 5 | 0.570 (+46%) | 8e-17 |
| 6 | 0.491 (+26%) | 2e-16 |
| 7 | 0.539 (+39%) | 5e-16 |

Every endpoint is a **machine-exact** solution of the factorized loop equations, yet all are wrong
and **non-monotonic** — worse than the Gaussian MF (0.377, −3%). **Diagnosis: the factorized loop
equations are underdetermined for a matrix-QM ground state** (8 free vars at level 6 after all
linear constraints; exact factorization adds too few equations), so branch-tracking reaches
zero-residual but unphysical solutions. **This sharply distinguishes BFSS from the KZ tracial
integral** (where exact factorization + continuation *did* pin the point, ~0.3–1%,
`2026-07-01-two-matrix-validation.md`): for a **QM ground state**, positivity `M⪰0`, `N⪰0` is
essential to select the physical point — you cannot get it from factorization + branch-tracking
alone. (Consistent with the M5 no-go: `[X,P]≠iN·𝟙`, Gauss law, momentum sector.) The paper's
bootstrap relaxes factorization but keeps positivity → island; the missing piece for our *point*
is therefore positivity, not more factorization.

**⇒ The reliable master-field number for BFSS is the leading Gaussian (= D=∞ exact saddle):
0.87% (D=2,M²=1), 0.75% (D=9 BFSS).**

## Vehicle A + positivity (done): correct but truncation-limited

Added ground-state positivity — ported their `inner2` (M⪰0) and `innerground2` (N⪰0), with the
hermitian conjugation `ref2`: Z↔z̄, P→−p̄. **Both verified PSD at the exact free Gaussian** (computed
independently via planar Wick, `lz_gauss_moments.py`; that Gaussian also satisfies the g=0 loop
equations to 1e-15 — end-to-end validation of the engine). Then minimize `𝓔` s.t. loop eqs +
exact factorization (SQP) + `M,N⪰0`, continued from the Gaussian.

Result (D=2,M²=1): the positivity **lower bound** reproduces the analytic level-5 value
`𝓔 ≥ D·M·√3/4 = 0.866` (level 6: 0.892), and `⟨trX²⟩ ≥ 0.377`. With the Gaussian **upper** bound
1.182, the honest bracket is **`𝓔 ∈ [0.89, 1.18] ∋ 1.172`** — valid but wide. **The level-5/6
feasible set is still large** (min-𝓔 lands at ⟨trX²⟩≈0.76, not 0.389): pinning the point to their
8-digit island needs level 10–14, i.e. the **O(D) irrep block-diagonalization** (App F; level 14 =
192374 vars) — a large build I did not do.

## Honest thesis assessment for BFSS matrix QM

For a **QM ground state**, positivity is *essential and shared* between the master field and the
bootstrap, and exact factorization adds little at reachable levels — so the "master field gives a
point where the bootstrap gives a bracket" advantage is **weak here** (unlike the KZ tracial
integral / QCD loop equations, where factorization+continuation *does* pin the point cheaply). The
genuine master-field value-add for BFSS is therefore **large observables** (vehicle B: long single-
trace words the cutoff bootstrap can't reach), not the point. The cheap, reliable number remains
the **Gaussian = D=∞ saddle** (0.75% at D=9).

## Files (committed under `matrix_master_field/bfss/`)
- `lz_level5.py` — E39 reproduction + Gaussian anchor (understanding check).
- `lz_gaussian_mf.py` — leading Gaussian master field + RM cross-check.
- `lz_port.py` — faithful port of their engine (Table-I validated).
- `lz_gauss_moments.py` — exact free-Gaussian moments via planar Wick (true anchor; g=0 loop eqs to 1e-15).
- `lz_point2.py`, `lz_point6.py` — exact-factorization + continuation point solver.
- `lz_pos2.py` — positivity (M⪰0, N⪰0) augmented solver; PSD-verified at the true Gaussian.
- (external, not committed) `lz2507.pdf`, `O2MassCode.nb` — the paper and their Mathematica notebook.
