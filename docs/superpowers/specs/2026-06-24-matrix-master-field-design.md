# Design: `matrix_master_field/` — operator master field for matrix models

- **Date:** 2026-06-24
- **Status:** approved (brainstorming complete; proceeding to implementation plan)
- **Author:** Deliang Zhong (with Claude)
- **Scope of this spec:** Milestone 1 (matrix integrals) in full; Milestone 2 (matrix QM → BFSS/BMN) sketched as the deferred follow-on.

---

## 1. One-line thesis

Construct the large-N master field of "unsolvable" matrix models as **ML-optimized operators on a truncated Cuntz–Fock space**, by solving the **exact nonlinear loop equations** with **positivity automatic** (the operators define a genuine tracial state). This yields the *sharp, unique* field and *every* planar observable at once — eigenvalue densities, the **Brown measure** of non-normal combinations, mixed correlators — and a single **neural-amortized representation M̂(λ)** across the coupling/phase family.

## 2. The precise problem

The master field of a large-N matrix model is a tuple of operators (M̂₁,…,M̂ₙ) on a Hilbert space with a tracial vacuum state τ, such that **for every word**
> τ(M̂_{i₁}···M̂_{i_k}) = lim_{N→∞} (1/N)⟨tr M_{i₁}···M_{i_k}⟩.

By GNS, the master field is *equivalent in information* to the complete set of single-trace planar correlators (the ⋆-distribution) — but it is a far more powerful package: one object yields any observable by a single operator computation, including **non-polynomial** functionals (spectral densities, Brown measures) that are resummations of the moments, not individual moments.

The master field is pinned by the **intersection of two constraints**:
1. **Nonlinear loop equations** (Schwinger–Dyson / EOM): factorized, *quadratic* in the single-trace moments. Universal — the true moments satisfy them.
2. **Positivity**: the moments come from a genuine (tracial) state — equivalently the moment matrix is PSD.

**Either constraint alone is insufficient**, and this is the crux:
- Relax the nonlinearity to get a convex SDP (the "relaxation bootstrap") → the constraint weakens → **loose bounds**, not the sharp field.
- Drop positivity and solve the loop equations directly → they admit **unphysical / non-unique solutions** → spurious answers.

This is not speculation; it is documented (see §4).

## 3. Why operators on a Cuntz–Fock space

Representing the master field as Hermitian operators on a truncated Cuntz–Fock space makes the intersection in §2 native:
- **Positivity is automatic and exact.** Any vacuum expectation ⟨Ω|·|Ω⟩ is a positive state — no PSD cone to relax, no ex-post check.
- **Factorization is exact.** Single traces are vacuum expectations; multi-traces factorize by construction at N=∞.
- We then impose the **exact nonlinear loop equations** as the optimization target — not a relaxation.
- The object is an explicit operator, so **all observables** (spectra, Brown measure) follow, and the *same* representation carries to matrix QM and BFSS/BMN (Milestone 2).
- The naive *classical/commuting* master field provably fails to exist for the coupled two-matrix model (Haan 1980; McGuigan arXiv:2305.14664); the operator/free-probability object is the one that exists. The operator route is therefore necessary, not merely convenient.

**Caveat (not automatic):** the Cuntz vacuum is *not* tracial in general, so **cyclicity must be imposed** as a (linear) constraint. Positivity is the part that is free.

## 4. Honest novelty positioning (post-literature-verification)

Verified by full-text reads of the competing works. We state plainly what is and is not novel.

**What is NOT a valid novelty claim:**
- **"Exact nonlinear solve instead of SDP"** — *not* differentiating. All recent master-field-construction papers already solve the loop equations directly, not by SDP relaxation.
- **"First to reach strong-coupling multi-matrix"** — *false*. The de Mello Koch–Rodrigues group (arXiv:2108.08803, arXiv:2306.00935) already solves the strong-coupling two-matrix Yang–Mills model to high precision (e₀/N² = 0.8890(2)·λ^{1/3}, beating prior literature; loop equations to 10⁻¹⁰–10⁻¹⁸; finite-mass case included).

**The genuine, unoccupied openings** (full-text grep: zero occurrences across all surveyed papers):
1. **Cuntz–Fock operator representation** with a tracial vacuum. Maeta (arXiv:2601.16099 §5) *sketches but never builds* an operator representation (operators + density kernels), concedes the eigenvalue-density route "deadlocks" for multi-matrix, and worries the kernels are singular/oscillatory. We *build* it, in the Fock setting whose tracial state supplies the automatic positivity that defeats his failure mode.
2. **Neural amortization M̂(λ)** across the coupling/phase family — nobody does it.
3. **Brown measure / spectral observables** of the master field — nobody computes them; the collective-field papers do not even compute ρ(λ) (they output loop VEVs and a mass spectrum).
4. **Scalability in the number of matrices** toward BFSS/BMN, where the collective Hamiltonian becomes unwieldy and SDP bounds go loose.

**Positioning vs. each reference family:**

| vs. | They have | We add |
|---|---|---|
| SDP bootstrap (Lin 2002.08387; Kazakov–Zheng 2108.04830; Han–Hartnoll–Kruthoff 2004.10212) | rigorous bounds | the sharp field + observables |
| Maeta (2605.10720 positivity-light / 2601.16099 positivity-free) | direct nonlinear solve | **automatic positivity → no spurious roots** + the operator rep they only sketched |
| de Mello Koch / Rodrigues collective field (2108.08803, 2306.00935) | sharp, strong-coupling, both constraints respected | **operators (not loop-space) + amortization + Brown/ρ(λ) + scaling in #matrices** |

**The smoking gun (positivity necessity):** Maeta arXiv:2605.10720 §3.2.1 reports that of 15 seeds, several drive the loop-equation residual to F ~ 10⁻³⁰ (essentially exact) yet return the **wrong** moment — *satisfying the loop equations to machine zero does not pin the master field*. arXiv:2601.16099 drops positivity, states "we do not pursue the issue of non-uniqueness any further," and fails at g = 0.1 > g_c = 1/12. This is the experimental case for positivity-by-construction.

**Headline to lead with:** *operators + amortization + Brown/spectral observables + scalability*, with *automatic positivity defeats the spurious-solution failure mode* as the mechanism (our clean win over Maeta; the collective group also respects positivity, so it is the mechanism, not the standalone pitch).

## 5. Conventions (pinned; full `CONVENTIONS.md` to live in the package)

Per project rules, conventions are explicit and never guessed.

- **Matrices:** N×N **Hermitian**, M_i = M_i†.
- **Large-N scaling:** action written as **S = N · Tr[ … ]**; the coupling is the 't Hooft coupling, held fixed as N→∞. Single-trace moments **m_w ≡ lim_{N→∞} (1/N) ⟨tr M_{w₁}···M_{w_k}⟩**, with **m_∅ = 1** (hard constraint). Factorization: (1/N²)⟨tr A · tr B⟩ = m_A m_B + O(1/N²).
- **One-matrix potentials:** V(M) = ½M² (Gaussian) and ½M² + (g/4)M⁴ (quartic), so V′ = M (+ g M³). Matches existing validated code.
- **Two-matrix (commutator+mass), the primary target:** **S = N · tr[ ½(M₁² + M₂²) − (λ/4) [M₁,M₂]² ]**, λ > 0 (confining: tr[M₁,M₂]² ≤ 0). V′₁ = M₁ + (λ/2)(M₁M₂² + M₂²M₁ − 2 M₂M₁M₂), and 1↔2. Matches `schwinger_dyson.TwoMatrixSD`.
- **Kazakov–Zheng model (external validation, Milestone 4):** **ACTION TO BE TRANSCRIBED VERBATIM from arXiv:2108.04830 §2 before Milestone 4 — do not guess the quartic-potential and commutator coefficients.** Flagged as an explicit task.
- **Loop equations:** from ∫dM Σ_{ij} ∂/∂(M_a)_{ij}[ (word)_{ij} e^{−S} ] = 0 ⇒ ⟨tr(V′_a · w)⟩ = Σ_{splits} ⟨tr w_L⟩⟨tr w_R⟩ (factorized at N=∞). The existing `OneMatrixSD`/`TwoMatrixSD` implement this and are validated to machine precision on Gaussian/quartic.
- **Cuntz–Fock:** a_i a†_j = δ_ij; vacuum |Ω⟩; tracial state τ(·) = ⟨Ω|·|Ω⟩ with **cyclicity imposed**; M̂_i Hermitian.
- **Float64** everywhere (`jax.config.update("jax_enable_x64", True)`).

## 6. Method

**Operator ansatz.** On the truncated Cuntz–Fock space (n matrices, max word length L_trunc), each Hermitian master operator is a bounded-degree polynomial in creation/annihilation operators:
> M̂_i = Σ_{|u|+|v| ≤ d} c^{(i)}_{u,v} â†_{u₁}···â†_{u_p} â_{v₁}···â_{v_q},  with c^{(i)}_{u,v} = (c^{(i)}_{v,u})*.

- d = 1 (M̂_i = â_i + â†_i) is the free/Gaussian master field — exact baseline, already validated (`cuntz_fock.py`).
- For one matrix this reduces to the Voiculescu form M̂ = â + Σ_n M_n(â†)ⁿ → instant cross-check.
- Trainable parameters: the real coefficients {c^{(i)}_{u,v}}. Moments are vacuum expectations m_w = ⟨Ω|M̂_{w₁}···M̂_{w_k}|Ω⟩ via sparse matvecs.

**Loss / constraints** (positivity is *free*, never a loss term):
- Loop-equation residuals (the exact nonlinear equations; reuse `TwoMatrixSD`).
- Cyclicity/traciality (⟨Ω|AB|Ω⟩ = ⟨Ω|BA|Ω⟩).
- Symmetries (M₁↔M₂ exchange, Z₂).
- Normalization m_∅ = 1 (built in).

**Optimization.** Gradient descent (JAX/optax), with the non-convexity mitigations: free-field initialization, **λ-homotopy** (anneal from λ = 0 where it is the exact free field), over-parametrized operators, multi-restart. **No global-optimality guarantee** — this is the scientific bet; validation (§8) closes the loop.

**Amortization (Milestone 5).** A network λ ↦ {c^{(i)}_{u,v}(λ)} trained on the per-λ loop-equation residual; observables become differentiable in λ. Compared on **observables, not raw operator coefficients** (the master field is unique only up to unitary equivalence).

**Observables (`observables.py`).** Moments; eigenvalue density ρ(x) of any self-adjoint element (spectral theorem on the truncated operator); **Brown measure of X+iY** (Fuglede–Kadison / regularized — flagged delicate); λ-derivatives.

## 7. Package layout (`matrix_master_field/`)

```
matrix_master_field/
  cuntz_fock.py        # ported & cleaned n-matrix operator space
  one_matrix.py        # exact resolvents / free cumulants (ported) — validation truth
  operator_field.py    # NEW: Hermitian operator ansatz + moment evaluation
  loop_equations.py    # SD residuals (1- & 2-matrix), cyclicity, symmetries (ported/cleaned)
  bootstrap_sdp.py     # cvxpy SDP — 1-matrix (ported) + NEW 2-matrix island
  train.py             # JAX optimizer, λ-homotopy, Fock-order ladder
  amortized.py         # NEW (M5): network λ → M̂(λ)
  observables.py       # NEW: moments, ρ(x), Brown measure, λ-derivatives
  validate.py          # exact (1-matrix), SDP island, KZ moments, loop-eq residual checks
  CONVENTIONS.md  REFERENCES.bib
  tests/               # unit + regression (TDD)
```
QCD packages (`cuntz_bootstrap/`, `tek_master_field/`, QCD parts of `cluster/`) are untouched and postponed.

## 8. Milestones & measurable success

| # | Milestone | Success criterion |
|---|---|---|
| 1 | Scaffold + 1-matrix | Moments + ρ(x) to ≤ 1e-6 vs exact resolvent (Gaussian, quartic). |
| 2 | 2-matrix SDP island | Reproduce Kazakov–Zheng bounds (our own cvxpy SDP) to published precision. |
| 3 | 2-matrix operator field (commutator+mass) | Moments inside our SDP island, **converging** as L_trunc grows; ρ(x) of X; ⟨tr[M₁,M₂]²⟩ tracked vs λ. |
| 4 | KZ quartic model | Match KZ published ~6-digit moments (after transcribing their action). |
| 5 | Amortized M̂(λ) | One network reproducing per-λ solves across a λ-interval; observables differentiable in λ. |
| 6 | Brown measure (stretch) | Brown measure of X+iY. |
| — | *(Deferred)* Matrix QM → BFSS/BMN | Same operator core; stationarity ⟨[H,O]⟩=0 + energy in place of integral loop equations. |

## 9. Validation strategy

- **Exact** (1-matrix): resolvents, Catalan/quartic moments, free cumulants.
- **SDP island** (our own cvxpy 2-matrix bootstrap): the operator solution must land *inside* it.
- **Kazakov–Zheng** published moments (Milestone 4).
- **Loop-equation residual** and **convergence in L_trunc**: used as *consistency checks*, not as the solving method.
- **Cross-method** for the de Mello Koch/Rodrigues regime: reproduce their strong-coupling numbers as a target (not a beat-claim).

## 10. Testing

TDD where it bites: operator assembly + Hermiticity; Cuntz relations (`verify_cuntz_relations`); moment evaluation vs known free results (semicircle Catalan; tr[M₁M₂M₁M₂] = 1 for free semicirculars); regression tests pinning the 1-matrix exact values. Every physics claim that lands in a writeup gets a notebook with the required verification cells (dimensional/limit/cross-method).

## 11. Risks & honest caveats

- **Non-convex optimization** has no global-optimality certificate (local minima, sub-dominant saddles/phases). The mitigations (§6) + validation (§8) are the answer; *demonstrating reliable global convergence is the contribution.*
- **Truncation:** finite L_trunc limits accuracy; high moments / fine spectral features need larger L_trunc. Monitor convergence.
- **Multi-matrix determinacy** ("loop eqs + positivity ⇒ unique") is a *theorem for one matrix* (Hamburger) but *empirical/conjectural* for multi-matrix; we rely on it as the field does, backed by validation. Distinct saddles are physical phases (discrete choice by symmetry/action).
- **Brown measure** is determined by the ⋆-distribution but not continuous under moment convergence — its numerical extraction is the delicate, stretch deliverable.
- **Positioning honesty:** lead on operators/amortization/Brown; do not headline "avoid SDP" or "first to strong coupling."

## 12. References (verified arXiv IDs; full `REFERENCES.bib` to be built and book/older-paper IDs re-verified)

- Master field / free probability: Gopakumar–Gross **hep-th/9411021**; Douglas **hep-th/9411025**; Douglas–Li **hep-th/9412203**; Halpern–Schwartz **hep-th/9809197**; Voiculescu free entropy (Nica–Speicher, *Lectures on the Combinatorics of Free Probability*, CUP 2006; Mingo–Speicher, *Free Probability and Random Matrices*, Springer 2017 — cite by DOI/ISBN, no arXiv). Non-existence of naive master field: Haan, Z. Phys. C6 (1980) 345 *(verify pages)*; McGuigan **2305.14664**.
- Bootstrap (bounds): Lin **2002.08387**; Kazakov–Zheng **2108.04830**; Han–Hartnoll–Kruthoff **2004.10212**; Anderson–Kruczenski **1612.08140**.
- Master-field construction (competitors): de Mello Koch–Jevicki–Liu–Mathaba–Rodrigues **2108.08803**; Mathaba–Mulokwe–Rodrigues **2306.00935**; Maeta **2605.10720**, **2601.16099**.
- NN wavefunctions (different object — contrast): Han–Hartnoll **1906.08781**; Bodendorfer et al. **2409.00398**.
- BFSS/BMN bootstrap (Milestone 2 context): Lin **2302.04416**; Lin–Zheng **2410.14647**, **2507.21007**.
- Brown measure: L. G. Brown (1986); Haagerup–Larsen *(verify exact references when used)*.
