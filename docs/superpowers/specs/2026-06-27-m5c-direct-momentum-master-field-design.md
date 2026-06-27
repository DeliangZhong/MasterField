# M5c follow-up sub-project 2 — direct-momentum variational master field (design)

**Date:** 2026-06-27  **Status:** design (brainstormed + approved; spec under review)

**Goal.** Construct a genuine, reusable **N→∞ operator master field** for the unsolvable two-matrix
QM (HHK arXiv:2004.10212 Eq 17) carrying an **explicit conjugate momentum** `P̂` (`[X,P]=i`), so the
kinetic energy `⟨P²⟩` is computed **directly** as a moment of a genuine state and `⟨H⟩` is a
variational **upper** bound that cannot be exploited downward — replacing M5c's from-below
free-Fisher `Φ*`. Validated throughout against the sub-project-1 exact-diag ground truth.

**Why now (motivation).** M5c estimated the kinetic energy as `¼Φ* = ¼ sup_h{2τ⊗τ(∂h)−τ(h²)}`, a
**from-below** functional whose truncation the optimizer exploits — the "beats the Gaussian" 2.322
was a truncation artifact (`docs/superpowers/results/2026-06-25-m5c-two-matrix-qm.md`). The fix (the
M5c spec's deferred "Approach 2"): an explicit momentum operator so `⟨P²⟩` is a direct moment of a
genuine state, not a from-below sup. Sub-project 1 (exact diag, DONE) now supplies the independent
benchmark M5c lacked: `E/N² ≈ 2.34` at λ=1 (N→∞), `2.2578` at N=2
(`docs/superpowers/results/2026-06-26-m5c-exact-diag.md`).

---

## Conventions (pinned — declared/derived, not guessed)

- **Model:** `H = Tr(P_X²+P_Y²+m²(X²+Y²) − g²[X,Y]²)` (HHK Eq 17). `ℏ=1`, `m=1`, `X,Y` Hermitian
  `N×N` (U(N)), `[X_{ij},P_{X,kl}]=iδ_{il}δ_{jk}`. 't Hooft `λ=Ng²` fixed; report `E/N²`. The T1
  rescaling (`X=√N X̃`, …) and the `E/N²` energy functional are inherited verbatim from M5c.
- **Canonical pair / momentum rep:** the **bosonic** Fock space (proper CCR `[â,â†]=1`),
  `x̂=(â+â†)/√(2Ω)`, `p̂=−i√(Ω/2)(â−â†)` ⟹ `[x̂,p̂]=i`. This is sub-project 1's framework — NOT
  M5c's free Cuntz–Fock (`â_iâ†_j=δ_ij`), which has no canonical momentum.
- **Moments / large N:** single-trace `m[w]=lim_{N→∞}(1/N)⟨Tr w⟩`, `m[∅]=1`; words `w` in
  `{X,Y,P_X,P_Y}`; single-trace expectations factorize at large N (’t Hooft).
- **Symmetries preserved by the ansatz:** `X↔Y`, `X→−X`, `Y→−Y`, SU(N) singlet, cyclicity.
- **Reference state:** the large-N **Gaussian (Hartree) master field** `|G⟩` = the frequency-Ω,
  X↔Y-symmetric squeezed vacuum; its moments are the Wick values (M5c C2). `Â=0 ⟹ |ψ⟩=|G⟩` exactly,
  so the family **contains the Gaussian** (large-N upper bound `2.36452` at λ=1) and can only improve.

---

## Architecture

**1. The variational object.** `|ψ⟩ = U|G⟩`, `U = exp(iÂ)`, with `Â` a Hermitian **single-trace**
polynomial in `{X,Y,P_X,P_Y}` of degree `≤ d`, symmetry-respecting, 't Hooft-scaled. Variational
parameters `θ = (Ω, coefficients of Â)`.

**2. Genuine upper bound — structural (the M5c fix).** `U` unitary ⟹ `|ψ⟩` is a genuine normalized
state ⟹ `⟨ψ|H|ψ⟩ ≥ E_ground` for **every** `θ` (the variational principle — an exact inequality, no
functional sup, no basis truncation). Unitary conjugation preserves the CCR:
`U†[X,P]U = [U†XU, U†PU] = i` (a c-number, conjugation-invariant), so `X̂=U†XU`, `P̂=U†PU` remain a
canonical pair and `⟨P²⟩ = ⟨ψ|P²|ψ⟩` is a **direct moment**, not the inverted moment-Gram `bᵀG⁻¹b`.
This is the entire point: there is no from-below sup to exploit.

**3. The large-N moment-flow (the computation).** Heisenberg flow `O(s)=e^{-isÂ}O e^{isÂ}`,
`dO(s)/ds = i[O(s),Â]`, with `U†OU = O(1)`. The single-trace moments obey a closed-at-large-N
nonlinear ODE system
$$\frac{d}{ds}m_s[w] \;=\; i\,\big\langle G\big|\,e^{-isÂ}\,\tfrac1N[\mathrm{Tr}\,w,\,Â]\,e^{isÂ}\,\big|G\big\rangle,$$
where `[Tr w, Â]` (Â single-trace) is a single-/double-trace object that **factorizes** at large N
into products of `{m_s[·]}`. Integrate from `m_0` = Gaussian moments (`s=0`) to `s=1`; then
`E/N²(θ)` = the M5c energy functional (kinetic + mass + λ·commutator single-trace moments)
evaluated on `m_1`, with the kinetic `m_1[P_X²]+m_1[P_Y²]` now a **direct** moment. Minimize over `θ`.

**4. Truncation (the rigor seam — stated honestly).** `[Tr w, Â]` lengthens words, so the
moment-flow does **not** close on a finite word set; truncate at word-length `L` (RHS references to
length-`>L` moments are set by a stated closure — default: dropped/set to zero, with the V6
`L`-convergence study validating the choice). A truncated flow computes an **approximation** to
`⟨ψ|H|ψ⟩` and is **not guaranteed `≥ E_ground`** — the same *class* of risk that sank M5c. The
structural bound (§2) holds for the exact state; the truncated *evaluation* needs the guards below.

**5. Rigor (three layers).**
- **(a) SP1 artifact detector.** Every `E/N²(θ)` is checked against the exact ground truth
  (`≥ 2.34` at λ=1, N→∞; `≥ E_exact(N)` if specialized to finite N). This alone catches a
  2.322-style artifact in one line — the safety net sub-project 1 was built to provide.
- **(b) Convergence + higher-basis stability.** Report `E` vs `L`; re-optimize and re-check the
  optimum at `L+1, L+2` (the explicit M5c lesson: validate at a basis *larger* than the optimized
  one). Built-in self-diagnostic: `Tr[X,P_X]=iN²` is a **c-number identity** (the trace kills the
  operator parts), so `m_s[XP_X−P_XX]=i` is conserved by the exact flow; its drift under the
  truncated flow **measures** the truncation error.
- **(c) Exactly-closing sub-families → strict bounds.** Seek `Â` whose large-N moment-flow closes
  on a finite word set (beyond the trivially-closing Bogoliubov/Gaussian family). On those the
  energy is exact ⟹ a **strict** from-above bound, no truncation caveat.

**Net posture:** a trustworthy, SP1-validated N→∞ master field — strict on closing sub-families, and
a convergence-controlled, stability-checked, ground-truth-validated *estimate* elsewhere. A large,
honest step past M5c (which had none of these guards). A null result (master field ≈ Gaussian) is a
valid outcome — unlike M5c's false positive.

---

## Modules

- **`matrix_master_field/momentum_master_field.py` (new):** symmetry+cyclicity-reduced word basis;
  the factorized moment-flow RHS; the ODE integrator (`s: 0→1`); the `E/N²` functional on `m_1`; the
  variational solve over `θ`; the CCR-invariant + `L`-convergence diagnostics; a closing-family
  detector.
- **`matrix_master_field/tests/test_momentum_master_field.py`:** the validation obligations below.
- **Reuses:** `qm_master_field.py` (Gaussian reference moments + the large-N Gaussian number),
  `exact_diag.py` (the SP1 benchmark, imported for the V5 gate).

---

## Validation obligations (proof / computation for each)

| # | Claim | Check |
|---|-------|-------|
| V1 | `Â=0` recovers the large-N Gaussian | `E/N²(Ω, Â=0)` = M5c C2 Gaussian (λ=1 → `2.36452`), via the trivial `s`-flow (`m_1=m_0`) |
| V2 | `λ=0` anchor (exact) | `E/N² = 2m` to machine precision (two free oscillators; optimum at the Gaussian `Ω=m`) |
| V3 | single-matrix cross-check (method validity) | the SAME moment-flow applied to M5b (`H=Tr P²+Tr X²+(g/N)Tr X⁴`) recovers M5b's exact collective energy (`g=0→1`, `g=1→1.302`) — tests the momentum/flow machinery on a solved model |
| V4 | CCR preserved along the flow | `m_s[XP_X−P_XX] = i` is conserved by the exact flow (c-number identity `Tr[X,P_X]=iN²`); the truncated-flow drift `→ 0` as `L` grows (self-diagnostic) |
| V5 | genuine bound vs SP1 (artifact gate) | `E/N²(θ*) ≥` exact SP1 truth in the same regime (`≥ 2.34` at λ=1, N→∞); a value below it is a truncation artifact, fail-closed |
| V6 | `L`-convergence + higher-basis stability | `E/N²(θ*)` converges in `L`; the optimum re-checked at `L+1,L+2` is stable (no M5c-style higher-basis blow-up of the diagnostics) |
| V7 | closing family ⟹ strict bound (if found) | on an exactly-closing `Â`-family, verify the flow closes (no length-`>L` references) so `E/N²` is a strict from-above bound |
| V8 | the deliverable | the master field `θ*` + `E/N²(θ*)` at `λ∈{0,0.5,1}`, with the `L`-convergence series and the SP1 bracket, and whether it gets **below the Gaussian** (the ~0.025 prize) |

Verification medium: Python + `pytest` (numpy/scipy), as M1–M5 / SP1; cross-checked against
`exact_diag` (SP1) and the Gaussian/collective (M5b/c).

---

## Risks / open questions
- **R1 — expressiveness (main physics risk).** The `Â`-family may not reach below the Gaussian — the
  true gap is only `~0.025` at λ=1 (SP1). *Mitigation:* start with the lowest non-trivial symmetric
  `Â`, grow `d`, track `E` against the SP1 truth. **A null result (≈ Gaussian) is a legitimate,
  trustworthy outcome** — the deliverable is the *trustworthy method + master field object*, not a
  guaranteed numerical win.
- **R2 — truncation rigor.** The truncated moment-flow is not a strict bound (Architecture §4).
  *Mitigation:* the three-layer rigor machinery; results reported as SP1-validated estimates, strict
  only on closing families. No repeat of the M5c overclaim.
- **R3 — numerics.** The moment-flow ODE may be stiff; the factorized RHS large; the variational
  landscape non-convex. *Mitigation:* standard stiff ODE solvers + multi-start; the CCR-invariant
  and SP1 as guards at every step.
- **R4 — closing families may be trivial.** They might coincide with the Gaussian (no improvement);
  then the strict-bound layer yields only the Gaussian and the beyond-Gaussian result is the
  validated estimate. Acceptable and honestly reported.

## Out of scope
- Downstream BFSS/BMN application of the master field (future milestone).
- A tightened certified **lower** bound (separate hard problem; SP1 + the Gaussian already bracket).
- Finite-N exact diagonalization (that is sub-project 1).
