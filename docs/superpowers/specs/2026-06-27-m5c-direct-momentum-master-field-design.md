# M5c follow-up sub-project 2 — direct-momentum variational master field (design)

**Date:** 2026-06-27 · **v2: 2026-06-28** (pivoted to finite-N staging per the go/no-go gate
`d68b4bf`; supersedes the pure-large-N v1). **Status:** design (re-worked after Codex audit + the
closure/rigor derivation; spec under review).

**Goal.** A genuine, reusable operator master field for the unsolvable two-matrix QM (HHK Eq 17) with
an **explicit conjugate momentum** `P̃` (`[X̃,P̃]∝i`), giving **rigorous Rayleigh–Ritz upper bounds at
finite N** (exact by construction — no from-below estimate) plus the **N→∞ master-field object** by
extrapolation, with the (now proven-to-close) large-N **moment-flow** as a fast surrogate/initializer.
Validated throughout against the sub-project-1 exact-diag ground truth and bracketed below by the
existing bootstrap SDP.

**What changed (v1 → v2), and why.** v1 proposed computing the energy from a *pure-large-N truncated
moment-flow*. The go/no-go derivation (`matrix_master_field/derivations/sp2-moment-flow-closure.md`,
commit `d68b4bf`) established two things by derivation + machine-precision numerics:
- **Closure (S1) holds:** the single-trace moment-flow closes at large N (infinite, `L`-truncated,
  nonlinear hierarchy, via Makeenko–Migdal/Jevicki–Sakita factorization; generator normalization
  `Â=N²τ(â)` derived and verified N-independent to `1e-14`; the `1/N²` factorization measured,
  `C_N·N²=0.500` flat for N=2,3,4). The large-N flow is therefore a sound, cheap **surrogate**.
- **Truncation rigor (S2) fails:** the `L`-truncated large-N flow does **not** compute `⟨ψ|H|ψ⟩` for
  any state; bootstrap positivity is intrinsically **one-sided** (certifies only the *lower* bound we
  already have) and cannot upper-bound `E₀` from the trial; minimizing a non-bound over `Â` drives the
  result **unsafely downward** (the M5c selection mechanism). So pure-large-N is **not** a bound.

The fix is to take the bound from a **literal finite-N state**: `e^{iÂ}|G⟩` at fixed N is a genuine
unit vector, so `⟨ψ|H|ψ⟩` is an exact variational **upper** bound by Rayleigh–Ritz.

**Honest value proposition (read before reviewing).** At N=2,3 a variational bound is *looser* than
SP1's exact diagonalization — so B does **not** beat SP1's number there. B's deliverable is (i) a
**reusable, parametrized operator master field** (the `e^{iÂ}|G⟩` family + optimal `Â`) — which SP1
(bare numbers) does not provide and which is the stated goal for downstream BFSS/BMN; (ii) **rigorous
upper bounds at each N**; (iii) the **N→∞ master-field object** via the surrogate flow, honestly
bracketed by `[SDP lower, Gaussian upper]` and anchored to SP1. A tighter *number* than SP1/Gaussian
is **not** promised; obtaining genuinely new numbers requires reaching N beyond exact-diag's ceiling
(R1 below) — the key open lever.

---

## Conventions (pinned — corrected per the gate; do not re-guess)

- **Model:** `H = Tr(P_X²+P_Y²+m²(X²+Y²) − g²[X,Y]²)` (HHK Eq 17), `ℏ=1`, `m=1`,
  `[X_{ij},P_{X,kl}]=iδ_{il}δ_{jk}` — the **matrix** commutator `[X,P]` is an OPERATOR, not `iN·𝟙`;
  only `Tr[X,P]=iN²`. 't Hooft `λ=Ng²`; report `E/N²`.
- **Scaled variables (use these):** `X̃=X/√N`, `P̃=P/√N` ⟹ `[X̃_{ab},P̃_{cd}]=(i/N)δ_{ad}δ_{bc}`,
  `τ=(1/N)Tr`, so `τ([X̃,P̃_X])=i` (exact at every N — derived in the gate doc §1.1).
  `E/N²=τ(P̃_X²)+τ(P̃_Y²)+m²(τ(X̃²)+τ(Ỹ²))−λτ([X̃,Ỹ]²)`.
- **No cyclicity for words containing momenta** (gate doc §1.2): `τ(O₁O₂)−τ(O₂O₁)=τ([O₁,O₂])≠0`;
  words are operator-valued and kept strictly ordered; a cyclic move across a non-commuting letter
  carries the explicit `Σ_{ij}[a_{ij},b_{ji}]` correction.
- **Generator normalization** (gate doc §1.3): a single-trace generator scales as `Â=N·Tr(â)=N²·τ(â)`
  (the 't Hooft `S=N·Tr[…]` scaling) for an `O(1)` flow.
- **Reference state `|G⟩`:** the Gaussian (frequency-Ω, X↔Y-symmetric) squeezed vacuum, the `g=0`
  ground state of `Tr(P̃²+m²X̃²)`; Wick moments. `Â=0 ⟹ |ψ⟩=|G⟩` (the family contains the Gaussian).
- **Symmetries preserved:** `X↔Y`, `X→−X`, `Y→−Y`, SU(N) singlet, cyclicity *of c-number observables
  only*. `Â` built in a Weyl-ordered, symmetry-projected Hermitian single-trace basis.

---

## Architecture

**1. The variational object.** `|ψ⟩=e^{iÂ}|G⟩`, `Â=N²·Σ_a θ_a τ(â_a)` a Hermitian (Weyl-ordered)
single-trace polynomial in `{X̃,Ỹ,P̃_X,P̃_Y}` of degree `≤ d`, symmetry-projected. Parameters
`θ=(Ω, {θ_a})`. `θ_a=0` recovers the Gaussian.

**2. Rigorous finite-N bound (the certified core).** At fixed N, `e^{iÂ}` is unitary and `|G⟩` is
unit-norm, so `E_hi(N;θ)=⟨ψ|H|ψ⟩ ≥ E₀(N)` by Rayleigh–Ritz — **for every θ**, exactly. Computed in
the truncated bosonic Fock space (reuse `exact_diag.py`: mode basis + occupation-truncated ladders);
the only error is the **monotone, one-sided Fock-`K` truncation** (it can only *raise* the Rayleigh
quotient, so the upper-bound property is preserved). Minimize over `θ`. **Validated against SP1:**
`E₀(N) ≤ E_hi(N;θ*)`, and `E_hi → E₀` as the family/`K` grow (SP1's exact `E₀(2)=2.2578` at λ=1 is the
floor the variational bound must respect and approach).

**3. The large-N surrogate (S1, proven).** The moment-flow `dm_s[w]/ds=i·m_s[(scaled)[Tr w,Â]]`
(factorized, integrated `s:0→1`) gives, fast and Fock-free: (a) an **initializer** for the finite-N
search (the planar energy landscape over `Â`); (b) the **N→∞ master-field object** (the limiting
single-trace configuration, proven to exist + close); (c) a **consistency cross-check** on the
extrapolation. **It is an estimate, NOT a bound** (S2) — used only to guide and to define the N→∞
object, never as the certified number.

**4. N→∞ master field.** Extrapolate the finite-N rigorous bounds `E_hi(N;θ*(N))` to N→∞ (the
master-field energy), with the surrogate flow as the N=∞ anchor and consistency check; report the
result inside the certified bracket `[E_lo (bootstrap SDP), E_hi (Gaussian)]` and against SP1's
exact-diag extrapolation (≈2.34 at λ=1). The reusable master-field object is `θ*` + the limiting flow.

**5. Rigor posture (honest, post-gate).** The *certified* statements are the finite-N Rayleigh–Ritz
upper bounds (exact by construction, Fock-`K` one-sided) and the bootstrap SDP lower bound; the
surrogate flow and the N→∞ extrapolation are *estimates* (stated as such). No quantity is claimed a
bound unless it is `⟨literal state|H|literal state⟩` or an SDP-certified relaxation.

---

## Modules
- **`matrix_master_field/momentum_master_field.py` (new):** the symmetry-projected Weyl-ordered
  single-trace generator basis; `Â`-builder; the finite-N variational energy `E_hi(N;θ)` in the
  truncated Fock space (reusing `exact_diag`); the optimizer over `θ`; the large-N moment-flow
  surrogate (reusing the gate's `sp2_flow_test.py` machinery) for initialization + the N→∞ object;
  the N→∞ extrapolation + bracket assembly.
- **Tests:** the validation obligations below.
- **Reuses:** `exact_diag.py` (finite-N Fock + the SP1 benchmark), `qm_master_field.py` (Gaussian +
  bootstrap SDP lower bound), the gate derivation's verified flow code.

---

## Validation obligations

| # | Claim | Check |
|---|-------|-------|
| V1 | `Â=0` recovers the Gaussian | `E_hi(N; θ=0)` = the Gaussian energy (large-N → `2.36452` at λ=1), exactly |
| V2 | `λ=0` anchor (exact) | `E_hi = 2m` to machine precision; optimum at `Â=0, Ω=m` |
| V3 | finite-N bound is a TRUE upper bound | `E_hi(N;θ*) ≥ E₀(N)` (SP1 exact) at N=2,3 for λ∈{0,0.5,1}; never below |
| V4 | bound tightens toward exact | `E_hi(N;θ*)` decreases monotonically as `d`/`K` grow, approaching SP1's `E₀(N)` from above |
| V5 | surrogate ↔ finite-N consistency | the large-N flow estimate matches `E_hi(N;θ*)` as N grows (the surrogate is faithful) — and the flow's closure invariants (`τ([X̃,P̃])=i`, generator scaling) hold (gate Tests C/E) |
| V6 | CCR / ordering correctness | no cyclicity used on momentum words; `τ(X̃P̃)−τ(P̃X̃)=i` exact at finite N in the built operators |
| V7 | the deliverable | the master field `θ*` + `E_hi(N;θ*)` at N=2,3 (and any feasible N>3) for λ∈{0,0.5,1}, the N→∞ extrapolation, and the certified bracket `[SDP, Gaussian]` + the SP1 comparison |

Verification medium: Python + `pytest` (numpy/scipy), as M1–M5/SP1; cross-checked against
`exact_diag` (SP1, exact) and the Gaussian/SDP (M5b/c).

---

## Risks / open questions
- **R1 — reaching N beyond exact-diag's ceiling (the key value lever).** Computing `E_hi(N;θ)` in the
  Fock space caps at N≈3 (same wall as exact-diag: `2(N²−1)` modes). At N=2,3 the variational bound is
  *looser* than SP1's exact diagonalization, so B adds **no new number** there — only the reusable
  object + per-N rigor. Genuinely new numbers require an exact `E_hi(N)` evaluation that scales past
  N=3. *Candidate routes (to assess in planning, NOT assumed):* (i) `e^{-iÂ}He^{iÂ}` via a BCH series
  with a **controlled, sign-known remainder** (a true bound only if the remainder is bounded — else it
  is the C7/M5c trap and must be rejected); (ii) a structured/compressed representation of `e^{iÂ}|G⟩`
  with closed-form Wick moments. If neither pans out, B's honest deliverable is the object + bracket at
  N≤3 (still a legitimate reusable master field, just not a tighter energy than SP1).
- **R2 — expressiveness.** The `Â`-family may not improve on the Gaussian; the true gap is only ~0.025
  at λ=1 (SP1). A null result (master field ≈ Gaussian) is a legitimate, trustworthy outcome.
- **R3 — surrogate↔bound mismatch.** If the large-N flow estimate disagrees with the finite-N bounds'
  extrapolation, trust the finite-N bounds (certified) + SP1; treat the flow as falsified-as-surrogate
  and report the discrepancy (do not paper over it).
- **R4 — numerics.** Variational landscape non-convex; the Fock build heavy at N=3. *Mitigation:*
  surrogate-flow initialization (R1 makes this cheap), multi-start, SP1 as the artifact gate.

## Out of scope
- Downstream BFSS/BMN application of the master field (future milestone).
- A tightened certified **lower** bound (separate hard problem; SDP + Gaussian already bracket).
- A certified upper bound *better than the Gaussian* at N→∞ — desirable but not promised here; it
  hinges on R1 succeeding.
