# SP2 — R1 feasibility gate: is there a SCALABLE, Fock-free, EXACT evaluation of the non-Gaussian variational upper bound `E_hi(N;θ)`?

**Date:** 2026-06-28
**Branch:** `matrix-master-field`
**Verification script:** `matrix_master_field/derivations/sp2_r1_degree_growth.py`
(run: `PYTHONPATH=<repo root> uv run --no-project --with numpy --with scipy python matrix_master_field/derivations/sp2_r1_degree_growth.py`; runs end-to-end)
**Numerics reuse:** the bosonic-Fock construction (Hermitian mode basis + occupation-truncated ladder operators) of `matrix_master_field/exact_diag.py`, `matrix_master_field/qm_fock.py`, and `matrix_master_field/derivations/sp2_flow_test.py`.

> **The question (R1).** For the trial `|ψ⟩=e^{iÂ}|G⟩` with Â a Hermitian **single-trace** polynomial in `{X̃,Ỹ,P̃}` (normalization `Â=N²τ(â)`, from `sp2-moment-flow-closure.md`) and `|G⟩` the Gaussian reference, the variational upper bound is `E_hi(N;θ)=⟨ψ|H|ψ⟩=⟨G|e^{-iÂ}He^{iÂ}|G⟩`. **Is there a scalable (Fock-space-free) way to evaluate `E_hi` EXACTLY for NON-Gaussian (cubic-or-higher) Â, at N beyond the exact-diag ceiling (N≳4), that PRESERVES the upper-bound property?**
>
> **Verdict up front: R1 FAILS.** Only the **quadratic** (Bogoliubov/squeezed) family resums to a Fock-free closed form, and that family *is* the Gaussian — no improvement. For **cubic+** Â there is no finite closure of the Heisenberg series; the only exact finite-N evaluation lives in the (exponential) Fock space, which caps at N=3; and the only Fock-free large-N surrogate (the moment-flow) is **not** an upper bound (S2). Consequently B is limited to N≤3, where it is necessarily **looser** than the exact diagonalization SP1 already has, so its only deliverable is the reusable variational object, not a better number. **Recommend pausing the full SP2 build.**

---

## Conventions (pinned — used verbatim, from `CONVENTIONS.md` and `sp2-moment-flow-closure.md`)

- Model (HHK arXiv:2004.10212 Eq 17): `H = Tr(P_X² + P_Y² + m²(X²+Y²) − g²[X,Y]²)`, Hermitian `N×N`, `ℏ=1`, `m=1`, matrix CCR `[X_ij,P_{X,kl}]=i δ_il δ_jk`.
- 't Hooft scaling `X̃=X/√N`, `P̃=P/√N` ⟹ `[X̃_ab,P̃_cd]=(i/N)δ_ad δ_bc`; `τ(·)=(1/N)Tr`; `λ=Ng²` fixed; report `E/N²`.
- `|G⟩` = the `g=0` (free) ground state of `Tr(P̃²+X̃²)` (frequency `m=1`); the squeezed/oscillator vacuum.
- Generator normalization (DERIVED in `sp2-moment-flow-closure.md` Part 1.3): `Â=N·Tr(â)=N²·τ(â)`.
- Heisenberg flow of the trial: `O(s)=e^{-isÂ}O e^{isÂ}`, `dO/ds=i[O,Â]`, `s:0→1`; `E_hi=⟨G|H(1)|G⟩`.
- All numerics below are in the single-Hermitian-matrix sector (`X̃`, `P̃`; words over `{0=X̃,1=P̃}`), which is sufficient to exercise the matrix CCR and the degree-growth mechanism. The `Ỹ` sector and the `[X̃,Ỹ]²` interaction only *raise* the relevant degrees (they never lower the growth rate), so the obstruction is, if anything, stronger in the full model.

This file is the R1 companion to `sp2-moment-flow-closure.md` (which settled S1 closure / S2 truncation for the *large-N* flow). R1 is the sharper, distinct question: can the *finite-N, exact, upper-bound* number be gotten Fock-free for non-Gaussian Â?

---

## Part 0 — Why a Fock-free route is needed at all (the exact-diag ceiling, quantified)

Both ways of computing `E_hi=⟨G|e^{-iÂ}He^{iÂ}|G⟩` exactly at finite N live in the **bosonic Fock space of the `2N²` Hermitian modes** (occupation truncation `Σ nᵢ ≤ K`): either evolve the state `e^{iÂ}|G⟩` and take `⟨H⟩`, or evolve the operator `e^{-iÂ}He^{iÂ}` and take `⟨G|·|G⟩`. Both `exact_diag.py` and `qm_fock.py` do exactly this. The Fock dimension `D = C(2(N²−1)+Kₚ, Kₚ)` (built on a padded `Kₚ=K+2` basis) is exponential in `N²` (script Part 0 / the `dim` probe):

| N | modes `2(N²−1)` | D at K=6 | D at K=8 |
|---|---|---|---|
| 2 | 6  | 3.0e3 | 8.0e3 |
| 3 | 16 | 7.4e5 | 5.3e6 |
| 4 | 30 | 4.9e7 | 8.5e8 |
| 5 | 48 | 1.4e9 | 5.2e10 |

The quartic assembly is additionally `O(n_tl²)=O(N⁴)` sparse mat-muls of `O(N²)`-term operators. Exact diagonalization (SP1) therefore tops out at **N=3** (already capped at K=6 there, `D~7e5`, with a residual K-tail; N=4 is reachable only at tiny, unconverged K). **Any Fock-based evaluation of the B trial inherits the same ceiling.** R1 asks whether the *single-trace* structure of Â lets us bypass the Fock space and evaluate `E_hi` exactly for N≳4. The answer below is no (except for the Gaussian-equivalent quadratic case).

---

## Part 1 — Route 1(a): BCH/Heisenberg resummation. Closes ONLY for quadratic Â (= Gaussian).

`e^{-iÂ}He^{iÂ}=Σ_{n≥0} (1/n!)(ad_{-iÂ})^n H`, `ad_{-iÂ}H=-i[Â,H]`. Each `⟨G|(ad)^n H|G⟩` is a Gaussian (Wick) expectation — exact and Fock-free. The series **resums to a closed form** (exact at any N) iff the nested-commutator action **closes on a finite-dimensional operator space**.

### 1.1 The degree-counting lemma (the exact obstruction)

Assign each polynomial in `{X̃_ij,P̃_ij}` its **total degree** = number of letters. The CCR `[X̃_ab,P̃_cd]=(i/N)δ_ad δ_bc` means **one canonical contraction removes one `X̃` and one `P̃`**, replacing the pair by the c-number `(i/N)` times the surrounding letters. Hence by Leibniz:

```
deg([A,O])_max = deg(A) + deg(O) − 2.                                  (R1.1)
```

(The leading single-contraction term has degree `deg A + deg O − 2`; multi-contraction terms have strictly lower degree by even amounts. The maximum is what controls closure.) Iterating from `deg(H)=2` (the `X`-sector density `Tr(P̃²+X̃²)`; the `[X̃,Ỹ]²` interaction has degree 4, which only shifts the constant, not the rate):

```
deg((ad_Â)^n H)_max = deg(H) + n·(deg(Â) − 2).                          (R1.2)
```

| deg(Â) | `deg((ad)^n H)_max` | closure |
|---|---|---|
| **2** (quadratic) | `2` — **constant** | finite operator space → **series RESUMS** |
| **3** (cubic) | `2 + n` — grows `+1`/order | unbounded → **no finite closure** |
| **4** (quartic) | `2 + 2n` — grows `+2`/order | unbounded → no finite closure |

**Verified two ways** (script Parts 1 & 2a):
- *Symbolic* (N-independent, exact): Part 1 prints the table above for `n=0..8`.
- *Fock cross-check* (N=2,K=10 and N=3,K=8): a degree-`D` polynomial in `{X̃,P̃}={(a+a†),−i(a−a†)}/√2` connects occupation `n↔n±D`, so the **occupation bandwidth** of the sparse nested commutator equals its polynomial degree. Measured bandwidths: quadratic `Â=N Tr(X̃²)` → `[0,2,4,6,8,10]` (constant *content*, the value just reflects nesting depth — see 1.2); cubic `Â=N Tr(X̃³)` and `Â=(N/2)Tr(X̃²P̃+P̃X̃²)` → `[0,3,6,9,...]`, i.e. degree `=2+n` exactly, until clipped by K. Agrees with (R1.2) to the bandwidth resolution.

### 1.2 Quadratic Â resums to the linear (Bogoliubov) map — and that is the Gaussian

For a Hermitian **quadratic** Â the Heisenberg map is **symplectic**: `X̃(1),P̃(1)` are **linear** in `X̃(0),P̃(0)`, so `e^{iÂ}|G⟩` is a **squeezed Gaussian state**. Concretely (script Part 2b), the dilatation/squeeze `Â=(N/2)Tr(X̃P̃+P̃X̃)` has the DERIVED (and numerically confirmed, `dX̃/ds≈−1.000·X̃`) matrix EOM

```
dX̃/ds = −X̃,   dP̃/ds = +P̃   ⟹   X̃(s)=e^{−s}X̃₀,  P̃(s)=e^{+s}P̃₀,        (R1.3)
```

linear at **any N**. The exact Fock flow matches `e^{∓s}` to a residual that is **pure K-truncation**, vanishing as K→∞ at fixed `s=0.1` (script Part 2b):

| K (N=2) | `max|X̃(s)|G⟩ − e^{−s}X̃|G⟩|` |
|---|---|
| 8  | 5.79e−6 |
| 12 | 5.62e−8 |
| 16 | 5.14e−10 |
| 20 | 5.25e−12 |

(factor ~10² per ΔK=4 — the operator series genuinely **resums** to the linear map). **So Route 1(a) succeeds for quadratic Â — but the quadratic/Bogoliubov family is exactly the Gaussian (squeezed) states.** This recovers the existing Gaussian upper bound (the frequency-optimized `gaussian_upper`, `E/N²=2.365` at `λ=1`), giving **no improvement**. This is the hypothesis in the gate, here confirmed by derivation + numerics.

### 1.3 Cubic+ Â does NOT close — the operator space is infinite-dimensional

By (R1.2) with `deg Â≥3`, the operator degree grows without bound, so the nested commutators never re-enter a finite operator basis — there is no closed-form resummation. The Frobenius norms `||(ad_Â)^n H||_F` confirm there is no decay/cycling (script Part 2a, N=2,K=10): cubic `Â=N Tr(X̃³)` gives `2.6e2, 1.5e3, 5.5e4, 2.8e6, 1.7e8, 1.1e10` — monotone super-exponential growth in the operator norm, the antithesis of closure. **R1(a) fails for every non-Gaussian (cubic+) Â.**

---

## Part 2 — Route 1(b): truncate the series + rigorous remainder bound? The unbounded-generator obstruction.

If the series does not terminate, can a finite truncation plus a **rigorous upper bound on the remainder** still sandwich `⟨ψ|H|ψ⟩`? The obstruction is that `Â` is an **unbounded** operator (a polynomial in `P̃`), and the BCH series in `s` has **finite radius of convergence `< 1`**, so it **diverges at `s=1`** — there is no convergent series whose tail could be bounded.

### 2.1 The expectation series can mislead — then generically diverges

A crucial subtlety (script Part 2c): for the **position-only affine** generator `Â=N Tr(X̃³)`, the flow is `P̃(s)=P̃₀+3sX̃²` (affine in P̃; `X̃` frozen — DERIVED in `sp2-moment-flow-closure.md` (6)), so `⟨G|H(s)|G⟩` is a **finite polynomial in s** and the BCH *expectation* series **terminates** (`⟨(ad)^n H⟩ = +2, 0, −20.25, 0, ~0,...`; residuals beyond n=2 are pure K-clipping). **This is a non-generic special case** — and even there the *operator* series is infinite (Part 1.3), it is only the Gaussian expectation that truncates, and that truncation does not survive once the full `H` (both sectors + `[X̃,Ỹ]²`) flows.

The **generic** cubic generator `Â=(N/2)Tr(X̃²P̃+P̃X̃²)` drives the nonlinear **Riccati** flow `dX̃/ds=−X̃²` ⟹ `X̃(s)=X̃₀(1+sX̃₀)⁻¹` (DERIVED in `sp2-moment-flow-closure.md` (8)). Its BCH expectation series **does not terminate and explodes** (script Part 2c, N=2):

```
⟨(ad)^n H⟩ = +2.00,  0,  −12.25,  0,  +240,  0,  −2.07e4,  0,  +4.02e6,  0,  −1.42e9,  ...
```

with **growing** consecutive-term ratios `6.1, 19.6, ...` — the Cauchy–Hadamard signature of a **finite radius of convergence in `s`**, not a term-bounded series.

### 2.2 The pole: `X̃(1)` is genuinely singular on a finite-measure set of Gaussian configurations

The Riccati operator `X̃(s)=X̃₀(1+sX̃₀)⁻¹` is **singular** at `s=−1/λ` for each eigenvalue `λ` of `X̃₀`. We flow to `s=1`. Sampling `X̃₀` from the Gaussian reference `|G⟩` (mode amplitudes `~N(0,½)`, the oscillator-vacuum Wigner distribution; script's classical Wigner probe over `hermitian_basis`):

| N | `⟨λ_min(X̃₀)⟩` | most-negative seen | `P(λ_min ≤ −1 ⟹ pole at s≤1)` |
|---|---|---|---|
| 3 | −0.771 | −1.895 | 24.2% |
| 5 | −0.966 | −1.869 | 43.5% |
| 8 | −1.091 | −1.633 | 71.4% |

The lower spectral edge sinks below `−1` on a finite-measure set (rising toward 100% with N, since the Wigner edge of `X̃₀` approaches and then crosses `−1`). On that set the Heisenberg-evolved operator at `s=1` is **literally singular**, so its Taylor/BCH series in `s` **cannot converge at `s=1`**. **No finite truncation + remainder bound can reproduce the (finite, exact) matrix element `⟨G|e^{-iÂ}He^{iÂ}|G⟩` from a divergent series.** (The matrix element itself is finite — `e^{iÂ}` is unitary — but it is *not* the sum of the BCH series at `s=1`; the series is asymptotic at best, and an asymptotic series gives **no rigorous one-sided remainder bound**, so the upper-bound property is lost.) **R1(b) fails.**

> Remark (sharper than the gate's phrasing). The gate worried about convergence of an unbounded `Â`; the concrete mechanism is the finite-`s` pole of the nonlinear flow on the unbounded spectrum. The lone exception (affine `Tr(X̃³)`, entire flow) is exactly the generator that produces **no genuine non-Gaussian structure** in the variational energy (it shifts `P̃` by `X̃²` but leaves the state's `X̃`-marginal Gaussian), so it cannot beat the Gaussian either.

---

## Part 3 — Routes 2 & 3: structured/compressed state, or any other exact Fock-free route. None preserves the bound.

**Route 2 (structured / compressed state).** A Fock-free *exact* evaluation needs a representation of `e^{iÂ}|G⟩` (cubic+ Â) with **closed-form finite-N moments**. The candidates and why each collapses to "Gaussian or Fock-space":

- *Coherent/squeezed (Gaussian) states* — exactly the quadratic family of Part 1.2; closed-form moments, but no improvement over the Gaussian. (Degree-2 expressiveness only reaches semicircular/Gaussian states; cf. the M3 lesson in `MEMORY.md`.)
- *Tensor network / matrix-product state in the `2N²` modes* — `e^{iÂ}` with cubic+ Â is a non-Gaussian unitary built from the **single-trace** `τ(â)`, which couples **all `O(N²)` modes** (e.g. `Tr(X̃²P̃)=Σ_{a,b,c}(T_aT_bT_c\text{-overlap})x_a x_b p_c`); together with the `[X̃,Ỹ]²` interaction this generates correlations across all modes with no fixed-bond-dimension factorization. Exact finite-N moments then require bond dimension growing with N (→ the Fock dimension again).
- *Naive finite-N moment flow* — the moment hierarchy of `e^{iÂ}|G⟩` does **not** close on single traces at finite N (closure is a large-N statement; `sp2-moment-flow-closure.md` S1). Finite-N closure needs **all multi-trace moments** = the full Fock data. No bypass.

**Route 3 (any other exact Fock-free route).** The only Fock-free *exact* object available is the **large-N moment-flow**, and `sp2-moment-flow-closure.md` **S2 already proved it is not a bound**: the L-truncated planar-flow energy is biased *from below* and, under minimization over Â, drifts unsafely below `E₀` (the M5c selection mechanism). Positivity/SDP certifies only the **lower** side (and does not even use the trial). There is no positivity route that turns the truncated planar trial into a rigorous **upper** bound, because moment-matrix relaxations are intrinsically one-sided (lower). So Route 3 = "use the moment-flow" = not an upper bound.

There is no large-N or integrability simplification of `⟨G|e^{-iÂ}He^{iÂ}|G⟩` at *finite* N for non-Gaussian Â: it is a genuine interacting matrix element of a non-Gaussian unitary, and its exact value requires the exponential Fock data.

---

## Part 4 — Verdict and recommendation

**R1 FAILS.** Tabulating the routes:

| Route | Result |
|---|---|
| 1(a) BCH resummation, **quadratic** Â | **Closes** (linear/Bogoliubov, Part 1.2) — but **is the Gaussian**; no improvement. |
| 1(a) BCH resummation, **cubic+** Â | **No finite closure** (degree grows `+(d−2)`/order, R1.2; norms blow up). |
| 1(b) truncate + remainder bound, cubic+ | **Fails** — unbounded Â; flow has a **finite-`s` pole** on the Gaussian spectrum (24%→71% of configs, N=3→8) ⟹ BCH series **diverges at `s=1`**; asymptotic series gives **no one-sided remainder bound**. |
| 2 structured/compressed state | Collapses to **Gaussian** (no gain) or to a representation whose exact finite-N moments need the **full Fock space**. |
| 3 other exact Fock-free (moment-flow) | The only Fock-free exact object is the planar flow, already proven **not an upper bound** (S2); positivity gives only the lower side. |

So **no scalable exact non-Gaussian upper-bound evaluation exists**: the only Fock-free closed form is the quadratic/Gaussian case (confirmed here by derivation + the K-converging Bogoliubov check), and every non-Gaussian route either loses the upper-bound property (1b, 3) or reverts to the exponential Fock space (1a-cubic, 2).

**Consequence for sub-project B (operator master field via `e^{iÂ}|G⟩`).** B's exact, upper-bound number is computable **only in the Fock space**, hence **only at N≤3** (Part 0). But at N≤3 the **exact diagonalization SP1 already returns the true ground-state energy** (`E/N²`: N=2,K=12 → 2.2578; N=3,K=6 → 2.3076; large-N extrap ≈ 2.34±0.06 at λ=1, per `MEMORY.md`/`exact_diag.py`), which is the variational minimum over **all** states. Any single trial `e^{iÂ}|G⟩` evaluated in the same Fock space is `≥` that exact value — **strictly looser than SP1**. Therefore B at N≤3 yields **no better number**; its only deliverable is the **reusable variational object** (the parametrized `e^{iÂ}|G⟩` machinery and the planar-flow surrogate/initializer that S1 certifies exists), not an improved energy.

**Recommendation: pause / reconsider the full SP2 (B) build.** Decisively:

1. The thing that would make B worth building — a scalable Fock-free **exact upper-bound** evaluation for non-Gaussian Â at N≳4 — **does not exist** (this gate). Without it, B cannot produce a number SP1 doesn't already have, and at N≤3 it is provably looser.
2. The genuine open route to a *certified non-Gaussian* result remains the **direct-momentum `P̂` bound** (`sp2-moment-flow-closure.md` and the `2026-06-27-m5c-direct-momentum-master-field-design.md` spec / `MEMORY.md` sub-project 2): build an explicit `P̂` with `[X̂,P̂]=i` so `⟨P̂²⟩` is computed **directly** → a true variational **upper** bound that cannot be exploited downward. That is where the effort should go, not into the `e^{iÂ}|G⟩` Fock evaluation.

**What survives from B (and is real):** the large-N **moment-flow** (S1: closed on single traces, factorization-driven, machine-verified) remains a **sound, cheap surrogate/initializer** for any finite-N variational search and a consistency check on `N→∞` extrapolation — but the *certified* number must come from a finite-N exact expectation (SP1) or the direct-`P̂` construction, **not** from `e^{iÂ}|G⟩`.

---

## Appendix — verification script map (`sp2_r1_degree_growth.py`)

| Part | What it checks | Headline result |
|---|---|---|
| **1** | Symbolic degree growth `deg((ad)^n H)=2+n(d−2)` (N-independent) | d=2 constant; d=3 → 2+n; d=4 → 2+2n |
| **2a** | Fock bandwidth = polynomial degree (N=2,K=10; N=3,K=8); operator-norm growth | quadratic content bounded; cubic bandwidth `[0,3,6,9,…]`, norms ↑ super-exp |
| **2b** | Quadratic Â resums to the linear Bogoliubov map (K-convergence) | residual `5.8e−6→5.3e−12` over K=8→20 (pure truncation) — exact resummation |
| **2c** | Cubic BCH expectation series: affine TERMINATES (special) vs Riccati DIVERGES (generic) | Riccati `⟨(ad)^n H⟩` ratios `6.1,19.6,…` ↑ ⟹ finite radius, diverges at s=1 |
| Part 0 / 2.2 probes | Fock-dim ceiling; Riccati pole `P(λ_min≤−1)` over the Gaussian spectrum | N=4 K=8 D~8.5e8; pole-fraction 24%(N=3)→71%(N=8) |

Reuses `build`, `word_mat`, `Tr_op`, `tau_op`, `expval`, `fock_ladders` from `sp2_flow_test.py` and `hermitian_basis` from `exact_diag.py`. All operator products kept ordered (no cyclicity). Float64 throughout.
