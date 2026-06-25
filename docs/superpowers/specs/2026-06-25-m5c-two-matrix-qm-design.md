# M5c — Two-matrix quantum mechanics as a certified SDP ↔ master-field sandwich

**Date:** 2026-06-25  **Status:** design (scope approved in brainstorming — full sandwich
*including* the operator master field, Gaussian baseline first; **revised after Codex adversarial
audit — see Revision log**)

**Goal:** Construct the large-$N$ master field of the *genuinely unsolvable* two-matrix quantum
mechanics $H=\mathrm{Tr}(P_X^2+P_Y^2+m^2(X^2+Y^2)-g^2[X,Y]^2)$ (HHK Eq 17) by squeezing the
ground-state energy density $E/N^2$ between a **certified SDP lower bound** and a **rigorous
Gaussian master-field upper bound**, with a **novel free-probability ("free-Fisher") operator
master field** as the sharp estimate inside the bracket — refereed by the $g{=}0$ exact anchor and
by HHK's published bootstrap numbers.

**Where this sits.** Third and final rung of Milestone 5 (matrix QM). M5a built the QM machinery
on $N{=}1$; M5b lifted it to a *solvable* large-$N$ matrix (single matrix = free fermions /
collective field). M5c is the step the whole project points at: **two matrices cannot be
simultaneously diagonalized, so the eigenvalue-density / free-fermion collective field that made
M5b solvable does not exist.** This is exactly where the project's thesis — the operator master
field (Cuntz–Fock + ML, positivity automatic) — earns its keep on a Hamiltonian ground state. The
$g{=}0$ limit decouples into two free matrix oscillators (the exact anchor); $g{>}0$ has no closed
form and HHK's bootstrap is the external reference.

---

## Conventions (pinned — transcribed from HHK arXiv:2004.10212, not guessed)

- **Model (HHK Eq 17, verbatim):**
  $$H = \mathrm{Tr}\big(P_X^2 + P_Y^2 + m^2(X^2+Y^2) - g^2[X,Y]^2\big).$$
  $X,Y$ Hermitian $N{\times}N$. Kinetic coefficient $1$ (no $\tfrac12$), explicit mass $m^2$,
  confining commutator $-g^2[X,Y]^2$ ($\mathrm{Tr}[X,Y]^2\le0$ since $[X,Y]$ is anti-Hermitian).
- **Canonical (HHK):** $[P_X^{ij},X_{kl}]=-i\,\delta_{il}\delta_{jk}$, i.e. **$[X_{ij},P_{X,kl}]=+i\,\delta_{il}\delta_{jk}$** (and likewise $Y$). $\hbar=1$. Same convention as M5a/M5b.
- **Symmetries:** O(2) rotation in $(X,Y)$; $\mathbb{Z}_2{\times}\mathbb{Z}_2$ — $X\to-X$ and
  $Y\to-Y$ independently (every term even in each matrix) ⟹ any word with odd total count of
  $X$/$P_X$ **or** of $Y$/$P_Y$ vanishes. (Same symmetry structure as the M3/M4 two-matrix models.)
- **'t Hooft scaling (PROVISIONAL — must be pinned in T1 before any SDP coefficient is frozen):**
  $X=\sqrt N\,\tilde X$, $P_X=\sqrt N\,\tilde P_X$ (etc.), $g^2=\lambda/N$ with $\lambda=Ng^2$ fixed.
  Normalized moments $m[w]=(1/N)\langle\mathrm{Tr}\,w\rangle=O(1)$ for words $w$ in
  $\{\tilde X,\tilde Y,\tilde P_X,\tilde P_Y\}$; $m[\varnothing]=1$ (hard). Hermiticity
  $m[w]^*=m[\mathrm{reverse}(w)]$. HHK's published dial is dimensionless $m^2/g^{4/3}$; the exact
  map $(m^2/g^{4/3})\leftrightarrow(m,\lambda)$ and whether $\tilde X=X/\sqrt N$ is HHK's own
  normalization are **not yet pinned** (T1).
- **Energy density (PROVISIONAL normalization — form fixed, coefficients follow T1):**
  $$\frac{E}{N^2}=m[\tilde P_X^2]+m[\tilde P_Y^2]+m^2\big(m[\tilde X^2]+m[\tilde Y^2]\big)-\lambda\,m\big[[\tilde X,\tilde Y]^2\big].$$
  *Which* moments enter and with *which signs* is fixed; the numerical normalization of each term is
  provisional until T1 is discharged.
- **$g{=}0$ exact anchor (derived, all $N$).** At $\lambda{=}0$, $H=H_1[X]\oplus H_1[Y]$ with
  $H_1=\mathrm{Tr}P^2+m^2\mathrm{Tr}X^2$. Each of the $N^2$ real modes is $h=p^2+m^2x^2=2(\tfrac12 p^2+\tfrac12 m^2 x^2)$, spectrum $(2n{+}1)m$, ground energy $m$; so $E_1=N^2 m$ and
  $$\boxed{E/N^2 = 2m}\ \ (=2\text{ at }m{=}1),\qquad m[\tilde X^2]=m[\tilde Y^2]=\tfrac{1}{2m},\quad m[\tilde P_X^2]=m[\tilde P_Y^2]=\tfrac{m}{2}.$$
  Every piece of the sandwich must reproduce this. (At $m{=}1$ this is two copies of the M5b
  $g{=}0$ result $E/N^2{=}1$, $m[\tilde X^2]{=}\tfrac12$.) **This anchor value is
  convention-independent** — it is the physical ground energy $2N^2m$ of Eq 17 at $g{=}0$ divided by
  $N^2$, so it does *not* depend on T1; only the *expression* of $E/N^2$ in normalized moments does.

### Transcription / derivation obligations — TO BE discharged in `derivations/m5c-two-matrix-qm.md` (not yet written)
These are **open deliverables, not completed work.** **Hard gate: no SDP coefficient, Gauss-law
constant, energy normalization, or module API may be frozen in code until T1, T2, T4, T5 have each
passed their numerical checks** (against the $g{=}0$ exact moments / the M5b 1-matrix limit) in the
derivation file. This gate is the implementation-plan's first wave.
- **T1 — exact large-$N$ scaling & coupling dial.** HHK quote a dimensionless control
  $m^2/g^{4/3}$ and 't Hooft $\lambda=Ng^2$. Pin their *exact* convention (whether $\tilde X=X/\sqrt N$ as above, and how $m^2/g^{4/3}$ maps to $(m,\lambda)$) from their two-matrix section before fixing the SDP's coefficients. The architecture below is scaling-convention-independent in *form*; only the numerical coefficients depend on T1.
- **T2 — SU(N) Gauss-law constant.** Two-matrix generator $G=i([X,P_X]+[Y,P_Y])+c\,\mathbb{1}$;
  derive the normal-ordering constant $c$ (single-matrix value was $N$, HHK Eq 11) and the singlet
  descendants $\langle\mathrm{Tr}(G\,\mathcal O)\rangle=0$ (the analog of M5b's $\langle\mathrm{Tr}XP\rangle=iN^2/2$ for each canonical pair).
- **T3 — HHK referee numbers (soft cross-check; resolved post-audit, see R3).** Extract $E/N^2$ vs
  $m^2/g^{4/3}$ from HHK Fig. 3 (a *figure*, like the KZ tables — extract with care; cross-check the
  extracted $\lambda{=}0$ point against the exact $2m$). In-scope as a *reported* cross-check **if
  digitizable**; `validated` does not hinge on it.
- **T4 — kinetic-energy = $\tfrac14$ free Fisher information.** Derive
  $m[\tilde P_X^2]+m[\tilde P_Y^2]=\tfrac14\Phi^*(\tilde X,\tilde Y)$ for the two-matrix ground
  state (see §C3); verify the 1-matrix reduction to M5b and the anchor.
- **T5 — stationarity loop equations.** Derive $\langle[H,\mathrm{Tr}\,w]\rangle=0$ for words in
  $\{\tilde X,\tilde Y,\tilde P_X,\tilde P_Y\}$ (the QM loop equations), and **verify each against
  the $g{=}0$ exact moments** (residual $\sim0$) before trusting the SDP — exactly the M5b discipline.

---

## Architecture — the staged sandwich

```
   E_lo (SDP, certified)   ≤    E/N²(m,λ)    ≤    E_hi  (Gaussian master field, rigorous)
                                                 ‖
                                E_MF  (free-Fisher operator master field — sharp estimate)

   anchored at λ=0 by the exact  E/N² = 2m ;   refereed at λ>0 by HHK Fig. 3 (T3)
```

The **certified bracket is `[E_lo (SDP), E_hi (Gaussian)]`.** The novel free-Fisher master field
$E_{\rm MF}$ is the *sharp* number that should land inside it (and on HHK), but its status as an
upper bound is qualified — see §C3. Staging the Gaussian first guarantees a valid sandwich before
the research-grade piece is attempted.

---

## C. The three pieces

### C1 — SDP bootstrap: certified lower bound (extends M5b two→words)
- **Variables:** ordered single-trace moments $m[w]$, $w$ a word in $\{\tilde X,\tilde Y,\tilde
  P_X,\tilde P_Y\}$ up to length $L$, modulo hermiticity + $\mathbb{Z}_2{\times}\mathbb{Z}_2$/O(2) +
  cyclicity. **Keep the moments ordered — do NOT reduce via a c-number commutator** (the central
  M5b lesson: $[\tilde X,\tilde P]$ has operator parts; a c-number reduction silently collapses the
  bootstrap to a single particle).
- **Constraints:** (i) $m[\varnothing]=1$; (ii) **stationarity** $\langle[H,\mathrm{Tr}\,w]\rangle=0$
  (T5) — the QM loop equations relating $m[w]$; (iii) **SU(N) Gauss law** (T2),
  $\langle\mathrm{Tr}(G\,\mathcal O)\rangle=0$ — forces the phase-space area that prevents the
  trivial $E/N^2\ge0$ collapse; (iv) ~~large-$N$ factorization via a product matrix~~ **— omitted
  (corrected during execution):** the M3 `_bootstrap_two_matrix` product-matrix device
  ($Q[0,k]=m_k$, $Q\succeq0$) linearizes the *double-trace* RHS of the *Euclidean* Schwinger–Dyson
  equations, but the QM stationarity $\langle[H,\mathrm{Tr}\,w]\rangle=0$ is *single-trace* (no
  double-trace products), so a first-row-anchored product matrix is **vacuous** here — it adds no
  constraint on the energy, exactly as in M5b's `bootstrap_single_matrix_qm`. (Consequence: the L=4
  certified lower bound is **loose** — a commuting moment set with $m[[\tilde X,\tilde Y]^2]=0$ is
  feasible, giving the trivial free floor $2m$; tightening needs higher $L$. See the result doc.)
  (v) **Gram positivity** $M[u,v]=m[\mathrm{reverse}(u)\cdot v]\succeq0$.
- **Output:** minimize $E/N^2$ over the feasible set → certified $E_{\rm lo}\le E/N^2$. (Minimizing a
  relaxation gives a lower bound; the *upper* bound must come from a genuine trial state — §C2/§C3.)
- **Reuse:** `bootstrap_sdp` machinery = M3/M4 two-matrix word handling ⊕ M5b QM momenta + Gauss
  law; trusted-solver hygiene (MOSEK→CLARABEL, `static_regularization`, certified-status gate) as in
  M3/M5b.

### C2 — Gaussian master field: rigorous upper bound (the de-risking baseline)
**An explicit, normalizable trial state — the bound does NOT route through the $\Phi^*$ identity**
(that identity is itself T4; using it here would be circular). Take $|\psi_G\rangle$ = the exact
ground state of a *trial quadratic* Hamiltonian
$H_0=\mathrm{Tr}\big(P_X^2+P_Y^2+\Omega^2(X^2+Y^2)\big)$ ($\Omega$ = variational frequency; an
$X{\leftrightarrow}Y$-symmetric covariance more generally). $|\psi_G\rangle$ is a genuine normalized
state, so by the variational principle (independent of T4)
$$E_{\rm hi}(\Omega)=\langle\psi_G|H|\psi_G\rangle\ \ge\ E_{\rm ground}\quad\text{for \emph{every} }\Omega.$$
- **Direct Wick evaluation.** Gaussian two-point functions
  $\langle X_{ij}X_{kl}\rangle=\tfrac{1}{2\Omega}\delta_{il}\delta_{jk}$,
  $\langle P_{ij}P_{kl}\rangle=\tfrac{\Omega}{2}\delta_{il}\delta_{jk}$ (saturating
  $\langle X^2\rangle\langle P^2\rangle=\tfrac14$, the imprint of $[X,P]=i$); the quartic
  $\langle\mathrm{Tr}[X,Y]^2\rangle$ factorizes by Wick into a closed-form polynomial in $1/\Omega$
  (leading planar contractions). $E_{\rm hi}=\min_\Omega\langle H\rangle$ is a one-parameter
  (Hartree / Gaussian-effective-potential) minimization.
- **Robust to T1.** C2 is a variational computation on the *bare* Eq 17, so it is independent of the
  SDP moment-normalization; $E/N^2$ finite requires $\lambda=Ng^2$ fixed — a useful independent
  cross-check of the T1 scaling. Anchor: $\min_\Omega N^2(\Omega+m^2/\Omega)=2mN^2$ at $\Omega=m$
  ⟹ $E/N^2=2m$ (the true $\lambda{=}0$ ground state). **This locks `E_lo ≤ E/N² ≤ E_hi`
  independently of the research-grade C3.**
- The Gaussian's exact linear-conjugate $\Phi^*$ is retained only as a **cross-check of T4**, never
  as the basis of the bound.

### C3 — Free-Fisher operator master field: the novel core (sharp estimate)
**Kinetic energy as free Fisher information.** Proposed identity (T4):
$$m[\tilde P_X^2]+m[\tilde P_Y^2]=\tfrac14\,\Phi^*(\tilde X,\tilde Y),$$
where $\Phi^*$ is Voiculescu's **free Fisher information** of the joint non-commutative
distribution. Rationale: for a fixed position-distribution, the minimal kinetic energy subject to
$[X,P]=i$ is a Fisher-information functional, whose large-$N$ (free) version is $\Phi^*$. So
$\tfrac14\Phi^*+\text{potential}$ is the energy of the genuine state that is optimal-in-$P$ for that
position-distribution.

**Verified anchors of the identity (analytic — auditor can check):**
- Single variable, density $\sigma$: $\Phi^*(\sigma)=\tfrac{4\pi^2}{3}\!\int\sigma^3$ (Voiculescu).
  Semicircle on $[-2,2]$, $p=\tfrac{1}{2\pi}\sqrt{4-x^2}$: $\int_{-2}^{2}(4-x^2)^{3/2}dx=6\pi$ ⟹
  $\int p^3=\tfrac{3}{4\pi^2}$ ⟹ $\Phi^*=1$. ✓
- **1-matrix reduction to M5b:** $\tfrac14\Phi^*=\int\tfrac{\pi^2}{3}\sigma^3$ = M5b's collective
  kinetic term *exactly*. ✓
- **Anchor ($\lambda{=}0$, $m{=}1$):** M5b $g{=}0$ density $\sigma=\tfrac1\pi\sqrt{2-y^2}$ gives
  $\int\tfrac{\pi^2}{3}\sigma^3=\tfrac12=m[\tilde P^2]$ per matrix; joint free additivity
  $\Phi^*(\tilde X,\tilde Y)=\Phi^*(\tilde X)+\Phi^*(\tilde Y)=4$, so $\tfrac14\Phi^*=1=$ total
  kinetic energy and $E/N^2=1+1=2$. ✓ (All three consistent with the boxed anchor.)

**Computing $\Phi^*$ for truncated multi-matrix moments.** The conjugate variables (free score)
$\xi_X,\xi_Y$ solve Voiculescu's relation, for all words $w$,
$$\tau(\xi_X\,w)=\tau\!\otimes\!\tau(\partial_X w),$$
with $\partial_X$ the **free difference quotient** ($\partial_X(A_1\cdots A_k)=\sum_{i:A_i=X}A_1\cdots A_{i-1}\otimes A_{i+1}\cdots A_k$). Representing $\xi_X$ in a polynomial basis up to degree $L$ turns
this into a **linear solve** for the $\xi$-coefficients against the moment Gram matrix; then
$\Phi^*=\tau(\xi_X^2)+\tau(\xi_Y^2)$ is a quadratic form in those coefficients. The trial moments
come from **Cuntz–Fock operators** $\hat{\tilde X},\hat{\tilde Y}$ (ML-optimized, **positivity
automatic**, exactly as M1–M4); the whole pipeline is differentiable in the operator parameters, so
$\tfrac14\Phi^*+\text{potential}$ is minimized by the existing optimizer (`sparse_fock`/`ansatz`).

**Honest rigor note (must appear in the result writeup).** $\Phi^*$ has a *supremum* (variational)
characterization $\Phi^*=\sup_h\{2\,\tau\!\otimes\!\tau(\partial h)-\tau(h^2)\}$, so a **truncated**
conjugate-variable basis yields a *lower* bound on $\Phi^*$ — hence $E_{\rm MF}$ approaches the true
upper bound **from below** as the basis grows. Therefore $E_{\rm MF}$ is the **sharp master-field
estimate**, certified-bracketed by $[E_{\rm lo},E_{\rm hi}]$ and refereed by HHK — *not itself* the
rigorous cap (that role is C2's). This is stated plainly, in keeping with the project's honesty bar
(no overclaiming the operator field as a certified bound when truncation makes it one-sided). It is
reported **only** with the V6 convergence-and-residual diagnostics — *never* on bracket inclusion
alone (cf. the M3/M4 truncation-artifact lesson, where a moment sat inside a loose island but
*outside* the tight one).

---

## D. Modules & validation

| Module | Role |
|---|---|
| `bootstrap_sdp.py` (extend) | `bootstrap_two_matrix_qm(m, lam, L, *, target, maximize, with_status)` — words in $\tilde X,\tilde Y,\tilde P_X,\tilde P_Y$; stationarity (T5) + two-matrix Gauss law (T2) + Gram PSD; certified bound on $E/N^2$ (no product matrix — single-trace EOM, see C1(iv)). |
| `qm_master_field.py` (new) | `free_fisher_information(moments)` (conjugate-variable linear solve + free difference quotient); `gaussian_master_field(m, lam)` (closed-form rigorous upper bound); `fisher_master_field(m, lam, ansatz)` (minimize $\tfrac14\Phi^*+V$ over Cuntz–Fock ops; reuse `sparse_fock`, `ansatz`). |
| `train.py` (extend) | `solve_two_matrix_qm(m, lam, …)` — SDP lower + Gaussian upper + free-Fisher estimate; fail-closed gate `_tm_qm_gate`. |
| `derivations/m5c-two-matrix-qm.md` (new) | Discharge T1–T5, each **verified numerically** against the $g{=}0$ exact moments and the M5b 1-matrix limit. |
| `tests/` | `test_qm_master_field.py`, `test_bootstrap_two_matrix_qm.py`, `test_train_two_matrix_qm.py`. |

**Validation obligations (proof, computation, or citation for each):**

| # | Claim | Check |
|---|---|---|
| V1 | $g{=}0$ anchor $E/N^2=2m$, $m[\tilde X^2]=\tfrac1{2m}$ | analytic (boxed above) + all three pieces reproduce it numerically |
| V2 | $\tfrac14\Phi^*\to$ M5b $\int\tfrac{\pi^2}{3}\sigma^3$ in 1-matrix limit | reduction shown; numeric on the M5b densities + semicircle $\Phi^*{=}1$ |
| V3 | stationarity (T5) & Gauss (T2) relations | residual $\sim0$ on the $g{=}0$ exact moments before SDP use |
| V4 | SDP brackets the truth | $E_{\rm lo}\le E/N^2$ certified; tightens with $L$; brackets HHK Fig. 3 (T3) |
| V5 | Gaussian is a rigorous upper bound | explicit $\langle\psi_G(\Omega)|H|\psi_G(\Omega)\rangle$ by Wick $\ge$ truth for **all** $\Omega$; $\min_\Omega=2m$ at $\lambda{=}0$ (at $\Omega{=}m$) — **not** via the $\Phi^*$ identity |
| V6 | the sandwich + a *converged* (not merely bracketed) $E_{\rm MF}$ | bracket inclusion $E_{\rm lo}\le E_{\rm MF}\le E_{\rm hi}$ is **necessary, not sufficient**; also require (a) $E_{\rm MF}$ monotone-convergent vs Fisher basis degree (plateau, since it rises from below); (b) conjugate-variable residuals $\sim0$ + reported Gram conditioning; (c) exact match to the $\lambda{=}0$ anchor and agreement with HHK (T3) within an explicit tolerance → `validated` |

Verification medium: Python + `pytest` (limits, residuals, convergence), as M1–M5b.

---

## Risks / open questions
- **R1 — free-Fisher $\Phi^*$ rigor & convergence (main research risk).** Truncated $\Phi^*$ is a
  one-sided (lower) estimate; the conjugate-variable solve may be ill-conditioned for poor trial
  states. *Mitigation:* the Gaussian cap (C2) + certified SDP (C1) + HHK referee bracket $E_{\rm MF}$
  regardless; report convergence vs basis degree.
- **R2 — SDP conditioning.** Mixed $\tilde X,\tilde Y,\tilde P$ words + reorderings + factorization +
  Gauss law make this the most intricate bootstrap yet. *Mitigation:* derive+verify every relation on
  the $g{=}0$ moments first; reuse M3/M5b solver hygiene; start at $L{=}3$ (HHK order) and grow.
- **R3 — referee from a figure (T3).** HHK numbers live in Fig. 3. *Mitigation:* cross-check the
  extracted $\lambda{=}0$ point against the exact $2m$; use HHK's Born–Oppenheimer bounds as a sanity
  window. **Resolved scope (post-audit):** the **hard** validation gate is the $g{=}0$ anchor + the
  internal certified SDP↔Gaussian bracket + the $E_{\rm MF}$ convergence diagnostics (V6); matching
  HHK at $g{>}0$ (T3) is an in-scope **soft cross-check** — reported with agreement tolerance *if*
  Fig. 3 is digitizable, but `validated` does not hinge on it. (Confirm with the user if a stronger
  HHK-matching obligation is wanted.)

## Out of scope (candidate follow-ons)
- **BFSS/BMN** (the next milestone after M5 — multiple matrices, supersymmetry).
- **Amortization** $\hat M(m,\lambda)$ across the coupling plane (à la M4 `AmortizedKZ`).
- **Spectral observables** (joint density / Brown measure of the master field).

---

## Revision log

**2026-06-25 — Codex adversarial review (`verdict: needs-attention`), three findings, all addressed:**
1. *[high] Derivations claimed but absent; coefficients pinned before T1.* The obligations header now
   states T1–T5 are **open deliverables, not done**, with a **hard gate** forbidding frozen SDP
   coefficients / APIs before T1, T2, T4, T5 pass their checks. The 't Hooft scaling and the
   normalized-moment energy density are marked **PROVISIONAL** (form fixed, normalization pending T1).
   The $g{=}0$ anchor $E/N^2{=}2m$ is clarified as **convention-independent** (physical ground energy),
   so it is not provisional.
2. *[high] Gaussian cap asserted without a variational construction.* C2 rewritten around an
   **explicit, normalizable Gaussian trial state** $|\psi_G(\Omega)\rangle$ with $\langle H\rangle$
   computed **directly by Wick** (two-point functions, factorized quartic) — a rigorous bound by the
   variational principle, **not** via the $\Phi^*$ identity (which is circular, since it is T4). The
   bound is kept rigorous (not downgraded to heuristic); the $\Phi^*$ link is demoted to a T4
   cross-check. V5 updated accordingly.
3. *[medium] V6 could bless a biased free-Fisher estimate by loose-bracket inclusion.* V6 strengthened:
   bracket inclusion is **necessary, not sufficient**; `validated` additionally requires monotone
   convergence vs Fisher basis degree, conjugate-variable residuals + Gram conditioning, and explicit
   anchor/HHK tolerances. C3's rigor note cross-references this (and the M3/M4 truncation-artifact
   lesson). Also resolved R3/T3: HHK $g{>}0$ matching is an in-scope **soft cross-check**, not a hard gate.
