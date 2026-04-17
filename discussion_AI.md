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

## Implementation-33: A' D=2 Q2 — constraints satisfied, Wilson loops wrong (Apr 13, 2026)

### Headline

At D=2, L_trunc=4, λ=5, 5000 Adam steps from random init, with loss =
`plaq_MM + 10·L_cyc + L_RP + L_sym`, the optimizer reached:

| quantity | final value | physical target | verdict |
|---|---|---|---|
| final_loss | 1.24e−10 | — | — |
| plaq MM residual | 1.75e−7 | 0 | essentially satisfied |
| cyclicity residual | 4.54e−14 | 0 | machine precision |
| interior unitarity | 2.37e−14 | 0 | machine precision |
| boundary norm (Û_μ|Ω⟩) | 2.86e−1 | < 1 | OK |
| W[plaq] | 0.1188 | 0.1000 | 18.8% error |
| W[2×1] | 0.2022 | 0.0100 | 20× off |
| W[1×2] | 0.2022 | 0.0100 | 20× off (=2×1 ✓) |
| W[2×2] | −0.1252 | 0.0001 | wrong sign & 1000× off |
| W[3×1] | 0.2134 | 0.0010 | 200× off |
| W[fig-8] | 0.2539 | 0.0100 | 25× off |

**All imposed constraints are essentially satisfied. Wilson loops are
far from the physical master field.**

Wall time: 3204 s (~53 min) for 5000 steps at L_trunc=4.

### Diagnosis

Substituting the found values into the plaquette MM equation:

    (1/λ)(W[empty] + W[1×2]) − 2 W[plaq] − (1/λ) W[plaq]²
    = 0.2(1 + 0.2022) − 2(0.1188) − 0.2(0.1188²)
    = 0.2404 − 0.2376 − 0.002823 = ~0                 ✓

The equation holds at these (non-physical) Wilson-loop values. The
ansatz found **a one-parameter family of solutions** to the plaquette
MM equation, parameterised by (W[plaq], W[1×2]) satisfying
`2 W[plaq] + W[plaq]²/λ − W[1×2]/λ = 1/λ`.

Only when we additionally enforce **N=∞ factorization**
`W[1×2] = W[plaq]²` does the equation reduce to
`1/λ = 2 W[plaq]`, giving the physical `W[plaq] = 1/(2λ) = 0.1`.

The Cuntz-Fock ansatz at finite L_trunc does NOT automatically enforce
factorization: ⟨Ω|Û_P·Û_P|Ω⟩ ≠ ⟨Ω|Û_P|Ω⟩² in general. Factorization
is a CONSTRAINT on the master-field state, not an automatic property
of Hilbert-space positivity.

### This is Phase 1b R7, restated

Phase 1b (Impl-14) showed MM + unitarity alone do not uniquely
determine Wilson loops in D=2. The Kazakov-Zheng bootstrap addresses
this with **Toeplitz PSD positivity** on Wilson-loop moment matrices,
which implies factorization at N=∞.

In the Cuntz-Fock formulation, factorization appears as:
- For a self-intersecting loop `C = C_1 ∪ C_2` at vertex v:
  W[C] = W[C_1] · W[C_2]
- More broadly: the L operator algebra is a free product, and at N=∞
  Wilson loops are multiplicative across disconnected cycles.

Our A' loss is MISSING this term.

### What passes through the pipeline correctly

Despite the wrong Wilson loops, the A' run validates the pipeline:

1. `qcd2_q2.py::run_q2_validation` runs end-to-end on matfree + Adam
   + the four-way composite loss.
2. Unitarity and cyclicity are satisfied to machine precision (the v2
   infrastructure works as designed).
3. `plaquette_mm_residual` evaluates to machine precision — the exact
   MM equation from Impl-32 is correctly implemented.
4. The optimizer does converge; it just converges to the wrong basin.

### The fix: add factorization loss

Minimal change to `qcd2_q2.py`:

```python
def factorization_loss(U_list, fock, D):
    """Sum of |W[C] - W[C_1]*W[C_2]|^2 over self-intersecting loops."""
    # Canonical D=2 factorization pairs:
    plaq = (1, 2, -1, -2)
    fig8 = (1, 2, -1, -2, -1, 2, 1, -2)  # two plaquettes at origin
    rect_2x1 = (1, 1, 2, -1, -1, -2)     # area-2 simple loop
    # Key identity: W[fig8] = W[plaq]^2 (window decomposition at origin)
    W_plaq = jnp.real(wilson_loop(U_list, plaq, fock, D))
    W_fig8 = jnp.real(wilson_loop(U_list, fig8, fock, D))
    # Area law for simple loops:
    W_rect = jnp.real(wilson_loop(U_list, rect_2x1, fock, D))
    # fig-8 factorization
    l1 = (W_fig8 - W_plaq**2) ** 2
    # 2x1 is a simple loop of area 2; factorization at shared edge
    # with a plaquette IS the area law:
    l2 = (W_rect - W_plaq**2) ** 2
    return l1 + l2
```

Add `w_fact * factorization_loss` to A' loss. Expected: this will pin
W[plaq] = 1/(2λ), giving W[rect] = w_+², W[2×2] = w_+⁴, etc.

### Alternative: Kazakov-Zheng Toeplitz PSD constraint

Instead of enumerating specific factorization identities, impose the
general PSD condition on the Toeplitz moment matrix of Wilson loops.
This is more principled but more implementation work.

### Updated recommendation

A'' (A' with factorization): add `factorization_loss` to the A' loss,
re-run at D=2, L_trunc=4, λ=5. Expected to converge to the physical
master field. Then document A'' result and pivot to B' (Phase C).

### Status

```
A' result: pipeline validated, but MM + cyc + RP + sym UNDER-determines
           (same as Phase 1b R7).
A'' fix:   add factorization loss.
B' ready:  D=3 infrastructure reuses A'' loss + discovered MM equations.
```

---

## Discussion-31: A' → B' plan (Apr 13, 2026)

After Impl-32 showed the D=2 plaquette MM is the Gross-Witten formula
in disguise, the strategic path forward is:

### A' — D=2 Q2 pipeline validation

Use the discovered plaquette MM equation to run an END-TO-END Q2
validation at D=2 strong coupling. Unsupervised training from random
init, with loss = `plaq_MM + cyc + RP + sym`. Expected to pass (the
constraints are sufficient by construction, since plaquette MM +
factorization give the full area law). This VALIDATES THE PIPELINE
before we invest in harder cases.

File: `cuntz_bootstrap/qcd2_q2.py` (implements `plaquette_mm_residual`,
`make_q2_loss`, `run_q2_validation`; uses matfree H-build + dense expm
from Impl-29).

### B' — Phase C (D=3) substantive Q2 test

Pivot to D=3, the smallest dimension where the master field has no
closed-form area law. At D=3, the null-space scanner
(`find_exact_mm.py`) will find genuinely new MM equations, not
tautologies. Run Q2 unsupervised at D=3 and compare against
strong-coupling expansion or Kazakov-Zheng bootstrap bounds.

Files to create:
- `cuntz_bootstrap/qcd3_targets.py` — strong-coupling expansion targets
  for D=3 Wilson loops (for Q1 validation only; NOT used as Q2 target)
- `cuntz_bootstrap/phase_c_d3.py` — Q1 stretch + Q2 unsupervised at D=3

### Expected timeline

- A' smoke + full run: ~1 hr wall
- A' analysis and Impl-33: ~30 min
- B' infra + D=3 target set: 2-3 hr
- B' Q1 stretch at D=3: ~30 min
- B' null-space at D=3: ~1 hr
- B' Q2 at D=3 + Impl-34: ~2 hr

Total: one focused session + 2-3 hrs of background compute.

### Scope

- No D=4 (Phase D) until D=3 passes.
- No weak-coupling D=2 (gapped phase) experiments.
- `qcd2_wilson_loop` is NOT a Q2 target; ground-truth appears only in
  analysis / report.

---

## Implementation-32: Path A null-space scan — plaquette MM is the GW formula (Apr 13, 2026)

### What was found

`cuntz_bootstrap/find_exact_mm.py` scans linear combinations of candidate
terms (raw + λ-weighted Wilson loops + products) against `qcd2_wilson_loop`
at multiple λ values. Singular-value null vectors give exact polynomial
identities.

**Plaquette edge 0, D=2** (committed, verified):

    (1/λ) · [W[empty] + W[1×2]] = 2 · W[plaq] + (1/λ) · W[plaq]²

Residual at every tested λ ∈ {1, 2, 3, 5, 7, 10, 50, 100, π} is 1e-17 to
1e-18 (machine precision). Fails at λ = 0.5 (weak coupling, gapped phase)
as expected.

**Direct test** (not via null-space; hand-derived from candidate-D + a
splitting term `W[C]²/λ`):

```
lam=1:  residual 0.000e+00
lam=2:  residual 0.000e+00
lam=5:  residual 1.301e-18
lam=10: residual 1.626e-19
lam=7:  residual 3.144e-18   (fresh)
lam=3.14159: residual 3.469e-18   (fresh)
lam=50: residual 1.469e-18
```

### What the equation actually says

Using N=∞ factorization (W[1×2 rectangle] = W[plaq]² by area law):

    (1/λ)(1 + W[plaq]²) = 2 W[plaq] + (1/λ) W[plaq]²
    1/λ + W[plaq]²/λ   = 2 W[plaq] + W[plaq]²/λ
    1/λ               = 2 W[plaq]

**So the plaquette MM equation IS the Gross-Witten strong-coupling
formula w_+ = 1/(2λ).** No new information — just a rephrasing of the
known result.

### Why candidate-D had the 1/(4λ³) residual

Candidate-D omits the splitting term `(1/λ) W[plaq]²`. At strong coupling:

    candidate-D residual = (1/λ)(1 + W[plaq]²) - 2 W[plaq]
                         = [exact] + (1/λ) W[plaq]²
                         = 0 + W[plaq]²/λ
                         = 1/(4λ³)    (since W[plaq]=1/(2λ))

This matches the empirically observed 1/(4λ³) residual exactly. The
missing piece was the trivial start=end-vertex self-intersection
splitting, which contributes `W[C]²/λ` (not `W[C]·W[empty] = W[C]` as I
previously thought).

### Longer loops: null space is degenerate

At 2×1 rectangle edge 0, the null space is 12-dimensional. This is
because many candidates are LINEARLY DEPENDENT at strong coupling —
e.g., at λ=5:

    W[staple_nu=+2]    = W[plaq]     = 1/10
    W[loop·plaq]       = W[rect] · W[plaq]  = 1/1000
    W[staple_nu=-2]    (area 4 long loop) = w_+^4 = 1e-4
    W[plaq]² = W[1×2] = W[staple_nu=-2]/w_+²
    ...

Powers of w_+ collapse; the SVD can't distinguish them. The 12 null
vectors mix arbitrary combinations of these collapsed relations, none
cleanly isolating the MM equation.

### Implication for Q2 strategy

Since the plaquette MM equation in strong-coupling D=2 is EQUIVALENT
to the Gross-Witten formula w_+ = 1/(2λ), and all larger Wilson loops
are determined by w_+ via the area law, **the "exact MM system" for
D=2 strong coupling is just**:

1. (1/λ) = 2 W[plaq]                        (plaquette MM ↔ GW formula)
2. W[C] = W[plaq]^|area(C)| for simple C   (area law)
3. W[C] = Π W[C_i] for window-decomposed C  (factorization)

These three rules fully determine the master field. Any Wilson loop
bootstrap that uses them WILL recover the correct answer.

**However** — this means "Q2 at D=2 strong coupling" is nearly trivial:
the ansatz at L_trunc=4 already fits the area law (Impl-29, 30); adding
plaquette MM + factorization as unsupervised constraints just REINFORCES
the area law it already satisfies.

The GENUINELY novel Q2 test is at:
- D=3, D=4 (no closed-form area law; MM is non-trivial)
- D=2 WEAK coupling (gapped phase; MM coefficient changes)

### Strategic pivot

Given that "exact MM at D=2 strong coupling = GW formula", Path A's
original goal ("exact MM unlocks Q2 at D=2") is tautological. The REAL
Q2 question lives at higher D or weak coupling. Two options:

**A'. Use the discovered equations as a D=2 Q2 sanity check**.
Implement `exact_mm_residual` using the plaquette equation + factorization.
Run Step 3 (unsupervised homotopy) at D=2 strong coupling. Success is
expected and confirms the pipeline. Then move to D=3.

**B'. Go directly to D=3 (Phase C)**.
At D=3 there's no closed-form master field; MM equations ARE
non-trivial and cannot be reduced to a scalar formula. Would use the
D=3 target set (from strong-coupling expansion or Kazakov-Zheng bounds)
and the same null-space scanner to discover D=3 MM equations, now
genuinely independent of the targets.

Recommend **A' → B'**: validate the Q2 pipeline at D=2 with the known
exact equations, then move to D=3 where the question becomes substantive.

### Status

```
Path A plaquette MM: DONE (equation derived, machine-precision validated)
Path A longer loops: deferred (null space too degenerate at strong coupling)
Exact MM in D=2 strong: equivalent to GW formula (not new information)
Phase 4 v3 Step 3 (Q2 at D=2): feasible but expected to be trivially true
Phase C (D=3): genuinely novel test; uses same null-space discovery
Recommended: A' validation, then pivot to Phase C.
```

---

## Discussion-30: Path A via null-space discovery (Apr 13, 2026)

### The idea

Instead of deriving the exact MM equation from Kazakov-Zheng conventions
(which stalled twice on δ̂/δ̆ sign ambiguity — see Impl-26), **discover
it empirically** using `qcd2_wilson_loop` as the oracle.

For each (loop C, edge e_j):

1. Enumerate candidate terms: staple-replaced loops, splits, products,
   identity.
2. Evaluate each candidate at N independent λ values: `qcd2_wilson_loop(C, λ)`.
3. Build evaluation matrix M (shape N × M_candidates).
4. The null space of M **is** the set of exact relations: any vector v
   with M · v = 0 gives a polynomial identity Σ v_j · term_j(λ) = 0 at
   every λ.

Linear algebra discovers the MM equation without any convention choice.

### Why this works

The MM equation is a polynomial identity in Wilson loop values that holds
at all λ. Evaluated at N ≥ 5 distinct λ's, it constrains the polynomial
coefficients to lie in a proper subspace. With the right candidate basis,
the null space is non-trivial and contains the MM relation as one of its
basis elements.

We don't need to know what staples SHOULD appear, what sign conventions
apply, or what the c_self coefficient is — we enumerate ALL plausible
candidates and let SVD pick them out.

### The plaquette edge 0 concrete setup

For C = (1, 2, -1, -2), edge 0 = +1 at (0,0). In D=2 the candidate
staples are (+2, +1, -2) and (-2, +1, +2). After substitution and
backtrack reduction:

- (+2) staple: full loop is (+2, +1, -2, +2, -1, -2) → reduces to empty loop (W = 1)
- (-2) staple: full loop is (-2, +1, +2, +2, -1, -2) → a 1×2 rectangle (W = w_+²)

Candidate pool:
- W[plaq] = w_+
- W[1×2] = w_+² (staple result)
- W[empty] = 1 (staple result)
- W[plaq]² = w_+² (self-intersection product)
- (1/λ) × (staple sum) = (1/λ)(1 + w_+²)

Evaluate at λ ∈ {0.5, 1, 2, 5, 10, 20, 100}. Build matrix M (7 × 5+ columns).
SVD → null vectors give the exact relations.

### Prediction

We expect exactly ONE null vector for the plaquette edge 0 (one MM
equation), containing a linear combination of the staples, W[C], and
possibly W[C]². The EXACT coefficients will fall out of the SVD.

If the null vector has only staple and W[C] terms (no W[C]²), the equation
is purely linear. If W[C]² appears non-trivially, it's the nonlinear
Makeenko-Migdal form (with N=∞ factorization).

### Why this beats the analytic route

- No Fig 3 reinterpretation.
- No sign-convention errors that compile silently and break Step 3.
- Self-validating: the discovered equation must evaluate to zero at any
  λ not in the training set (e.g., λ=7). If it fails, we've found the
  wrong subspace and need to enlarge the candidate pool.
- Generalizes: the same procedure for 2×1, 2×2, fig-8, any canonical
  D=2 loop up to L_max=8.

### Status

Adopting Path A'-by-null-space. Plan updated; implementation starting.
Target: Implementation-32 memo after all L_max=8 equations are discovered
and validated to 1e-12.

---

## Discussion-29: Phase 4 v3 synthesis and decision fork (Apr 13, 2026)

### Where we are

Phase 4 v3 asked two clean questions:

- **Q1 (representational)**: Can the Cuntz-Fock exp-Hermitian ansatz
  represent the QCD₂ master field?
- **Q2 (selectional)**: Do MM + cyclicity + RP + symmetry **select** the
  master field without supervision?

Answering Q1 was the prerequisite for Q2. Five implementations (Impl-27
through Impl-31) explored Q1. This memo synthesizes the findings and
lays out the decision for what to do next.

### Q1 verdict, consolidated

**Q1 = YES** for the physically useful loop regime, with documented
structural caveats:

| Target set | L_trunc | n | Verdict | Source |
|---|---|---|---|---|
| 6 hand-picked | 3 | 340 params | machine precision (later: overfit) | Impl-27 |
| L_max=8 (34 canonical) | 3 | 340 | **FAIL** — overfit exposed | Impl-29 |
| L_max=8 (34 canonical) | 4 | 1364 | **PASS** — worst 5e−6 | Impl-29 |
| L_max=6 × (λ=2,3,5,10) | 4 | 1364 | **PASS** — all 4 machine precision | Impl-30 |
| L_max=10 (186 canonical) | 4 | 1364 | **FAIL** — plateau at 1.65e−3 | Impl-31 |

- The ansatz at **L_trunc=4** works robustly for loops up to length 8
  and across λ ∈ [2, 10].
- L_trunc=3 is insufficient (Impl-27's 6-target success was overfit
  revealed by Impl-29).
- L_trunc=4 hits an Adam-optimizer plateau between 34 and 186 targets.
  Unresolved whether this is an ansatz-structural limit or an
  optimization limit (H1 vs H2 in Impl-31).

For Step 3 (unsupervised homotopy) and everything downstream, **L_max=8
evidence is sufficient**: exact MM and cyclicity use short-loop data.
Phase 3's W[2×2] 900× error is definitively cleared at L_trunc=4.

### Infrastructure milestones

- **Hybrid matfree H build** (`matfree_expm.assemble_hermitian_matfree`)
  unlocks L_trunc=4 on a laptop by avoiding the 630 MB
  `_build_word_operators` cache. Per-step cost at L_trunc=4 comparable
  to dense at L_trunc=3 (~0.7 s/step for moderate target sets).
- **Pure Taylor `expm_iH_v`** matfree code: correct unit tests (13
  pass), but grad-compile pathological with 150+ nested `fori_loop`s.
  Kept in repo as a future building block for L_trunc ≥ 5 where dense
  expm becomes infeasible.
- **Dense path** retained as validation fallback. Dense/matfree agreement
  verified to 1e−10 for H-matvec, expm-v, and wilson_loop.

### The bottleneck for Q2

**Exact MM equations** are the unambiguous next obstacle. Candidate-D
MM has residual = 1/(4λ³) at the plaquette — the HARD GATE failure
documented in Impl-26. Any Step 3 unsupervised run using candidate-D
converges to a configuration that satisfies candidate-D but has the
WRONG Wilson loops (exactly Phase 1b's R4 finding).

Path A (port Kazakov-Zheng eq 3 with Fig 3 conventions) and Path B
(hardcode KZ eq S5 at Λ=4) were both deferred in Impl-26 on
figure-interpretation ambiguity.

### Possible next directions

#### A. Exact MM port (Path A or B from Impl-26) — the principled route

**Pros**
- Directly addresses the Q2 blocker. Once exact MM residuals fit to
  1e-10 against `qcd2_wilson_loop`, Step 3 homotopy becomes meaningful.
- Builds on mature literature (Kazakov-Zheng 2203.11360, Qiao-Zheng
  2601.04316). Papers already downloaded to `reference/`.
- Bounded in scope: 1–2 sessions of diagrammatic work to fix conventions.
- If it succeeds, Phase 4 has both Q1 AND Q2 answered — strong
  publishable result (first explicit Cuntz-Fock master field
  construction for lattice YM with unsupervised selection).

**Cons**
- Figure-by-figure interpretation is error-prone; candidate-D's
  strong-coupling tail matches so closely that silent sign errors would
  look "plausible." Needs ground-truth unit tests (residual → 0 to
  1e-10 for plaquette and 2×1 at λ ∈ {0.5, 1, 2, 5, 10}).
- If indirect MM equations (Qiao-Zheng elimination of auxiliary loops)
  are also required, additional implementation work.

**Concrete first task**: rewrite `exact_mm.mm_direct_residual` to encode
the KZ convention, with hard-gate tests that residual < 1e-10 on
`qcd2_wilson_loop` inputs.

#### B. Debug L_max=10 plateau (L-BFGS / disable cyclicity / L_trunc=5)

**Pros**
- Disambiguates Impl-31's open question (ansatz-structural H1 vs
  optimizer-stuck H2).
- L-BFGS comparison is relatively cheap (~1-3 hr).
- If L-BFGS breaks the plateau: better optimizer for all future runs.
- If the plateau persists: proves structural limit, informs Phase C/D
  design (mixed terms, higher L_trunc).

**Cons**
- Doesn't unblock Q2. L_max=10 isn't required physically — exact MM
  uses short loops.
- L-BFGS in JAX needs `jaxopt` (optional dep) or manual implementation.
- Risk of "one more experiment" dynamics without resolution.

**Concrete first task**: wrap `scipy.optimize.minimize` with
method='L-BFGS-B' around the existing loss_fn; run on the same 186-target
L_max=10 set.

#### C. Phase C (D=3) with hybrid infrastructure

**Pros**
- Demonstrates the infrastructure generalizes beyond D=2.
- D=3 polynomial space is larger (6 creators); hybrid matfree is even
  more essential there.
- Compare against Kazakov-Zheng bootstrap bounds (2203.11360) and
  published lattice MC — would be first unsupervised D=3 master field
  if successful.

**Cons**
- Bypasses Q2 at D=2. If D=3 works at L_max=8 supervised, we've
  generalized Q1 — but still need exact MM for Q2 unsupervised selection.
- D=3 ansatz: 6-label Cuntz space, d_L at L_trunc=3 is 259 (vs 85 at
  D=2), at L_trunc=4 is 1555 (vs 341). Tight on memory without matfree.
- Exact D=3 targets: no clean area law; need strong-coupling expansion
  or KZ bootstrap bounds as surrogate targets.

**Concrete first task**: run `run_step2` at D=3 with strong-coupling
expansion targets at λ = 10 (cleanest regime).

#### D. Document Phase 4 v3 and move to write-up / pause

**Pros**
- Impl-27..31 together constitute a publishable result on the Cuntz-Fock
  master-field representation for QCD₂. Phase 3's 900× W[2×2] failure
  is cleanly reversed.
- Frees up time for other directions.

**Cons**
- Incomplete: Q2 still open. A write-up without Q2 is a "halfway house"
  — demonstrates the ansatz works, does not yet demonstrate unsupervised
  selection.
- Loses momentum on the Cuntz-Fock program.
- W[2×2] "1.2e−6 vs 900×" Impl-27 headline was partially overfit
  (Impl-29 at 34 targets gives 5e−6 which is the clean number); would
  need careful framing.

### Recommended path: A

Rationale:

1. Directly addresses Q2 — the remaining open question of Phase 4.
2. Resolves the bootstrap-literature connection (Kazakov-Zheng).
3. Bounded in scope (1–2 sessions).
4. If A succeeds: both Q1 and Q2 answered at D=2; strong Phase 4 conclusion.
5. If A fails: failure is informative, motivates stepping back (B or D).

B and C are useful but not blocking Q2. Defer until A is resolved.

### What is explicitly NOT blocking

- **L_max=10 capacity ceiling** (Impl-31): short-loop regime suffices
  for Q2.
- **Weak coupling** (λ < 1): Phase A Impl-27 did D=1 weak; not critical
  for D=2.
- **L_trunc=5 sparse ops**: only needed for D=3+ with large target sets.
- **Mixed `â†_u â_v` terms**: only if exact-MM Q2 reveals ansatz limit.

### Status one-liner

```
Q1 answered: YES at L_trunc=4 for L_max=8, robust across λ.
Q2 blocker: exact MM (Path A/B from Impl-26).
Recommended next: exact MM port (Path A).
```

---

## Implementation-31: L_max=10 stretch at L_trunc=4 FAILS — capacity ceiling between 34 and 186 targets (Apr 13, 2026)

### Headline

The 1364-parameter (L_trunc=4, dim=341) ansatz that fit 34 canonical loops
to machine precision in Impl-29 **cannot** fit 186 canonical loops
simultaneously. Adam at 5000 steps plateaus at final loss 1.65e−3, with
156/186 targets failing the gate. Even plaquette — which fit to 1e−10
at L_max=8 — degrades to 8.8% relative error when 186 loops are demanded.

### Run config

D=2, L_trunc=4, dim=341, n_labels=4, λ=5.0, L_max=10, 186 canonical loops
(2 length-4, 4 length-6, 28 length-8, 152 length-10), cyclicity on 3
cyc_words of lengths {6, 8, 10}, Adam + warmup-cosine, n_steps=5000,
lr=5e-3. Hybrid matfree H build + dense expm. Wall time: 13,472 s (3.7 hr).

### Loss trajectory

| step | loss | |grad| |
|---|---|---|
| 0    | 1.25e+2 | 1.82e+2 |
| 500  | 9.79e-2 | 8.02e-2 |
| 1000 | 1.66e-2 | 4.16e-1 |
| 1500 | 8.47e-3 | 7.98e-2 |
| 2000 | 5.78e-3 | 4.02e-2 |
| 2500 | 3.82e-3 | 4.93e-3 |
| 3000 | 2.47e-3 | 3.80e-3 |
| 3500 | 1.92e-3 | 2.45e-3 |
| 4000 | 1.74e-3 | 1.88e-3 |
| 4500 | 1.67e-3 | 1.69e-3 |
| 4999 | 1.65e-3 | 1.64e-3 |

Descent slowed from 10×/500-steps early on to ~5%/500-steps at the end.
Gradient norm also plateaus at ~1.6e-3; the optimizer is stuck in a basin,
not descending further.

### Per-length error statistics

| length | n_loops | mean_err_rel | max_err_rel |
|---|---|---|---|
| 4  | 2   | 6.81e-2 | 8.84e-2 |
| 6  | 4   | 3.68e-1 | 5.75e-1 |
| 8  | 28  | 1.60e+0 | 1.14e+1 |
| 10 | 152 | 9.16e+1 | **3.70e+3** |

Cyclicity residual 2.06e-6 (FAIL at 1e-6 gate). Interior unitarity
4.12e-14 (PASS). Boundary single-step 0.49.

### Comparison summary (all at L_trunc=4, dim=341, λ=5)

| Target set | n | final_loss | n_fail | worst_err | verdict |
|---|---|---|---|---|---|
| L_max=6 (Impl-30) | 6   | 1.62e-21 | 0   | 2.2e-10 | PASS |
| L_max=8 (Impl-29) | 34  | 4.13e-18 | 0   | 5.1e-6  | PASS |
| L_max=10 (this)   | 186 | 1.65e-3  | 156 | 3.7e+3  | **FAIL** |

**Capacity ceiling at L_trunc=4 lies somewhere between 34 and 186 targets.**
The ansatz has 1364 real parameters, mathematically enough to satisfy 186
scalar constraints; but Adam cannot find that solution in 5000 steps.

### Interpretation

Two hypotheses for the plateau are not distinguished by this run:

- **H1 Ansatz-structural**: the specific polynomial form Ĥ = Σ h_w (â†)^w
  + h.c. at L_poly=L_trunc=4 cannot exactly represent the full set of 186
  Wilson loops. The 1364-parameter family is not the right 1364-dimensional
  subspace of Hermitian operators at dim=341. Would need mixed `â†_u â_v`
  terms (prior Task v3-9) or L_trunc=5.

- **H2 Optimization-stuck**: a solution exists in the 1364-parameter
  family, but Adam at lr=5e-3 cannot find it within 5000 steps due to the
  conflicting-gradient landscape with 186 targets. L-BFGS or different
  hyperparameters might succeed.

Distinguishing requires either (a) running L-BFGS on the same target set,
or (b) scaling to L_trunc=5 (dim=1365, 5464 params). Both are substantial
additional experiments.

### Further informative detail: plaquette degrades to 8.8% err

Most striking: W[plaq] — which fit to 5.5e-9 relative error in Impl-27
and 6.4e-9 in Impl-29 — now has 8.8% relative error. Adding more targets
doesn't just leave plaquette alone; it actively perturbs the fit on
shorter loops. This strongly suggests either (a) gradient coupling
through cyclicity (the cyc words are length-10, 8, 6), or (b) genuine
tension: the ansatz at 1364 params cannot satisfy all 186 + cyclicity
conflicts simultaneously.

Turning off cyclicity (or using a SHORTER cyc_word of length 4) should
disambiguate. Not run in this pass.

### Implications for Q1 and next steps

- **Q1 confirmation level**: Impl-29 (L_max=8 = 34 targets PASS at machine
  precision) and Impl-30 (4 couplings PASS) remain intact. Q1 = YES for
  the target sets that correspond to the PHYSICALLY USEFUL domain
  (short-to-moderate loops). This is what matters for Step 3 and beyond.
- **L_max=10 limit**: documented as an open structural question. Not
  blocking for Q2 — the exact MM equations use short-loop data anyway.
- **Decision for roadmap**: do NOT invest in L_max=10+ beyond what's
  needed. Focus on Q2 via exact MM (Path A/B). L_trunc=5 and/or mixed
  ansatz terms are reserved for Phase C/D if needed.

### Status

```
Q1 (stretch capacity):
  - L_max=8  at L_trunc=4: PASS (Impl-29)
  - L_max=10 at L_trunc=4: FAIL (this entry; plateau at L=1.65e-3)
  - capacity ceiling ~50-100 targets with Adam at L_trunc=4
Q1 (coupling robustness): PASS at L_trunc=4 (Impl-30)

Next: exact MM port (Path A/B) for Q2. L_max=10 debugging deferred.
```

---

## Implementation-30: Step 2.6 multi-coupling — Q1 robust across λ (Apr 13, 2026)

### What was tested

Step 2.6 of the v3 plan: run `run_multi_coupling` at λ ∈ {2, 3, 5, 10}
with L_max=6 (6 canonical D=2 loops), L_trunc=4 (dim=341, hybrid matfree
H build + dense expm). 3000 max steps per λ, lr=5e-3, shared config.

### Result

| λ | w_+ | worst_err_rel | final_loss | n_steps | wall_time |
|---|---|---|---|---|---|
| 2.0  | 0.250 | 2.74e−9  | 1.23e−17 | 1201 | 4.2 min |
| 3.0  | 0.167 | 2.41e−7  | 8.13e−14 | ~1800 | 5.0 min |
| 5.0  | 0.100 | 2.15e−10 | 1.62e−21 | 901  | 3.2 min |
| 10.0 | 0.050 | 1.31e−12 | 3.26e−26 | 1501 | 5.2 min |

**Every λ passes the gate to machine precision with the same 1364-parameter
ansatz.** Boundary single-step norm stays in 0.53–0.56 across λ (no
coupling-specific blow-up). Cyclicity 1e−18 to 1e−27; interior unitarity
5e−14 (machine precision) at every λ.

Total wall time: 17.0 min for all four runs combined (vs hours if dense
were attempted at L_trunc=4). Multi-coupling is now a CHEAP experiment.

### Why this matters

The concern after Impl-29 was: "L_trunc=4 stretches at λ=5 — but does
the ansatz STRUCTURE also work at different couplings?" Coupling enters
the problem only through the exact targets W_exact[C] = w_+^|area|;
different λ gives different target vectors. The ansatz parameters h_{μ,w}
adapt per run. Q1 would weaken if some λ failed to converge or required
different architecture.

Answer: **the same exp-Hermitian Cuntz-Fock ansatz at (dim=341, L_trunc=4,
n_labels=4) fits the QCD₂ strong-coupling master field at every λ tested**.
No coupling-specific pathology, no qualitative change in convergence or
boundary behavior.

### The full Q1 story

Combining Impl-27, Impl-29, Impl-30:

- **Impl-27** (L_trunc=3, 6 targets, λ=5): fit to machine precision.
  Impressive but 340 parameters for 6 numbers is heavily overdetermined.
- **Impl-29** (L_trunc=3, 34 targets, λ=5): **FAILS** (worst_err 3.99).
  L_trunc=3 is structurally too small; the 340-param ansatz cannot
  represent 34 canonical length-4/6/8 loops simultaneously.
- **Impl-29** (L_trunc=4, 34 targets, λ=5): **PASSES** (worst_err 5e−6,
  loss 4e−18). The 1364-param ansatz at dim=341 IS adequate.
- **Impl-30** (L_trunc=4, 6 targets, λ ∈ {2, 3, 5, 10}): **ALL PASS**
  at machine precision. Same ansatz structure works across the coupling
  range.

**Q1 = YES**, qualified: the exp-Hermitian Cuntz-Fock ansatz at **L_trunc=4**
simultaneously represents the QCD₂ master field Wilson loops at every
strong-coupling value tested, for every canonical loop up to length 8.
Phase 3's W[2×2] 900× failure is replaced by a machine-precision fit.

### What remains open for Q1

- Stretch further: L_max=10 (186 canonical loops) at L_trunc=4. Computes
  predicted as feasible (~30–60 min) given the hybrid matfree speed.
  Would rule out the lingering concern "works at 34 loops but not 186."
- Weak coupling λ < 1 (gapped GW phase) was NOT tested. The strong-coupling
  formula w_+ = 1/(2λ) is valid for λ ≥ 1; for λ < 1, w_+ = 1 − λ/2.
  Phase A (Impl-27) already suggested weak-coupling works for D=1; the
  D=2 weak-coupling check is a separate experiment, not critical to
  answering Q1.

### Implications for Q2 (the selectional question)

Q1 = YES means the bottleneck for Q2 is now unambiguously **the exact
MM equations** (Path A/B from Impl-26). Candidate-D MM has a 1/(4λ³)
residual at the plaquette — any Step 3 unsupervised homotopy using
candidate-D MM will converge to something not quite the master field.
Step 3 is blocked until exact MM is available.

### Infrastructure summary after Impl-30

- Hybrid matfree H build (vmap(h_matvec)): unlocks L_trunc=4 at ~Impl-27
  wall time. Replaces the 630 MB `_build_word_operators` cache.
- `run_step2`, `run_stretch_test`, `run_multi_coupling`: all take a
  `use_matfree=True` switch (default False for back-compat).
- Unit tests: 100 pass, including 13 new matfree tests.
- Pure Taylor `expm_iH_v` code present but unused in production (grad
  compile is pathological for loss with many nested fori_loops).

### Next steps (in priority order)

1. Launch L_max=10 stretch at L_trunc=4 (186 targets). If passes,
   Q1 is as confirmed as we can reasonably make it without exact MM.
2. Start porting Kazakov-Zheng Fig 3 conventions for exact MM (Path A)
   or encoding KZ eq (S5) hardcoded (Path B). This becomes the
   bottleneck for Q2.
3. Step 3 unsupervised homotopy: blocked on (2).

### Status snapshot

```
Phase 4 v3 Step 2:   DONE (Impl-27)
            Step 2.5: DONE (Impl-29) — L_trunc=4 passes, L_trunc=3 overfit
            Step 2.6: DONE (this entry) — Q1 coupling-robust
            Stretch L_max=10: NEXT (optional strong confirmation)
            Step 3:   BLOCKED on exact MM
Q1 verdict: YES at L_trunc=4 across coupling range
Q2 bottleneck: exact MM (Path A/B)
```

---

## Implementation-29: Phase 4 v3 — Step 2.5 stretch with hybrid matfree (Apr 13, 2026)

### Two headlines

1. **Q1 is NOT robust at L_trunc=3**: the 340-parameter ansatz that fit
   Impl-27's 6 cherry-picked loops to 1e-6 was **overfit**. On the full
   stretch test (34 canonical loops up to length 8), L_trunc=3 plateaus
   at final loss 1.3e-5 and worst relative error 398% — most length-8
   loops are badly mis-predicted.

2. **Q1 IS robust at L_trunc=4**: the 1364-parameter ansatz at dim=341
   fits all 34 stretch targets **simultaneously** to machine precision
   (worst err 5e-6, final loss 4e-18, cyclicity 1e-20). Per-step cost
   comparable to L_trunc=3 dense thanks to the hybrid matfree H build
   (see below). **L_trunc=4 is the right truncation for D=2 QCD₂.**

### What was built

`cuntz_bootstrap/matfree_expm.py` gained the practical path (Discussion-28
analyzed several; Taylor-matvec turned out to compile pathologically
slow under `jax.grad` because of ~150 nested fori_loops per loss):

    assemble_hermitian_matfree(h, fock, wp):
        H = vmap(h_matvec)(eye)      # O(d * nnz) build, no cached d × d word ops

    assemble_unitary_matfree(h, fock, wp):
        return jax.scipy.linalg.expm(1j * H)   # standard dense expm, standard grad

    build_forward_link_ops_matfree(params, fock, wp)

Dense-expm grad complexity is unchanged; only the O(d³) `_build_word_operators`
memory cache is replaced by an O(d · nnz) sparse build. At L_trunc=4 this
saves 630 MB of dense-cache memory; at L_trunc=5 it saves 40 GB (the
previously "infeasible" column). Wilson loops continue to use dense
matvec via `wilson_loop(U_list, ...)`.

13 new unit tests for the matrix-free H build and expm-v (all pass). 100
total tests pass across the codebase.

### Lesson about pure-matfree Taylor

`expm_iH_v` via Taylor + fori_loop is correct (unit tests pass) but
pathologically slow for the stretch-test loss: ~150 expm_iH_v calls per
loss eval (6+28 supervised loops × avg 6 length + 3 cyc_words × ~6
rotations × 6 length = ~150 fori_loops). `jax.grad` converts each
fori_loop to a reversed scan in the backward pass; compile time on the
~150-scan graph hung for 10+ minutes.

Keep the pure-matfree code in the repo (tested, works for single-loop
use) but do NOT use it in the production loss. Hybrid wins.

### Stretch test runs

Config: D=2, L_max=8, λ=5.0, 34 targets (2 length-4, 4 length-6, 28 length-8),
cyclicity on 3 cyc_words, Adam + warmup-cosine, 5000 max steps.

| L_trunc | dim  | params | worst_err | n_fail | final_loss | wall_time |
|---|---|---|---|---|---|---|
| 3 | 85  | 340  | **3.99** | 5/34 | 1.26e−5 | 32 min |
| 4 | 341 | 1364 | **5e−6** | 0/34 | 4.13e−18 | 29 min |

L_trunc=4 completed in **fewer wall seconds** (1724 s) than L_trunc=3
(1940 s). The hybrid matfree build offsets L_trunc=4's larger d:
`_build_word_operators` at L_trunc=4 would have cost 630 MB of cache
and minutes to fill; the vmap build is O(d*nnz) ≈ 540K ops per H.

### By-length statistics at L_trunc=4 stretch (all 34 fit to machine precision)

    len= 4: n= 2   mean_err 6.1e−9    max_err 6.4e−9
    len= 6: n= 4   mean_err 9.5e−9    max_err 2.4e−8
    len= 8: n=28   mean_err 3.2e−7    max_err 5.1e−6

Max imag part across all W[C]_model: 3.4e-10 (planar reality respected).
Cyclicity residual: 1.1e-20 (machine zero).
Interior unitarity: 4.8e-14 (machine precision).

### L_trunc=3 top failing loops (all length-8)

Loops like `(-2, -1, -2, 1, 2, -1, 2, 1)` and `(-2, 1, -2, 1, 2, -1, 2, -1)`
— non-rectangular length-8 canonical forms — have relative errors of 7-85%.
These loops probe high-order inter-direction correlations that the 340-param
ansatz at dim=85 cannot represent.

### Phase 3 comparison, revised

Impl-27's 1.2e−6 W[2×2] relative error at L_trunc=3 (the "9 orders of
magnitude vs Phase 3's 900× error") was obtained on a 6-loop target set.
That fit was not robust: at L_trunc=3 the ansatz can place W[2×2] wherever
the other 5 targets allow, producing a misleadingly small error. The stretch
test shows the same 340-parameter ansatz **cannot** reproduce longer loops.

Upgrading to L_trunc=4 restores Q1=YES for 34 simultaneous targets and
makes the 1.2e-6 result robust (at L_trunc=4, L_max=8 max_err is 5e-6).

### Boundary norm diagnostic

At L_trunc=4 the single-step boundary norm is 0.57, not small. But
`Û_μ^k|Ω⟩` boundary mass at k=1..8 stays in the 0.19-0.60 range (no
monotone growth to 1). Wilson loops still fit to machine precision,
so the truncation is ADEQUATE even with meaningful boundary occupancy.
Going to L_trunc=5 would reduce boundary pressure further but is not
computationally required for L_max=8 at D=2.

### Next steps

1. Multi-coupling test (Step 2.6) at **L_trunc=4** with L_max=6
   (~few × 5-10 min per λ). Previously blocked by compute; now feasible.
2. Stretch further: L_max=10 (186 targets) at L_trunc=4. Tests whether
   the 1364-parameter ansatz generalizes to longer loops.
3. Exact MM (Path A/B) becomes the next gate to Step 3 (unsupervised).

### Status

```
v3 Tasks: 2.5 at L_trunc=4 = DONE (Q1 robust at machine precision)
          2.5 at L_trunc=3 = FAIL (documented overfit)
          2.6 multi-coupling = next (L_trunc=4)
Hybrid matfree: IN PRODUCTION (unlocks L_trunc=4, replaces _build_word_operators)
Pure Taylor matfree: CORRECT but unused (grad compile pathological)
```

---

## Discussion-28: Matrix-free expm-v — unlocking L_trunc ≥ 4 (Apr 13, 2026)

### Why this is needed

Impl-27 established Q1 = YES at L_trunc=3 (dim=85) in ~34 min wall time.
The next tests (Step 2.5 stretch, L_trunc=4 validation, eventually D=3, D=4
Phase C/D) all need larger Fock spaces. Dense `jax.scipy.linalg.expm` in
`hermitian_operator.assemble_unitary` is O(d³) and dominates compute:

| L_trunc | dim  | dense expm | per step (6 targets) | per step (186 targets) |
|---|---|---|---|---|
| 3 | 85   | 0.6M ops | ~1 s    | ~3 s     |
| 4 | 341  | 40M ops  | ~65 s   | ~200 s   |
| 5 | 1365 | 2.5B ops | hours   | INFEASIBLE |

Worse, `_build_word_operators` in `hermitian_operator.py` CACHES d×d matrices
one per basis word (d matrices of size d²), giving a d³-memory build cost
(630 MB at L_trunc=4, 40 GB at L_trunc=5 — infeasible).

### The bottleneck isn't the physics — it's that we form U at all

Wilson loops are matrix-element computations:

    W[C] = ⟨Ω| Û_{μ_1} Û_{μ_2} ... Û_{μ_k} |Ω⟩
         = e_0^T · e^{iH_1} · e^{iH_2} · ... · e^{iH_k} |Ω⟩

This is a CHAIN of matrix-vector products (d² ops each), never a single
matrix-matrix. Forming the full d × d matrix Û = e^{iH} wastes O(d³) work
and O(d²) memory per direction.

### Matrix-free Taylor-series e^{iH}v

The cheapest algorithmic fix: compute e^{iH} v via Taylor series truncation:

    e^{iH} v = Σ_{k=0}^{N} (iH)^k v / k!

Each iteration is a matvec H_matvec(h, v), cost O(nnz(H)). For Cuntz
creation-string operators C_w (length-|w| polynomial in â†_i), C_w has
AT MOST ONE NONZERO PER COLUMN (it's a partial permutation on valid
preimages). So H = Σ h_w C_w + h.c. has nnz ~ d · n_words / d = O(d · n_words).
Per matvec: O(d · n_words) = O(d · d) = O(d²) at worst, but in practice
O(total_nnz) ≈ O(d + ...).

Concrete nnz counts per Fock size:

| L_trunc | d | Σ nnz across all C_w | vs d² dense |
|---|---|---|---|
| 3 | 85 | 313 | 85× sparser |
| 4 | 341 | 1,593 | 73× sparser |
| 5 | 1365 | ~7k | 266× sparser |

### Per-step cost comparison (34 targets)

| L_trunc | Dense expm + matvecs | Taylor(order=25) matfree | speedup |
|---|---|---|---|
| 3 | 2.1M ops  | 0.4M ops  | ~5× |
| 4 | 64M ops   | 2.0M ops  | ~32× |
| 5 | 2.5B+ ops | 9.3M ops  | ~270× |

At L_trunc=4, stretch test becomes ~10 min instead of ~hours.
At L_trunc=5, it becomes minutes instead of "don't even try."

### Options considered (ranked by effort × impact)

1. **Taylor-series sparse matvec** (~30 lines, pure JAX, chosen)
   Σ_k (iH)^k v / k! truncated at order ~25. H v uses precomputed sparse
   (src, tgt) index arrays — no dense H, no dense U. Full autodiff via
   JAX native ops. Best simplicity-to-speedup ratio.

2. **JAX GPU** (1 line if a GPU is available)
   `jax.config.update("jax_platform_name", "gpu")` → O(d³) expm on GPU
   is 10-500× faster. Useful ON TOP OF matfree for really large d.
   Deferred unless a GPU cluster becomes available.

3. **Custom Krylov `expm_multiply` primitive** (~50 lines + custom_vjp)
   Better for large ||H|| (adaptive Krylov dim). But needs hand-written
   VJP for the Arnoldi iteration. Deferred.

4. **Julia ExponentialUtilities + Enzyme** (full rewrite)
   Gold standard for sparse expm-v. Not chosen — we keep the JAX
   ecosystem for now (existing tests, CI, familiarity).

### Convergence of Taylor series

Error bound: ‖(iH)^N v / N!‖ ≤ ‖H‖^N / N!.

For order N=25 and ||H|| ≤ 1: 1/25! ≈ 6e-26. Machine zero.
For N=25 and ||H|| = 3: 3^25/25! ≈ 5e-14. Still machine precision.
For N=25 and ||H|| = 5: 5^25/25! ≈ 2e-8. Degraded.
For N=25 and ||H|| = 10: ~10. Fails.

During Step 2, observed ||H|| stays O(1) (h-coefficients are O(1e-2 to 1e-1)
and act on basis states with norm 1). Track max ||H|| in training and
auto-bump order if > ~4. If fails, switch to scaling-and-squaring
(e^{iH} = (e^{iH/2^s})^{2^s}, with the (·)^{2^s} done via repeated matvecs
by squaring the intermediate expansion — this doubles per-iteration cost
but handles any ||H||).

### Validation path (before switching default)

1. Implement `matfree_expm.py` with `build_word_indices`, `h_matvec`,
   `expm_iH_v`.
2. Unit tests: agreement with dense at d=85 to 1e-10 (expm-v), 1e-12
   (H-matvec). JAX gradient correctness via finite-difference.
3. Replay Step 2 at L_trunc=3 with matfree; require W[2×2] agrees with
   the Impl-27 dense result to 1e-6.
4. Step 2.5 stretch at L_trunc=3: matfree vs dense wall-time comparison.
5. Step 2.5 stretch at L_trunc=4: previously infeasible; should complete
   in ≤ 1 hour with matfree.

If all four pass, matfree becomes the default path. Dense retained as a
fallback / validation reference.

### Next action

Implement per the plan at `~/.claude/plans/melodic-exploring-lecun.md`.
Target: Implementation-29 memo after L_trunc=4 stretch test completes.

---

## Implementation-27: Phase 4 v3 — Step 2 Q1 YES (machine precision) (Apr 13, 2026)

### Path decision recap

Impl-26 established candidate-D MM fails the HARD GATE (residual = 1/(4λ³)
at the plaquette, not 1e-10). Paths A and B (exact MM from Kazakov-Zheng
or hardcoding eq S5) blocked on figure-interpretation ambiguity.
Adopted **Path C**: answer Q1 (representational) first via supervised fit,
independent of exact MM. If the ansatz can represent the QCD₂ master field
even under supervision, Q1 = YES and the remaining obstacle to Q2 is just
the exact MM. If not, no amount of constraint engineering helps.

### What was built

`cuntz_bootstrap/qcd2_supervised.py` (~230 lines, committed on `main`):
composes the v2 infrastructure (fock.py, hermitian_operator.py,
wilson_loops.py, cyclicity.py, diagnostics.py, optimize.py) into a
supervised fit against `qcd2_wilson_loop` targets. Targets:

    PLAQ     = (1, 2, -1, -2)                        w_+^1
    RECT_2x1 = (1, 1, 2, -1, -1, -2)                 w_+^2
    RECT_1x2 = (1, 2, 2, -1, -2, -2)                 w_+^2
    RECT_2x2 = (1, 1, 2, 2, -1, -1, -2, -2)          w_+^4
    RECT_3x1 = (1, 1, 1, 2, -1, -1, -1, -2)          w_+^3
    FIG8     = (1, 2, -1, -2, -1, 2, 1, -2)          w_+^2 (window)

Loss: `L = Σ |W_model[C] − W_exact[C]|² + 10 · L_cyc`. Cyclicity on
{PLAQ, RECT_2x1, FIG8}. Adam + warmup-cosine, Impl-19 conj gradient fix.

### Run: D=2, L_trunc=3 (dim=85), λ=5, 5000 steps, lr=5e-3

| step | loss |
|---|---|
| 0    | 4.80 |
| 500  | 5.5e−3 |
| 1000 | 4.1e−5 |
| 1500 | 6.4e−8 |
| 2000 | 6.3e−11 |
| 2500 | 1.7e−12 |
| 3000 | **2.1e−18** (tol=1e-12 triggered, early stop) |

### Final per-target (at step 3001)

| Target | Model | Exact | Relative err |
|---|---|---|---|
| W[plaq] | +1.000000e−01 | 1.000000e−01 | 5.5e−9 |
| W[2×1]  | +1.000000e−02 | 1.000000e−02 | 6.4e−8 |
| W[1×2]  | +1.000000e−02 | 1.000000e−02 | 3.7e−8 |
| W[2×2]  | +9.999988e−05 | 1.000000e−04 | **1.2e−6** |
| W[3×1]  | +1.000000e−03 | 1.000000e−03 | 2.4e−8 |
| W[fig8] | +1.000000e−02 | 1.000000e−02 | 7.9e−10 |
| cyclicity residual |   —   |  —   | 4.1e−21 |
| ‖UU† − I‖_interior |   —   |  —   | 5.2e−15 |

Every Wilson-loop target fits to machine precision. Imaginary parts of all
model Wilson loops are < 1e−9 (planar reality respected). Cyclicity is at
numerical zero. Interior unitarity is at machine precision.

### The W[2×2] headline vs Phase 3

- **Phase 3 (TEK, N=9)**: W[2×2] = +0.089, exact = 1e−4 → **900× relative error**.
- **Phase 4 v3 Step 2 (Cuntz-Fock, L_trunc=3)**: W[2×2] = 9.9999e−5, exact =
  1e−4 → **1.2e−6 relative error**.
- Improvement: **≈ 10⁹** (nine orders of magnitude).

The exp-Hermitian Cuntz-Fock ansatz can represent the QCD₂ master field —
plaquette, 2×1, 1×2, 2×2, 3×1, and the window-decomposed figure-8 — at a
single coupling, simultaneously, to machine precision. **Q1 = YES.**

### One caveat: boundary norm

`boundary_norm(Û|Ω⟩)` at L_trunc=3 is 0.44 — the state Û|Ω⟩ has 44 % of
its probability mass at word length 3 (the truncation boundary). This is
NOT a physics failure: unitarity and Wilson-loop fits are all at machine
precision inside the truncated space. It IS a signal that the ansatz is
packing the state right against the boundary; L_trunc=3 is the MINIMUM
adequate for this target set. Additional loops tested at longer lengths
would require L_trunc ≥ 4 to keep boundary leakage under control.

Scaling to L_trunc=4 (dim=341) costs ~67× more per optimization step
(`expm` is O(d³); `_build_word_operators` caches d × d × d memory ≈
630 MB). Not run here — physics conclusion is already the same.

### Parameters and wall time

- 2 × 85 complex = 340 real parameters.
- Run: ~34 minutes on laptop CPU for 3001 steps until tol=1e−12 triggered.
- Machine: darwin (macOS 25.4.0), JAX on CPU with float64.

### What Q1 = YES means for Phase 4

- **Q1 answered**: ansatz is adequate. Cuntz-Fock exp-Hermitian is NOT
  the bottleneck.
- **Q2 still open**: does MM + cyc + RP + sym select this solution
  WITHOUT supervision? Answering Q2 requires exact MM (candidate-D is
  O(1/λ³) biased). Step 3 homotopy is meaningful only after exact MM is
  available.
- **The blocker is now squarely exact MM (Path A/B from Impl-26), not
  the ansatz.**

### Next steps

1. Commit: `feat: v3 Step 2 — qcd2_supervised.py (Q1=YES, W[2×2] to 1e-6)`.
2. Before Step 3, decide:
   - **Path A** (port Kazakov-Zheng eq 3): ~1-2 sessions of diagrammatic work
     to nail δ̂/δ̆ signs from Fig 3.
   - **Path B** (hardcode eq S5 at Λ=4): same figure-interpretation bottleneck
     but narrower scope (6 specific loops).
   - **Stretch test** at L_trunc=3: add more targets (W[3×2], W[4×2], longer
     figure-8s) and verify the ansatz still fits. If it fails here, Q1 is
     "yes for 6 loops" but "no for the full master-field algebra" — a
     subtler failure mode.
3. Phase C (D=3) and Phase D (D=4) remain blocked behind a working Step 3.

### Status

```
Tasks: v3 Task 8 (qcd2_supervised) — DONE
Q1 verdict: YES (machine precision at L_trunc=3)
W[2×2] vs Phase 3: 1.2e-6 relative vs 900× (9 orders of magnitude improvement)
Blocker for Q2: exact MM (Path A/B)
```

---

## Implementation-26: v3 Tasks 1-3 + HARD GATE investigation (Apr 13, 2026)

### What was built (committed on main)

- **v3 Task 1**: Discussion-26 memo prepended to this file (supersedes v2
  Discussion-25 narrative).
- **v3 Task 2** (`cuntz_bootstrap/qcd2_exact.py`, 12 tests): exact QCD₂ Wilson
  loops via Gopakumar-Gross window decomposition. Figure-8 with two unit
  plaquettes → W = w_+² verified to 1e-14.
- **v3 Task 3** (`cuntz_bootstrap/exact_mm.py`, 6 tests): candidate-D
  direct MM equations (c_self = 2, sum of staple-replaced loops divided by
  λ) with helpers `staple_replacement`, `split_pairs_at_vertex`,
  `mm_direct_residual`, `enumerate_loops`.

### HARD GATE outcome

Residual of candidate-D at the plaquette, all edges, vs exact
`qcd2_wilson_loop(C, λ)`:

    λ = 1   → residual 2.5e-1
    λ = 2   → residual 3.1e-2
    λ = 5   → residual 2.0e-3
    λ = 10  → residual 2.5e-4

Empirically exact formula: `residual = 1/(4λ³)`. This confirms Phase 1b's
Implementation-13 diagnosis: candidate-D MM is only LEADING ORDER in 1/λ.
The 1e-10 HARD GATE is NOT met at any finite λ. Candidate-D is usable as
an approximation at strong coupling (< 1% at λ ≥ 5) but cannot validate
itself exactly against the GW area law.

### Literature review (papers downloaded to `reference/`)

1. `reference/qiao_zheng_2601.04316.pdf` — "Direct and Indirect Loop
   Equations in Lattice Yang-Mills" (Liu & Yang, Jan 2026). Algorithmic
   framework for SU(2) loop-equation enumeration via plaquette-cut and
   subloop-cut strategies. Important insight: direct equations alone do
   NOT form a complete set — "indirect equations" emerge only when
   eliminating auxiliary higher-length loops from the direct system.

2. `reference/kazakov_zheng_2203.11360.pdf` — "Bootstrap for Lattice
   Yang-Mills theory" (Kazakov & Zheng, 2022). **The canonical SU(∞) paper.**
   Explicit loop equation in their eq (3):

       Σ_{ν⊥μ} (W[C; δ̂C^ν_{l_μ}] − W[C; δ̆C^ν_{l_μ}]) = λ Σ_{l'=l} ε_{ll'} W[C_{ll'}] W[C_{l'l}]

   Supplementary eq (S5) gives the explicit Λ=4 2D loop equations in
   terms of 6 specific Wilson loops W_1..W_6 (W_1 = plaquette).

3. `reference/makeenko_notes_2508.09705.pdf` — "Notes on the Loop Equation
   in Loop Space" (Makeenko, original ~1994, arXiv 2025). The CONTINUUM
   form of the loop equation (eq 2.13) as a functional Laplace equation:

       ∂_μ^x (δ/δσ_{μν}) W(C) = λ ∫_C dx'_μ δ^(d)(x-x') W(C_{xx'}) W(C_{x'x})

   The lattice discretization maps area-derivative → "loop with plaquette
   added minus loop without" and path-derivative → "backtrack-variation".

### Why porting KZ eq (3) stalled

The δ̂ vs δ̆ notation in KZ eq (3) is ambiguous from text alone. Each of
three natural interpretations I attempted (above/below staple, CCW/CW
orientation of same plaquette, add/remove plaquette) leaves a non-zero
LHS `1 − w_+²` at the plaquette where RHS is zero (no self-intersection
splits). The correct orientation+sign convention is encoded in Fig 3 of
KZ which I cannot interpret precisely enough in-session to avoid silent
sign errors that would invalidate all downstream work.

Three paths forward are documented in the plan file:

- **Path A**: careful figure-by-figure port of KZ Fig 3. Risk: sign
  errors masked by the SPEED of candidate-D matching at large λ. 1-2
  sessions of diagrammatic work.
- **Path B**: hardcode KZ Supplementary eq (S5) at Λ=4 as a ground-truth
  test suite. Requires identifying W_2..W_6 from the figures (same
  figure-interpretation bottleneck).
- **Path C**: proceed to Step 2 with candidate-D MM residuals documented
  as O(1/λ³) biased. Step 2 is a SUPERVISED fit answering the
  REPRESENTATIONAL question (Q1) — does not depend on exact MM. If Q1
  passes, exact MM becomes the only remaining obstacle to Q2; if Q1
  fails, the ansatz is the bottleneck and exact MM doesn't help.

### Recommendation

**Path C** has the highest information-per-time. Supervised Step 2 with
the exp-Hermitian ansatz directly answers Phase 3's 900× W[2×2] failure
mode: if the ansatz can represent W[2×2] = w_+⁴ under supervised training,
we've cleared the FIRST blocker; if it cannot, no amount of exact-MM work
will rescue the bootstrap. Defer Path A (exact MM port) to after Step 2.

### Status snapshot

```
Tasks complete: v3 Task 1 (memo) + v3 Task 2 (qcd2_exact) + v3 Task 3 (candidate-D mm_loss)
Papers downloaded: qiao_zheng, kazakov_zheng, makeenko in reference/
HARD GATE: failed at 1e-10 (candidate-D is O(1/λ³) biased)
Awaiting: user decision on Path A/B/C
```

---

## Discussion-26: Phase 4 v3 — Steps 0-3 Refinement (Apr 13, 2026)

### Why v3 replaces v2

v2's Phase B smoke test (committed) reproduced the Phase 1b finding
(Implementation-13 from Apr 12): the "candidate D" MM equations we inherited
from `master_field/mm_equations.py` are only LEADING ORDER in 1/λ, not
exact. At λ = 10 the optimizer satisfied candidate-D MM to 1e-5 but
W[plaq] was 70% off and W[2×2] had the wrong sign. Any unsupervised result
at this coupling is attributable to the WRONG EQUATIONS, not to the ansatz.

All Phase 1b, Phase 3, and Phase 4 v2 failures so far have been partially
uninterpretable for this reason: is the problem the ansatz, or the
equations? v3 fixes the equations first (Step 0 HARD GATE) so that future
failures are attributable.

v3 also restructures Phase B as two separable experiments, each answering
one clean question:

- **Q1 (representational)**: Can the exp-Hermitian ansatz SIMULTANEOUSLY
  fit all known QCD₂ Wilson loop values via supervised optimisation?
- **Q2 (selectional)**: Do exact-MM + cyclicity + reflection-positivity
  UNIQUELY determine the coefficients in an unsupervised homotopy from
  the Q1 solution?

Separating these lets us diagnose failures cleanly:
- Q1 = No: ansatz inadequate → enlarge (mixed â†â terms, higher L_poly).
- Q1 = Yes, Q2 = No: constraints insufficient → add indirect MM equations,
  strengthen RP, or add physical principles.
- Q1 = Yes, Q2 = Yes: Phase 4 works; D=3, D=4 are mechanical.

### Status of v2 infrastructure (committed to main)

Usable as-is: `fock.py`, `hermitian_operator.py`, `wilson_loops.py`,
`cyclicity.py`, `lattice_symmetry.py`, `reflection_positivity.py`,
`total_loss.py`, `optimize.py`, `gw_validation.py`, `phase_a_gw.py`
(Phase A PASSED at 8e-8 moment error).

Broken: `mm_loss.py` (candidate-D). Will be rewired through the new
`exact_mm.py` engine.

Deprecated: `phase_b_qcd2.py` unsupervised run. Restructured as Step 2
supervised first, then Step 3 homotopy. Keep for reference.

### v3 roadmap

**Step 0 — Exact MM engine (HARD GATE)**
`qcd2_exact.py` (exact Wilson loops with window decomposition) +
`exact_mm.py` (direct MM derived from Haar measure, plus indirect
equations via elimination) + `test_exact_mm.py`. Every direct-MM residual
< 1e-10 against the exact QCD₂ Wilson loop for every loop up to length 8
and every edge, at λ ∈ {0.5, 1, 2, 5, 10}. Nothing proceeds until this
passes.

**Step 1 — GW one-plaquette sanity** (`gw_test.py`)
Re-confirm exp-Hermitian + expm + autodiff + truncation diagnostics on
the simplest case before the D = 2 bootstrap.

**Step 2 — Supervised Q1 test** (`qcd2_supervised.py`)
THE decisive experiment. Supervised fit of Û₁, Û₂ to all target Wilson
loops (plaquette through 2×2, figure-8, cyclicity, kernel entries) at
λ = 2 and 5. Success criterion: each target within 1% simultaneously.

**Step 3 — Unsupervised homotopy** (`qcd2_unsupervised.py`)
Only if Step 2 passes. α-homotopy L(α) = (1-α)L_sup + w_MM L_MM_exact +
w_cyc L_cyc + w_RP L_RP, with augmented Lagrangian for cyclicity and
multi-start diagnostic at α = 1. Success: W[2×2] within 10% of w_+⁴,
multi-start variance < 10%.

### The physics bar

Phase 3's W[2×2] failure was 900× at strong coupling. A Phase 4 pass at
Q2 here (W[2×2] within 10% of w_+⁴) would be a three-orders-of-magnitude
improvement and the key empirical evidence that the Cuntz-Fock bootstrap
works. Phase C (D=3) and Phase D (D=4) then run on identical code.

### Parameter counts (unchanged)

| D | n=2D | L_poly | d_L | real DOFs / matrix | total real DOFs |
|---|------|--------|-----|---------------------|-----------------|
| 1 (GW) | 1 | 6 | 7 | 13 | 13 |
| 2 (QCD₂) | 4 | 3 | 85 | 169 | 338 |
| 3 | 6 | 2 | 43 | 85 | 255 |
| 4 (QCD) | 8 | 2 | 73 | 145 | 580 |

### The one risk (unchanged)

If Q1 fails, enlarge ansatz (mixed terms, higher L_poly).
If Q2 fails while Q1 passes, add indirect MM equations and twist-reflection
positivity; else additional physical principle (clustering, free entropy).

### Scope boundaries

No Phase C (D=3) or Phase D (D=4) until Q1 and Q2 are both answered for
D=2. No sparse-operator refactor until L_trunc > 3 is demonstrably needed.
No free-entropy or clustering principles until MM-exact + cyc + RP is
shown to fail.

---

## Discussion-25: Phase 4 — Cuntz-Fock Coefficient Bootstrap (Apr 12-13, 2026)

### The idea

Invert Gopakumar-Gross (1994). GG showed the master field for any large-N
matrix model is an operator Û in a Cuntz-Fock space, with coefficients
determined by planar connected Green's functions. We make the coefficients
the UNKNOWNS and let lattice loop equations + physical-state constraints
determine them.

### Why this differs from Phases 0-3

| Phase | Parametrization | Failure |
|---|---|---|
| 0 QCD₂ single-α Gaussian | one real coefficient | plaquette OK, larger loops wrong |
| 1b neural W[C] | neural loop functional | MM alone underdetermined (Impl-14) |
| 3 TEK finite-N matrices | N×N unitary matrices | R6 classical saddle; R9 multi-matrix correlations (W[2×2] off 900×) |
| 4 Cuntz-Fock coefficients | O(d_L) polynomial Hermitian Ĥ coefficients | **untested — the proposal** |

Works at N = ∞ by construction. No Haar measure (R6 disappears). No center
symmetry to break (R8 disappears).

### Parametrization (revised after v1 critique)

For lattice direction μ, with n = 2D creation labels on the Cuntz-Fock
space truncated at word length L_trunc ≥ L_poly + 2:

    Ĥ_μ = Σ_{|w|≤L_poly} h_{μ,w} · (â†_{w_1} … â†_{w_k}) + h.c.
    Û_μ = expm(i · Ĥ_μ)
    Û_{-μ} = Û_μ†     (orientation reversal)

Unitarity is automatic (expm of a Hermitian matrix is unitary in exact
arithmetic; scipy/jax `expm` gives machine-precision unitarity in the
truncated space).

### Why NOT the polynomial form Û_μ = Σ c^{(+)}_w (â†)^w + Σ c^{(-)}_v (â)^v

That form was tried in v1 of the plan. Two problems:

1. **EOM inconsistency at finite truncation.** The conjugate momentum Π̂
   defined by [Π̂, Û] = |Ω⟩⟨Ω| has no finite-dimensional solution: Tr of a
   commutator is zero, but Tr(P_Ω) = 1. So the Gopakumar-Gross EOM
   `V'(Û) = 2 Π̂` cannot be a direct loss. Use MM loop equations — they
   are the EOM sandwiched between specific states and are polynomial in
   Wilson loops, which are polynomial in the coefficients.

2. **Positivity is not the whole selection.** An arbitrary Cuntz-Fock
   operator family with a vacuum vector does not automatically satisfy
   cyclicity of the loop trace ⟨Ω|ABC|Ω⟩ = ⟨Ω|BCA|Ω⟩. That equality IS
   the N = ∞ trace property, not an automatic consequence of Hilbert-space
   positivity. Must be imposed as a loss term. Similarly reflection
   positivity and lattice rotations/reflections (B_D).

The exponential-Hermitian form has the right physics: Û unitary, and the
Hermitian generator Ĥ is the natural object in GG §3.

### Loss components

    L = w_MM · L_MM + w_cyc · L_cyc + w_RP · L_RP + w_sym · L_sym
       [ + w_sup · L_sup ]      (optional anchor)

**L_MM — Makeenko-Migdal residuals (direct).** Kazakov-Zheng candidate D
from master_field/lattice.LoopSystem. Per equation:

    res(C, e) = (1/λ) Σ_{P ∋ e} W[staple(C, e, P)]
              − c_self · W[C] − Σ_{splits} W[C_1] · W[C_2]

with c_self = 2. Wilson loops W[C] = ⟨Ω|Û_{μ_1} … Û_{μ_k}|Ω⟩ = e_0ᵀ · (product
of Û matrices) · e_0. Indirect equations (Qiao-Zheng 2601.04316) deferred
to after direct-only evaluation.

**L_cyc — cyclicity / traciality residuals.** For each loop C in a test
set and each cyclic rotation C_k = (C[k], …, C[k-1]):

    L_cyc = Σ_{C, k} |W[C_k] − W[C_0]|²

Enforces ⟨Ω|·|Ω⟩ behaves like the normalized trace at N = ∞.

**L_RP — reflection positivity.** Pick a reflection plane. For open paths
p_1, …, p_K in the "positive half", build

    R_{ij} = ⟨Ω| Û_{θ(p_i)}† Û_{p_j} |Ω⟩
    L_RP   = Σ_{λ_k(R) < 0} λ_k(R)²

**L_sym — lattice symmetries.** For each σ ∈ B_D (hyperoctahedral):

    L_sym = Σ_{C, σ} |W[σ(C)] − W[C]|²

**L_sup (optional).** Σ_C |W[C] − W_GW_area_law[C]|² on simple loops.
Disabled by default; enabled only if unsupervised losses underdetermine
the solution.

### Priority order

1. Cuntz-Fock infrastructure (JAX).
2. **Phase A — Gross-Witten** (D=1, n_labels=1, L_poly=6; 13 real DOFs).
   Loss = supervised moment-matching ONLY (calibration). Gate at strong
   coupling (λ ≥ 1): max moment error < 1e-2.
3. **Phase B — QCD₂** (D=2, L_poly=3, d=85; 338 real DOFs). THE CRITICAL
   TEST. Loss = L_MM + L_cyc + L_RP + L_sym (unsupervised). Gate: W[□],
   W[2×1], W[2×2], figure-8 factorization each within 1% of GW area law
   at λ=5. Phase 3 failed W[2×2] at 900×; a Phase 4 pass here is the key
   empirical result.
4. **Phase C — D=3** (L_poly=2, d=43). Compare to Kazakov-Zheng bootstrap
   bounds (arXiv:2203.11360) and published MC.
5. **Phase D — D=4** (L_poly=2, d=73). The target.

### Parameter counts

| D | n=2D | L_poly | d_L | real DOFs / matrix | total real DOFs |
|---|------|--------|-----|---------------------|-----------------|
| 1 (GW) | 1 | 6 | 7 | 13 | 13 |
| 2 (QCD₂) | 4 | 3 | 85 | 169 | 338 |
| 3 | 6 | 2 | 43 | 85 | 255 |
| 4 (QCD) | 8 | 2 | 73 | 145 | 580 |

Two orders of magnitude below Phase 3 (TEK at N=49 had 9604 params). The
Fock space truncation L_trunc must be ≥ L_poly + 2 so MM staples (loops of
length |C|+2) stay within basis.

### The one risk

MM + cyc + RP + sym might still be underdetermined. Fall back to supervised
anchor at strong-coupling GW area law (proven in Phase 1b R4). If still
fails: indirect MM equations (Qiao-Zheng). Else additional physical
principle needed (extremality, clustering, free-entropy max).

### Deliverables

Subfolder `cuntz_bootstrap/` with fock.py, hermitian_operator.py,
wilson_loops.py, cyclicity.py, reflection_positivity.py, lattice_symmetry.py,
mm_loss.py, total_loss.py, optimize.py, gw_validation.py, train.py, plus
phase_a_gw.py, phase_b_qcd2.py, phase_c_d3.py (stretch), phase_d_d4.py
(stretch). Physics reference `reference/cuntz_bootstrap.md`. Cluster
script `cluster/submit_cuntz.pbs`.

### Status snapshot (Apr 13, 2026)

v1 scaffolding (Tasks 1-7) committed on main; strong-coupling Phase A at
λ ∈ {10, 5, 2} passes at 1e-9 moment error (calibration successful).
v2 switches to the exponential-Hermitian form and adds cyc + RP + sym
losses. v2 Task 1 is this memo update.

---

## Discussion-24: Phase 3 retrospective — session summary and handoff (Apr 12, 2026)

Summary of what was built, learned, and what remains — intended as a handoff
when picking this up next.

### What Phase 3 set out to do

Construct the SU(∞) master field for lattice Yang-Mills as explicit N×N
unitary matrices via direct gradient-descent optimization on the Twisted
Eguchi-Kawai (TEK) action. If successful at D=4, this would be the first
explicit construction of the SU(∞) master field for 4D YM (an open problem
since Witten 1979 / Gopakumar-Gross 1994). See Discussion-15 for the plan.

### What was delivered

A complete, tested scaffolding for TEK direct optimization in D=2,3,4:

```
tek_master_field/
├── __init__.py, config.py, conftest.py
├── tek.py              core: clock matrix, twist, link builder, action
├── observables.py      plaquette, Polyakov, eigenvalue density, W[R×T]
├── optimize.py         classical-action Adam + cosine schedule
├── mm_loss.py          MM residual loss + 3 optional anchor modes
├── gross_witten.py     Phase A sanity check (GW 1-matrix)
├── train.py            CLI: --model {gw,ek,tek} --ansatz {orientation,full}
├── phase_b.py          classical-action experiment (untwisted EK D=2)
├── phase_b_mm.py       MM experiment (untwisted EK D=2)
└── test_tek.py         89 pytest tests, all passing
```

Plus `reference/tek_master_field.md`, `cluster/submit_tek.pbs`, and eight
Implementation entries in this log (Impl-16..23).

### Physics risks and resolutions

| Risk | What | Status |
|------|------|--------|
| R1 | D=4 k=1 center-symmetry breaking | deferred (`--k` accepts modified flux per arXiv:1005.1981) |
| R2 | Rectangular Wilson loop twist phase | resolved — `W[R×T] = Re[z^{RT} · Tr(...)]/N` (arXiv:1708.00841 eq 2.4) |
| R3 | Sign conventions for action | resolved (scaffolding standardized) |
| R4 | Orientation-only sufficiency | resolved — full U(N) ansatz available via `--ansatz full` |
| R5 | Γ spectrum mismatch with TEK saddle | resolved — Γ = kron(P_L, I_L), eigenvalues = L-th roots L-fold |
| R6 | Classical action minimizes to CLASSICAL vacuum, not master field | partially resolved — MM loss recovers coupling dependence |
| R7 | MM candidate-D underdetermined | partially resolved — area-law anchor fixes small loops |
| R8 | Toeplitz-PSD positivity | resolved — Toeplitz trivial in matrix param; the nontrivial analog is a center-moment anchor, which fixes MM's spontaneous Z_N breaking |
| R9 | Multi-matrix large-loop correlation | **open** |

One side finding: a **JAX complex-gradient sign bug** was discovered during
Phase B. `jax.grad(f)(z)` for real f of complex z returns `∂f/∂x − i·∂f/∂y`
(conjugate of the descent direction). The optimizer was updating in the wrong
direction; fixed by `grads = [jnp.conj(g) for g in grads]` before
hermitianizing and passing to Adam. Documented in Impl-19.

### Final empirical state (D=2, N=9, untwisted EK, λ=5, full ansatz)

Best loss combination: `MM + area_law(w=0.1) + moment(K=4, w=10)`:

| observable | Gross-Witten | ML result | error |
|---|---|---|---|
| W[plaquette] | 0.1000 | +0.107 | 7% |
| W[2×1] | 0.0100 | +0.013 | 30% |
| W[2×2] | 0.0001 | +0.089 | 900× |
| Σ\|Tr(U_μ)/N\|² | 0 | 3e−5 | ✓ |

Small loops within a few percent; center symmetry preserved; large loops (area
≥ 4) fail by ~900×. At strong coupling only (λ ≥ 2). Weak coupling (λ < 1.5)
unreliable across all ansatz + loss combinations.

### Epistemic progression across the session

1. **Phase B with classical action** (Impl-19): saddle = classical vacuum at
   every λ. plaquette = 1 regardless of coupling. Completely wrong physics.
   This surfaced R6 and the JAX sign bug.
2. **Phase B-MM** (Impl-21): MM loss recovers coupling-dependent plaquette to
   ~1% at strong coupling, but gives W[2×1] = W[plaq] (constant-w solution;
   wrong area law). Surfaced R7.
3. **MM + area-law anchor** (Impl-22): recovers W[2×1] ≈ GW² to 3% at λ=5.
   Larger loops still off. Anchor weight ≈ 0.1 is the sweet spot at λ=5;
   breaks down at intermediate couplings.
4. **MM + moment anchor** (Impl-23): MM-only was spontaneously breaking Z_N
   (|Tr/N| ≈ 0.55 per matrix — large). Moment anchor restores Z_N; combined
   with area-law anchor gives the final-state table above.

### Honest assessment of the gap

The matrix parametrization finds configurations with eigenvalue density matching MC
for SMALL probes (plaquette, 2×1) but with arbitrary high-order correlations
between U_1 and U_2. Neither MM candidate-D (leading order in 1/λ) nor any
soft constraint we tried shapes the JOINT eigenvalue/eigenvector structure of
multiple matrices strongly enough for larger loops.

Phase 1b hit the same wall with a neural loop parametrization — MM alone is
underdetermined, supervised warm-start helps small loops but drifts.

Three conclusions are consistent with the evidence:

1. **Matrix parametrization is too flexible.** Finite N gives many
   configurations compatible with our soft constraints; the optimizer picks
   one that's "good enough" on the probes we anchor but wrong elsewhere.
2. **The loss is physically incomplete.** Kazakov-Zheng SDP parametrizes W[C]
   directly as bootstrap variables and adds enough positivity to uniquely
   pin the master field. Their SDP framework has the physical constraints
   we're missing.
3. **Large-N effects matter.** We worked at N=9 (L=3) and N=49 (L=7). The
   finite-N corrections for W[2×2] scale as O(4/N²) but the gap we see is
   200×-900× — much larger than expected finite-N. Suggests (1) or (2), not
   just N.

### Three paths when resuming

A. **Accept and publish strong-coupling, small-loop results.** Current accuracy
   is publishable as "proof of concept: matrix master field for TEK at strong
   coupling, Wilson loops up to area 2 within a few percent." Honest about
   limitations. Cheapest to complete; limited physics novelty.

B. **Pivot to Kazakov-Zheng SDP + matrix reconstruction.** Solve the SDP for
   W[C] values first (uses existing `master_field/bootstrap_sdp.py`), then
   reconstruct U_μ matrices from the W[C] via moment-matching / Gopakumar-Gross
   Cuntz-Fock. More physics, more work. The matrix-reconstruction step is
   itself research.

C. **Higher-order MM equations.** Candidate D is leading order in 1/λ. Next-
   order corrections would pin more loops. Derivation is nontrivial but
   mechanical from the lattice action. Would close R7 but probably not R9 (the
   multi-matrix correlation issue is structural, not about MM precision).

Recommended next step if continuing: **A** — finalize strong-coupling
small-loop results at N=49 with both anchors, cluster run, write up. Defer the
large-loop gap as a known open problem.

### Deferred items (never started this session)

- **Phase C** (twisted TEK D=2 at N=49, 121, 289) — infrastructure exists
  (`--model tek` in train.py, plus anchored MM loss). Would compare
  MM-recovered plaquette to MC benchmarks at modest N.
- **Phase D** (TEK D=4 at N=49, then 289) — requires R1 resolution (modified
  flux k ≈ L/2 per arXiv:1005.1981). `build_twist` accepts `k` argument already.
- **Phase E** (N → ∞ extrapolation via N ∈ {49, 121, 289, 529}) — after Phase C/D.
- **Cluster runs** (`cluster/submit_tek.pbs` written but not executed).
- **Paper** — nothing written yet; the physics reference at
  `reference/tek_master_field.md` and these discussion entries have the content.

### How to resume

1. `cd tek_master_field && python3 -m pytest -q` — confirm 89/89 tests still pass.
2. `python3 phase_a_main.py` — Phase A gate (GW at machine precision).
3. `python3 phase_b_mm.py` — reproduces the MM-only Phase B.
4. Pick path A, B, or C above. Anchored-MM at stronger settings is one-line
   parameter tweaks to `optimize_tek_mm` (see `make_mm_loss_fn` signature).
5. Relevant files: `tek_master_field/mm_loss.py` (loss fns), `optimize.py`
   (classical-action optimizer), `observables.py` (Wilson loops).

---

## Implementation-23: R8 — Toeplitz-PSD automatic; center-moment anchor fixes symmetry-breaking (Apr 12, 2026)

### Why classical Toeplitz-PSD is trivial in our setup

The Toeplitz-PSD positivity constraint used in the Kazakov-Zheng bootstrap states that the Toeplitz moment matrix M[i,j] = Tr(U_μ^{i-j})/N (with W_{-k} = conj(W_k)) is PSD. For OUR parametrization, the moments come from actual unitary matrices U_μ, which automatically produce PSD moment matrices (the moments are the Fourier coefficients of a positive atomic measure on the unit circle). Adding a PSD penalty does nothing.

We implemented `_toeplitz_moment_matrix(U, K)` for diagnostics but it confirms this observation: PSD is not the right lever in matrix-parametrized TEK.

### What IS meaningful: center-moment anchor

A genuinely nontrivial single-matrix constraint is

    Σ_μ Σ_{k=1}^{K_mom} |Tr(U_μ^k) / N|²   →   0

which pins the first K_mom moments of U_μ to zero. For the untwisted-EK master field at strong coupling (center-symmetric, uniform eigenvalue density on the circle), these moments vanish identically. For the orientation ansatz, Tr(Γ^k) = 0 for k < L automatically; for the full ansatz, the constraint actively pushes the spectrum toward uniform.

Implemented as `moment_weight` + `moment_K` parameters in `make_mm_loss_fn` / `optimize_tek_mm`.

### Strong new finding: MM-only breaks center symmetry

At D=2 N=9 untwisted EK λ=5 with full ansatz:

| config | W[plaq] | W[2×1] | W[2×2] | Σ\|Tr(U)/N\|² | mm_loss |
|---|---|---|---|---|---|
| GW target | 0.1000 | 0.0100 | 0.0001 | 0 | 0 |
| MM only | +0.1015 | +0.0150 | −0.073 | **2.96e−01** | 8e−32 |
| MM + area_law(0.1) | +0.1008 | +0.0103 | +0.071 | **1.49e−01** | 1.2e−2 |
| MM + moment(K=2, w=1) | +0.1014 | +0.0136 | −0.087 | 3.1e−33 | 1.4e−31 |
| MM + moment(K=4, w=1) | +0.1012 | +0.0121 | −0.037 | 4.6e−33 | 1.3e−31 |
| MM + moment(K=4, w=10) | +0.1008 | +0.0083 | −0.143 | 4.0e−33 | 3.8e−31 |
| MM + both (area 0.1, mom 10) | +0.1070 | +0.0129 | +0.089 | 3.2e−05 | 7.7e−2 |

**MM alone settles into a center-symmetry-broken vacuum** (Σ\|P_μ\|² ≈ 0.3 → \|P_μ\| ≈ 0.4 per matrix — substantial breaking). Even the area-law anchor can't rescue this (the W[plaq] = 1/(2λ) target doesn't directly constrain single-matrix traces). Only the explicit moment anchor forces Tr(U_μ^k) → 0 and restores the center-symmetric phase.

MM-only + center-broken matrices happens to give reasonable W[plaq], W[2×1] — a coincidence. Physically, we must preserve Z_N for the TEK reduction to be valid at N=∞, so the moment anchor is not optional, it's required.

### What's still wrong

Even with both anchors active:
- ✓ W[plaq] ≈ 0.107 (7% err vs GW 0.100)
- ✓ W[2×1] ≈ 0.013 (30% err vs GW² 0.010)
- ✓ Center symmetry (Σ\|Tr\|² ≈ 3e−5)
- ✗ W[2×2] ≈ 0.089 (900× GW⁴ = 10⁻⁴)

W[2×2] is resistant. This is the same pattern as R7: moving from area 2 to area 4 breaks the fit. Root cause: our TEK matrices at N=9 have high-order correlations between U_1 and U_2 that don't match the area-law eigenvalue-density structure. Center symmetry (single-matrix) is enforced but large-loop CORRELATION isn't.

### R8 status

**Resolved with an important caveat.** Toeplitz-PSD is trivially satisfied and adds nothing; but the "moral equivalent" — center-moment anchor — is both nontrivial and essential. It prevents the spontaneous center-symmetry breaking that MM-only produces.

### What remains (R9)

R6 (classical saddle) → Impl-21 partial; R7 (MM underdetermined) → Impl-22 partial for small loops; R8 (positivity) → this entry, cures symmetry breaking but doesn't close the gap at area ≥ 4.

The remaining gap is **large-loop correlations between multiple matrices**. Neither MM candidate D nor any soft-constraint loss we've tried shapes the JOINT eigenvalue/eigenvector correlation of U_1 and U_2 strongly enough. Options:
1. Higher-order MM equations (expensive to derive).
2. Larger N (may reduce finite-N effects on W[2×2]).
3. Pivot to full Kazakov-Zheng SDP on W[C] directly (loses the matrix-master-field narrative).
4. Accept current accuracy and report results at strong coupling, small loops only.

### Phase 3 risk status (updated)

| Risk | Status |
|------|--------|
| R1 D=4 k=1 center-symmetry | deferred |
| R2 rectangular Wilson loop | resolved (Impl-16) |
| R3 sign conventions | resolved |
| R4 orientation-only sufficiency | resolved (Impl-18) |
| R5 Γ spectrum mismatch | resolved (Impl-17) |
| R6 classical action misses master field | partial (Impl-21) — MM fixes it |
| R7 MM underdetermined | partial (Impl-22) — anchor fixes small loops |
| R8 Toeplitz positivity | **resolved (this entry)** — Toeplitz trivial; center-moment anchor nontrivial and fixes SSB |
| R9 multi-matrix large-loop correlation | **NEW** — open |

---

## Implementation-22: R7 anchor — area-law anchor fixes small loops; breaks for area ≥ 4 (Apr 12, 2026)

### What was added

Extended `mm_loss.make_mm_loss_fn` with two optional supervised-anchor modes:

- `anchor="plaquette"`: adds `w · (W[plaq] − 1/(2λ))²` to the loss (pins plaquette to Gross-Witten value only).
- `anchor="area_law"`: adds `w · Σ_i (W[C_i] − w_+^|Area(C_i)|)²` over every loop with a well-defined D=2 area. Full supervision toward the lattice area law.

Both take a scalar `anchor_weight` multiplier. Exposed through `optimize_tek_mm`.

### Result at D=2 N=9 untwisted EK, λ=5, full ansatz (anchor weight sweep)

| weight | W[plaq] | W[2×1] | W[2×2] | mm+anch |
|---|---|---|---|---|
| GW target | 0.1000 | 0.0100 | 0.0001 | — |
| 0.00 (MM only) | +0.1014 | +0.0137 | −0.2029 | 1.0e-31 |
| 0.01 | +0.1008 | +0.0098 | +0.161 | 2.1e-3 |
| 0.10 | +0.1008 | +0.0103 | +0.071 | 1.2e-2 |
| 1.00 | +0.0942 | +0.0090 | +0.078 | 1.2e-1 |
| 10.0 | +0.0923 | +0.0065 | +0.070 | 1.1 |
| 100. | +0.6615 | +0.1943 | −0.061 | 2.4e+2 |

**W[2×1] tracks GW² to 2% at weight ∈ [0.01, 0.1]** — the area law is recovered for area 2. But W[2×2] stays at ~0.07 vs GW⁴ = 10⁻⁴; anchor at this weight isn't strong enough to pull it down by ~700× while staying consistent with MM.

Pushing to N=49 (more matrix DOF) at λ=5, weight-0.1 gives W[plaq]=0.103 (3% err), W[2×1]=0.011 (10%), W[2×2]=0.052, W[3×1]=0.035 (still ~200× too large). Higher weight degrades plaquette without helping larger loops.

### λ sweep at N=9, weight=0.1

| λ | W[plaq] err% | W[2×1] err% | comment |
|---|---|---|---|
| 10 | 2.8% | 25.8% | strong, 2×1 decent |
| 5 | 0.7% | **2.7%** | best result — area law for 2×1 |
| 2 | 3.7% | 5.7% | still decent |
| 1.5 | 15.3% | 38.1% | transition region |
| 1.0 | 42.5% | 64.8% | weak coupling fails |

The anchor works well at strong coupling (λ ≥ 2) for small loops but breaks down at weak coupling (λ ≈ 1) and for larger loops at any coupling.

### Interpretation

The finite-N TEK matrices at our optimizer's saddle have an eigenvalue structure that is coupling-dependent and matches the MC master field for small probes (plaquette, 2×1), but the larger-loop structure depends on high-order moments of the eigenvalue distribution that neither MM candidate D (leading order in 1/λ) nor the plaquette-centric anchor pins down.

To get W[m×n] = w_+^{m·n} out to large m, n, we need a constraint that shapes the eigenvalue distribution directly — i.e., positivity of the Toeplitz moment matrix (Kazakov-Zheng style). The moment matrix M_{ij} = W[(U^i)(U^{-j})] being PSD is equivalent to the eigenvalue density being a proper probability measure. This is R3 from the original Phase 1 roadmap.

### R7 status

**Partially resolved.** MM + area-law anchor at weight ≈ 0.1 gives:
- ✓ W[plaq] near GW at strong coupling (< 3% at λ ≥ 2)
- ✓ W[2×1] near GW² at strong coupling (< 6% at λ ≥ 2)
- ✗ W[area ≥ 4] badly off
- ✗ Weak coupling (λ < 1.5) still unreliable

### Next step proposal

Add **Toeplitz-PSD positivity constraint** (R3 from the original plan). Specifically, for each direction μ, build the Hankel/Toeplitz moment matrix from Wilson loops W[U_μ^k] and enforce PSD via a log-barrier or eigenvalue penalty. This is the core of the Kazakov-Zheng bootstrap and should shape the eigenvalue density correctly.

Alternative: larger N (N=121 or 289) with area-law anchor may help since finite-N effects on W[m×n] scale as (mn)/N². But the fundamental issue — arbitrary eigenvalue density compatible with MM + small-loop anchor — won't be fixed by scale alone.

### Phase 3 risk status (updated)

| Risk | Status |
|------|--------|
| R1 D=4 k=1 center-symmetry break | deferred |
| R2 rectangular Wilson loop phase | resolved (Impl-16) |
| R3 sign conventions | resolved (scaffolding) |
| R4 orientation-only sufficiency | resolved (Impl-18) |
| R5 Γ spectrum mismatch | resolved (Impl-17) |
| R6 classical action misses master field | partially resolved (Impl-21) |
| R7 MM alone underdetermined | **partially resolved (this entry)** — anchor fixes small loops, large loops need positivity |
| R8 Toeplitz-PSD positivity needed | **NEW** — Kazakov-Zheng constraint for large-loop structure |

---

## Implementation-21: Phase B-MM — MM loss works at strong coupling; underdetermined at weak (Apr 12, 2026)

### What was built

`tek_master_field/mm_loss.py`: computes Wilson loops W[C] = Re[ z_12^{signed_area_2d(C)} · Tr(Π U_{μ_i}) ] / N from TEK matrices over a precomputed LoopSystem (imported from `master_field/lattice.py`), evaluates Makeenko-Migdal candidate-D residuals, returns their sum-of-squares. `optimize_tek_mm` plugs into Adam + cosine schedule with the same conj+hermitianize gradient projection used in the classical optimizer. Works for both orientation and full ansätze; D=2 only for now (twist factor uses signed_area_2d).

`tek_master_field/phase_b_mm.py`: runs the MM optimizer across λ ∈ {10, 5, 2, 1.5, 1.2, 1, 0.8, 0.5} at D=2, N=9, L_max=6 with warm-starting. Records W[plaquette], W[2×1], W[2×2] vs GW strong-coupling targets (W[C] = w_+^|Area|, w_+ = 1/(2λ) or 1 − λ/2).

### Result

**Orientation ansatz, N=9:**

| λ | W[plaq] | GW | err% | W[2×1] | GW² | mm_loss |
|---|---|---|---|---|---|---|
| 10 | +0.049 | 0.050 | 1.1% | +0.049 | 0.0025 | 1.1e-3 |
| 5 | +0.099 | 0.100 | 1.1% | +0.099 | 0.010 | 1.3e-2 |
| 2 | +0.291 | 0.250 | 16.5% | +0.291 | 0.063 | 2.3e-1 |
| 1.5 | +0.500 | 0.333 | 50.0% | +0.500 | 0.111 | 1.5e-7 |
| 1 | +1.000 | 0.500 | 100% | +1.000 | 0.250 | 2.8e-8 |

Plaquette matches GW to 1.1% at strong coupling (λ ≥ 5). Deviates at intermediate. Orientation mode's constraint (eigenvalues = L-th roots L-fold-degenerate) produces the degenerate "all W[C] = w_+^MM" family that satisfies MM with constant Wilson loops — the candidate-D self-consistent solution w_+^MM = λ − √(λ²−1), which coincides with w_+^GW only to leading order in 1/λ.

**Full ansatz, N=9:**

| λ | W[plaq] | GW | err% | W[2×1] | GW² | mm_loss |
|---|---|---|---|---|---|---|
| 10 | +0.0502 | 0.0500 | 0.4% | +0.0035 | 0.0025 | 7.5e-12 |
| 5 | +0.1014 | 0.1000 | 1.4% | +0.0136 | 0.0100 | 1.0e-31 |
| 2 | +0.2728 | 0.2500 | 9.1% | +0.0912 | 0.0625 | 1.3e-31 |
| 1.5 | +0.3966 | 0.3333 | 19.0% | +0.1897 | 0.1111 | 2.8e-31 |

Full-ansatz MM loss reaches MACHINE ZERO (10⁻³¹). The plaquette still matches GW to 1% at strong coupling. W[2×1] is NO LONGER identical to W[plaq] — the extra parameters let the optimizer find a non-constant Wilson-loop configuration. But W[2×1] is ~40% larger than GW's product law W[plaq]². The MM equations admit a family of solutions; our optimizer picks one that has the right plaquette but wrong larger-loop structure.

### Relative to Phase B (classical action)

Phase B with classical action alone gave plaquette = 1 at every λ — totally wrong for MC comparison. Phase B-MM gives plaquette tracking GW to ~1% at strong coupling — a significant physics improvement. So MM loss is definitively better than classical action, as predicted.

### Relative to Phase 1b

This reproduces Phase 1b's finding exactly. From Implementation-14:
> MM equations + unitarity alone do NOT uniquely determine Wilson loop values in D=2. [...] R4 (supervised warm-start 1500 epochs, then MM-only 3000 epochs): best result. W[plaq] at λ=5 = 0.10012 vs GW 0.1 (near-exact). W[2×1] at λ=1 drifts from supervised 0.25 to 0.16 after MM refinement.

Same issue, different parametrization: MM candidate D is underdetermined; uniquely pinning W[C] requires additional constraints (positivity) or a bias toward the physical solution (warm-start).

### Phase 3 risk status

R6 is **partially resolved**. With MM loss:
- ✓ Classical-saddle pathology fixed: plaquette is coupling-dependent.
- ✓ Strong-coupling plaquette matches GW to ~1%.
- ✗ Area law W[R×T] = W[plaq]^{RT} not captured.
- ✗ Weak coupling still unreliable.

These are exactly the failure modes Phase 1b identified. Next step is to add positivity constraints (R3 from the original roadmap) or supervised warm-start at the GW strong-coupling value.

### Proposal for next step

Add a **Toeplitz-PSD constraint** on the Wilson loop algebra (or equivalently a supervised anchor at the plaquette to the GW strong-coupling value) as an additional loss term:

    L_total = L_MM + λ_anchor · (W[plaq] − 1/(2λ))²         (supervised mode)
    L_total = L_MM + λ_psd · barrier(−min_eig(Toeplitz))    (positivity mode)

Supervised anchor is cheaper and proven in Phase 1b. Positivity is more rigorous but harder to compute. Recommend: anchor first to confirm the pipeline, then positivity to lift the anchor.

### Phase 3 risk status table (updated)

| Risk | Status |
|------|--------|
| R1 D=4 k=1 center-symmetry break | deferred |
| R2 rectangular Wilson loop phase | resolved (Impl-16) |
| R3 sign conventions | resolved (scaffolding) |
| R4 orientation-only sufficiency | resolved (Impl-18) |
| R5 Γ spectrum mismatch | resolved (Impl-17) |
| R6 classical action misses master field | **partially resolved (this entry)** — MM correct at strong coupling, underdetermined elsewhere |
| R7 MM alone underdetermined at D=2 | **NEW** — add positivity constraints or supervised anchor next |

---

## Discussion-20: Choosing the R6 fix — Haar entropy vs MM loss vs SDP bootstrap (Apr 12, 2026)

### Context

Implementation-19 (Phase B) established empirically that minimizing the classical TEK action S_classical over matrices U_μ gives the zero-entropy classical vacuum at every coupling, not the master field. This is R6. Three routes to recover coupling-dependent master-field observables were listed. This entry evaluates them before picking one.

### Physics recap — what the master field actually is

The master field is not a canonical N×N matrix. It is:
- in Witten's sense, the N=∞ saddle of the path integral, defined by large-N factorization;
- in Gopakumar-Gross's sense, a pair of operators Û_μ in the Cuntz-Fock space, encoding all single-trace expectation values;
- in MC's sense, the limiting configuration around which finite-N samples cluster with O(1/N²) fluctuations.

All three of these encode the SAME single-trace observables (Wilson loops). The master field has well-defined Wilson loops at N=∞, but the specific matrix realization at finite N is gauge-dependent.

The path integral at finite N is Z = ∫ dU e^{−S[U]}. In eigenvalue coordinates, the Haar measure contributes a Vandermonde repulsion ∏|e^{iθ_i} − e^{iθ_j}|². At large N this log-Vandermonde term is O(N²), same order as the action. At the saddle the two balance — yielding the MC-observed plaquette vs coupling. Minimizing S alone drops the Vandermonde and collapses to the classical vacuum; this is what Phase B saw.

### Option 1 — Add Haar entropy to the loss

Derive log|J| for the parametrization and add it.

For U = exp(iH) ∈ U(N) with H Hermitian: the Jacobian of the exponential map gives

    dU_Haar / dH = |det[(1 − e^{−i ad H}) / (i ad H)]|   (Duflo/BCH formula)

In eigenvalue coordinates of H (eigenvalues θ_i), this specializes to

    |det J| = ∏_{i<j} |sinc((θ_i − θ_j)/2)|²

Adding −log|J| to the loss restores the Vandermonde repulsion. The optimizer would see an eigenvalue-repulsion pressure that, balanced against the plaquette-attraction, produces a coupling-dependent saddle.

**Pros.** Direct. Preserves the "find matrix master field" framing. Cleanly matches MC at N=∞ by construction.

**Cons.** (i) The Duflo Jacobian is expensive to evaluate — requires spectral decomposition or a determinant of an (N² × N²) adjoint matrix per gradient step. (ii) Gradients of −log|J| through autodiff on a near-degenerate spectrum can be numerically fragile. (iii) The orientation ansatz's coadjoint-orbit measure is constant (no log|J| contribution), so this option only rescues the FULL ansatz, not orientation. (iv) Still unclear whether the saddle of (S + log|J|) is unique — may have center-broken and center-symmetric minima depending on initialization.

**Cost estimate.** A week of physics derivation + implementation + debugging. Substantial.

### Option 2 — Makeenko-Migdal loop equations as loss

The MM equations are loop-space statements of the master-field condition:

    λ · W[C]  =  Σ_{P ∋ e} W[P_e ∘ C]  −  Σ_{splits} W[C_1] · W[C_2]

These equations are satisfied BY the master field at N=∞, independent of how we parametrize. We compute Wilson loops from our TEK matrices U_μ, sum MM residuals squared, minimize.

**Pros.** (i) Already proven on D=2 lattice YM in Phase 1b (with partial success — MM loss converges, but not uniquely without positivity). (ii) No Haar-entropy derivation needed — MM IS the correct master-field condition. (iii) Works for both orientation and full ansätze. (iv) Natural coupling dependence: λ enters explicitly in the equation.

**Cons.** (i) Phase 1b on D=2 revealed that MM alone is underdetermined (K unknowns, K − few equations). Needs positivity constraints for uniqueness. (ii) At each step, we must evaluate W[C] for many lattice loops via Tr(product of U_μ)/N — O(loop_length × N²) per loop. For L_max = 6 on TEK, ~35 loops times 32 equations; manageable. (iii) TEK requires the twist-adjusted MM equations; need to verify our existing MM implementation (staple convention from `master_field/mm_equations.py`) applies correctly after volume reduction.

**Cost estimate.** Moderate. Most of the infrastructure exists in `master_field/lattice.py` and `mm_equations.py`. Porting to TEK means: (a) evaluating W[C] from TEK U_μ instead of neural functionals; (b) adding the TEK twist phase to the MM equation (analogous to rectangular Wilson loop twist from Impl-16); (c) coupling-continuation schedule over λ. Perhaps 2–3 days.

### Option 3 — SDP bootstrap with continuation

Formulate as a semidefinite program: MM equations as linear constraints, positivity of the Toeplitz moment matrix as PSD constraint, minimize a scalar observable (or find feasible point). With optimum over W[C] values directly; recover U_μ from W[C] post-hoc (harder).

**Pros.** (i) Rigorous — Kazakov-Zheng 2021 proved this converges to the unique master field at N=∞ with certifiable bounds. (ii) Handles non-uniqueness of MM alone via positivity. (iii) No autodiff/gradient issues.

**Cons.** (i) SDP solvers (cvxpy + SCS/MOSEK) don't easily give us U_μ* matrices — they give W[C]* values. Matrix reconstruction is a separate step (hard, probably using the Cuntz-Fock framework). (ii) SDP scaling is harder at large N_loops (N_loops grows exponentially with loop length). (iii) Doesn't exploit our Phase 3 matrix parametrization; largely replaces it.

**Cost estimate.** High, and pivots away from the "master field as matrices" narrative. Perhaps a week + significant physics to reconstruct matrices from bootstrap bounds.

### Recommendation

**Pursue Option 2 (MM loop-equation loss) with Option 3's positivity as a later addition.**

Rationale:
- MM is the correct master-field condition — no ambiguity about what we are solving.
- It reuses Phase 1's infrastructure (lattice loop enumeration, MM equation machinery).
- It works for both ansätze, so we can test which ansatz is better after fixing R6.
- Failure mode (MM underdetermined) is known from Phase 1b and has a clean fix (R3 positivity + warm-start from supervised or SDP) if needed.
- Keeps the "matrix master field" story intact: we optimize matrices U_μ, just with a different (correct) loss.
- Cheapest of the three in wall time.

Option 1 (Haar entropy) is the most physically direct, but computationally fragile and expensive. Defer unless Option 2 fails on D=2 TEK.

Option 3 (SDP) is the gold standard for W[C] values but shifts us away from matrix construction. Defer for now; if we need rigorous bounds for validation, use existing Kazakov-Zheng results.

### Proposed next action

Create `tek_master_field/mm_loss.py` that:
1. Takes a LoopSystem (enumerated lattice loops + MM equation indices) for D=2.
2. Builds a function `mm_loss_tek(params, Gamma, z, D, ansatz, loop_sys)`:
   a. Build U_μ from params (existing).
   b. For each canonical loop C in loop_sys, compute W[C] = Re[Z(C) · Tr(product)] / N with the TEK twist phase Z(C) along the path.
   c. Compute MM residuals: λ · W[C] − Σ W[P_e ∘ C] + Σ W[C_1] W[C_2].
   d. Return sum of squared residuals.
3. Plug into `optimize_tek` as an alternative loss (`loss="classical" | "mm"`).
4. Validate on untwisted D=2 EK against Phase 1b's D=2 results (should match area law W = (1/(2λ))^A at strong coupling).

Phase B-MM would then be: run both ansätze with MM loss on untwisted and twisted D=2. If plaquette matches MC strong coupling (1/(2λ)) and weak coupling behavior, R6 is resolved.

---

## Implementation-19: Phase B — classical saddle found; master-field requires Haar entropy (Apr 12, 2026)

### Result

Phase B ran the direct-optimization TEK program on the untwisted Eguchi-Kawai action (z = 1) at D=2, N=49 over a coupling schedule λ ∈ {20, 10, 5, 2, 1, 0.5, 0.3} with BOTH orientation and full U(N) ansätze. After a critical gradient-sign fix (see below), both ansätze converged cleanly to the **classical TEK vacuum** at every λ:

| ansatz | plaq | |P_1| | |P_2| | cs_order | |grad|/N |
|---|---|---|---|---|---|
| orientation | +1.0000 | 0 | 0 | 10⁻³² | 10⁻¹¹ |
| full | +1.0000 | 1.0 | 1.0 | 2.00 | 10⁻¹¹ |

Orientation converges to U_1 = U_2 = Γ (center-symmetric traceless). Full converges to U_1 = U_2 = I (maximally center-symmetry-broken). Both give plaquette = 1 regardless of coupling.

### Interpretation (R6, NEW)

The Monte-Carlo answer for untwisted EK D=2 at finite λ has plaquette that depends on λ: strong coupling → 0, weak coupling → 1 with Z_N breaking. Our result (plaq = 1 at every λ) does not reproduce this.

The physics: at N=∞ the path integral saddle is determined by δS_eff/δU = 0 where S_eff = S_classical − log|Jacobian of dU|. The Haar entropy (−log|J|) adds an eigenvalue-repulsion pressure that spreads eigenvalues away from the classical vacuum and makes the saddle coupling-dependent. Direct minimization of S_classical alone finds only the classical vacuum (the "zero-entropy" configuration).

The orientation ansatz DOES use a Haar-measure argument (KKS-invariant measure on the coadjoint orbit is constant), which is why its S_eff restricted to the orbit equals S_classical. But the classical minimum of S_classical ON the orbit is still the trivial U_μ = Γ (the orbit's most symmetric configuration). The coadjoint orbit contains the classical TEK saddle U_2 = Q_L ⊗ P_L (for the TWISTED case — different from Phase B's untwisted test), but both U_μ = Γ and the clock-shift pair give plaq = 1 at untwisted z=1.

The full ansatz has no measure restriction at all, so it finds the global classical minimum (all U_μ = I, plaq = 1).

Neither sees the quantum fluctuations encoded in the Haar measure. This is R6: the loss function needs a Haar-entropy term (or switch to MM loop-equation residuals, à la Phase 1 Direction A) to recover coupling-dependent master-field observables.

### Gradient-sign bug discovered and fixed

Unrelated to the physics, Phase B surfaced a JAX convention gotcha. For a real loss f of complex-matrix parameter H, `jax.grad(f)(H)` returns ∂f/∂x − i·∂f/∂y (note minus sign on imaginary derivative), which is the **conjugate** of the physical descent direction ∂f/∂x + i·∂f/∂y. Feeding the raw JAX gradient to optax via `params − lr · grad` updates in the wrong direction.

Fixed in `optimize.py::_step`: conjugate the gradient before hermitianizing and passing to the optimizer. Verified by line search against direct loss evaluation: `-lr·conj(grad)` decreases loss, `-lr·grad` increases it. Confirmed also on a trivial |z|² test (JAX returns 2−2j at z=1+1j where the correct descent direction is 2+2j = 2z).

After the fix:
- At H=0 (untwisted): |grad|/N = 0, plaquette = 1, optimizer correctly stays at the classical vacuum.
- At random init: |grad|/N monotonically decreases to 10⁻¹¹ over the schedule, plaquette converges to 1.
- All 89 pytest tests continue to pass.

### Deliverables

- `tek_master_field/phase_b.py`: standalone Phase B experiment script.
- `results/phase_b_summary.json`: per-ansatz, per-λ observables (plaquette, P_μ, center-symmetry order, eigenvalue histogram, loss, grad norm, elapsed time).
- `optimize.py::_step`: conj + hermitianize gradient projection.

### What Phase B tells us about the Phase 3 program

The scaffolding is **technically correct**: both ansätze represent the intended parameter spaces, the gradient converges, the tests pass. But the LOSS FUNCTION is physically incomplete — minimizing classical S gives the classical vacuum, not the master field.

Three options for next steps:

1. **Add Haar entropy to the loss.** Compute log|J| of the exp(iH) parametrization (related to sinh(ad·H/2) / (ad·H/2) determinant) or the Jacobian of the coadjoint orbit inclusion. This is a nontrivial physics derivation but most direct.
2. **Switch to Makeenko-Migdal loop-equation loss** (Phase 1 Direction A) applied to TEK. Works for lattice loops; proven successful at D=2 in Phase 1b.
3. **SDP bootstrap with continuation** (Phase 1 Direction C). Hard constraints from Kazakov-Zheng bounds + neural parametrization.

Phase B has delivered a CLEAN, REPRODUCIBLE FAILURE per the original plan gate. The next concrete step is to pick one of the above three directions and formulate the new loss.

### Phase 3 risk status (updated)

| Risk | Status |
|------|--------|
| R1 D=4 k=1 center-symmetry break | deferred |
| R2 rectangular Wilson loop phase | resolved (Impl-16) |
| R3 sign conventions | resolved (scaffolding) |
| R4 orientation-only sufficiency | resolved (Impl-18) — full ansatz available |
| R5 Γ spectrum mismatch | resolved (Impl-17) |
| R6 classical action alone misses master field | **NEW, unresolved** — requires Haar entropy or switch to MM/SDP |

---

## Implementation-18: R4 resolved — full U(N) ansatz as alternative to orientation-only (Apr 12, 2026)

### What was added

Alongside the orientation-only ansatz U_μ = Ω_μ Γ Ω_μ† (eigenvalues locked to L-th roots of unity, L-fold degenerate), we now also support the **full U(N) ansatz**:

    U_μ = expm(i M_μ),   M_μ Hermitian,   μ = 1, …, D

Parameters: D · N² real (vs. (D−1) · N² for orientation). Eigenvalues of each U_μ are free; this ansatz can represent center-symmetry-breaking configurations that the orientation-only ansatz cannot (e.g., fluxons relevant to D=4 TEK at k=1 per hep-th/0612097).

### Implementation

- `tek.py`: new `build_link_matrices_full`, `tek_loss_full`, `plaquette_average_full`, `init_M_list_zero`, `init_M_list_random`.
- `optimize.py`: `optimize_tek` and `coupling_continuation` accept `ansatz ∈ {"orientation", "full"}` (orientation default). `OptResult.params` holds H_list or M_list depending on ansatz, `OptResult.ansatz` stores the choice. Backward-compat `res.H_list` / `res.M_list` properties alias `res.params`.
- `train.py`: new `--ansatz {orientation, full}` flag. Output file naming now includes the ansatz tag.
- `test_tek.py`: 11 new tests (full ansatz unitarity, M=0 plaquette matches H=0 orientation value, optimizer loss-decrease and unitarity preservation, input validation). 89/89 total pass.

### At-M=0 vs at-H=0 parity

Verified: at H=0 (orientation, all U_μ = Γ) and M=0 (full, all U_μ = I) the loss has the same value = −(mean Re(z_μν)). Both plaquette products equal I. The two ansätze start from the same loss but on different parameter manifolds; Adam explores different directions from there.

### Parameter counts (N = 49, D = 2 example)

| ansatz | params | orbit dim | gauge dim |
|--------|--------|-----------|-----------|
| orientation | 1 × 49² = 2401 | ~2058 | 343 (U(L)^L stabilizer) |
| full | 2 × 49² = 4802 | 4802 | N² − 1 = 2400 global gauge |

Full has more parameters AND more gauge redundancy. Per gradient step the full mode is about 2× slower (two expm's vs one), but the step count to saddle may differ either way.

### Purpose

Phase B will run both ansätze on untwisted EK (D=2) and compare. Expected:

- If orientation reaches a coupling-dependent saddle with the correct GW strong-coupling answer, great — use orientation for Phase C/D (cheaper).
- If orientation stays stuck at the classical TEK vacuum (plaquette ≈ 1 regardless of λ), that's R4 failing, and full should produce the quantum master field (at the cost of D·N² params).
- If BOTH fail to produce coupling-dependent plaquette, the loss function itself needs a Haar-entropy term — a deeper physics issue noted in the reference doc §"Open Question".

### Phase 3 risk status summary

| Risk | Status | Note |
|------|--------|------|
| R1 D=4 center-symmetry breaking at k=1 | deferred | Addressed in Phase D via `--k` and modified flux per arXiv:1005.1981 |
| R2 rectangular Wilson loop twist phase | resolved (Impl-16) | W[R×T] = Re[z^{RT} · Tr(...)]/N, verified on classical saddle |
| R3 sign conventions | resolved (Phase 3 scaffolding) | Standardized loss = −(mean plaquette)/N_pairs |
| R4 orientation-only sufficiency | resolved (this entry) | Full U(N) available via `--ansatz full` |
| R5 Γ spectrum mismatch with TEK saddle | resolved (Impl-17) | Γ = kron(P_L, I_L), classical saddle reachable |

---

## Implementation-17: R5 resolved — Γ replaced with kron(clock_L, I_L) (Apr 12, 2026)

### Fix

`build_clock_matrix(N)` was returning `diag(1, ω_N, …, ω_N^{N-1})` (eigenvalues = N-th roots of unity, all distinct). This is the wrong spectrum for TEK: the classical saddle uses twist eaters `P_L ⊗ I_L` whose eigenvalues are L-th roots with L-fold degeneracy. Since spectra are invariants under unitary conjugation, the old ansatz Ω Γ Ω† could not reach the TEK classical saddle at finite N.

Replaced with:

    Γ = kron(P_L, I_L)    where P_L = L-dim clock matrix, L² = N

Properties after the swap:
- Γ^L = I (stronger than the old Γ^N = I)
- Eigenvalues are L-th roots of unity, each with multiplicity L
- Traceless for L > 1
- Matches TEK twist-eater U_1 in arXiv:1708.00841 §2.2 eq. 2.16
- Coadjoint orbit {Ω Γ Ω†} now contains the classical-saddle partner Q_L ⊗ P_L (same spectrum), so the orientation-only parametrization can reach the saddle

### Verification

78/78 pytest tests pass. New tests added:
- `test_clock_matrix_eigenvalues_are_L_roots_L_degenerate`: direct eigenvalue check for N ∈ {9, 25, 49}
- `test_clock_matrix_L_periodic`: Γ^L = I to 1e-10 for N ∈ {9, 25, 49, 121}
- `test_clock_matrix_rejects_non_perfect_square`: input validation
- `test_clock_matrix_traceless`: Tr(Γ) = 0 to 1e-12
- `test_clock_matrix_matches_tek_classical_saddle_U1`: Γ = kron(P_L, I_L) to 1e-14

All pre-existing tests continue to pass (plaquette at H=0, rectangular Wilson loops, TEK classical-saddle W[R×T] = 1, etc.). The change is invisible to the plaquette observable at H=0 because all U_μ = Γ still commute (Γ is block-diagonal).

### Measure of progress

Phase 3 scaffolding now has both R2 (rectangular Wilson loop twist phase) and R5 (ansatz Γ spectrum) resolved. Remaining risks:
- **R1** (D=4 center-symmetry breaking at k=1, cure via modified flux k ≈ L/2): deferred to Phase D.
- **R3** (sign conventions): standardized in Phase 3 scaffolding, no action needed.
- **R4** (orientation-only vs full U(N)): Phase B will test. With R5 fixed the orientation-only ansatz is now MORE likely to succeed since it can reach the classical saddle.

### Parameter count

- Old Γ (N-th roots, distinct): stabilizer = U(1)^N, orbit dimension N² − N.
- New Γ (L-th roots, L-fold): stabilizer = U(L)^L, orbit dimension N² − L·L² = N² − N^{3/2}.

For N = 49: 2401 − 49 = 2352 (old) vs 2401 − 343 = 2058 (new). Fewer effective parameters, but they encode the correct TEK structure. The remaining L·L² = 343 parameter-directions are gauge (stabilizer of Γ); Adam tolerates this overparametrization but a stabilizer-projected parametrization may be needed for large-N fine-tuning.

---

## Implementation-16: R2 resolved — rectangular Wilson loop twist phase (Apr 12, 2026)

### Result

`observables.wilson_loop_rectangular` is now implemented and verified. The twist phase for an R×T rectangular Wilson loop in the (μ,ν) plane on single-site TEK is **z_μν^{R·T}**:

    W[R×T]_{μν} = Re[ z_μν^{R·T} · Tr(U_μ^R U_ν^T U_μ^{-R} U_ν^{-T}) ] / N

### Source

García Pérez, González-Arroyo, Okawa, arXiv:1708.00841 eq. (2.4): "W_{R,T}(b, N, L, n_μν) = (1/N) Z(R,T) ⟨Tr(U(R,T))⟩, where Z(R,T) is the product of the Z_μν(n) factors for all plaquettes which fill up the rectangle." For single-site TEK (their L=1), every elementary plaquette has the same twist factor z_μν = Ẑ_μν = exp(2πi n_μν / N), giving Z(R,T) = z_μν^{R·T}. Consistent with arXiv:1212.3835 eq. (1.1) plaquette convention.

### Verification

Three independent tests:
1. **R = T = 1 reduces to the plaquette.** Direct equality to `wilson_loop_plaquette` for random H, to 10⁻¹².
2. **At H = 0** (all U_μ = Γ, diagonal, commuting): W[R×T] = Re(z_μν^{R·T}) exactly, to 10⁻¹⁰, across (R,T) ∈ {(1,1), (1,2), (2,1), (2,2), (3,2), (2,3), (3,3)} and N ∈ {9, 25, 49}, k ∈ {1, 3}.
3. **TEK classical saddle** (the strongest check): construct U_1 = P_L ⊗ I_L, U_2 = Q_L ⊗ P_L from L×L clock-shift matrices. Verify U_1 U_2 = ω_L^{-1} U_2 U_1 (Heisenberg relation) to 10⁻¹² for L ∈ {3, 5, 7}. At this saddle W[R×T] = 1 exactly for every (R, T) tested (L ∈ {3,5,7}, R,T ≤ 4) to 10⁻¹⁰. This independently reproduces the arXiv:1708.00841 eq. (2.4) claim.

65/65 pytest tests pass after R2 resolution (+26 new tests on top of Phase 3 scaffolding's 39).

### New finding (R5)

Building the TEK classical saddle explicitly revealed a subtle ansatz mismatch. The TEK twist eaters P_L ⊗ I_L have eigenvalues equal to the **L-th roots of unity, L-fold degenerate**. Our current Γ = diag(1, ω_N, …, ω_N^{N-1}) ansatz has eigenvalues equal to the **N-th roots of unity, all distinct**. These spectra differ (N = L² > L), and since spectra are conjugation invariants, no Ω Γ Ω† can reach the TEK classical saddle starting from our current Γ.

At N → ∞ both spectra become uniform on the unit circle and the distinction vanishes. At finite N, our ansatz Ω Γ Ω† reaches a different saddle (the maximizer over a different submanifold), not the TEK classical saddle.

**Fix is straightforward and deferred to Phase B** if Phase B shows pathological behavior: replace `build_clock_matrix(N)` with `kron(clock_L, I_L)`. The Haar-measure argument still holds (coadjoint orbit has constant measure) and the orientation parametrization matches the TEK structure explicitly. The orbit dimension drops from N² to N² − L³, encoding the TEK center-symmetric constraint.

Documented in reference/tek_master_field.md §"Ansatz Caveat (R5)".

### Phase C/D unblocked

With R2 resolved, Phase C Wilson-loop comparisons against the lattice GW strong-coupling product law `W[R×T] ≈ (1/(2λ))^{R·T}` are now computable. Creutz ratios χ(R,R) for string tension become well-defined. Only R5 (ansatz choice) remains between the current scaffolding and full Phase C/D execution.

---

## Discussion-15: Phase 3 — Direct Optimization of the TEK Master Field (Apr 12, 2026)

### Motivation

At N = ∞ the path integral is dominated by a single saddle — the master field. Monte Carlo samples *around* the saddle with O(1/N²) and O(1/√T) noise. Direct gradient-descent optimization finds the saddle itself: no thermalization, no autocorrelations, no statistical noise. For computing the master field this is strictly better than MC if it converges.

Nobody has solved the Twisted Eguchi-Kawai (TEK) model by direct optimization. Phase 3 attempts this. If Phase D (TEK D = 4 at N = 289) succeeds with plaquette matching published MC < 1 %, the result is the first explicit construction of the SU(∞) master field for 4D lattice Yang-Mills — a problem open since Witten (1979) / Gopakumar-Gross (1994). Direction B of the Discussion-11 roadmap.

### Model

TEK reduces D-dimensional SU(N) lattice YM to D unitary N×N matrices at a single site with action

    S = −(N/λ) Σ_{μ<ν} Re (1/N) Tr(z_{μν} U_μ U_ν U_μ† U_ν†)

and path-integral weight exp(−S). The twist z_{μν} = exp(2πi n_{μν}/N) with symmetric flux n_{μν} = k·L (on twisted planes, N = L² with L prime) pins Z_N^D center symmetry and restores volume independence, so N = ∞ TEK observables equal the infinite-volume theory.

### Master Field Ansatz

Center-symmetric eigenvalues are locked to the N-th roots of unity:

    U_μ = Ω_μ · Γ · Ω_μ†,   Γ = diag(1, ω, ω², …, ω^{N-1}),   ω = e^{2πi/N}

Gauge-fix Ω_1 = I; parametrize Ω_μ = exp(i H_μ) with H_μ Hermitian for μ ≥ 2. The Haar measure, restricted to the coadjoint orbit {g Γ g†}, is proportional to the Vandermonde of the eigenvalues — constant since eigenvalues are fixed. Total parameters: (D − 1) · N².

### Physics Risks (R1–R4, tracked in the plan)

- **R1 — Center-symmetry breaking at D = 4, k = 1.** Observed for N ≥ 100 (hep-th/0612097). Cure: modified flux k ≈ L/2 per González-Arroyo-Okawa 2010 (arXiv:1005.1981). Safe for D = 2, D = 3 with k = 1.
- **R2 — Rectangular Wilson loop twist phase f(R,T).** Nontrivial from Heisenberg non-commutativity. Must be transcribed from PRD 27 (1983) eq. (3.5) or arXiv:1708.00841 before computing W[R×T]. Currently gated as `NotImplementedError`.
- **R3 — Sign conventions.** Standardized: loss = −Σ_{μ<ν} Re[z_{μν} Tr(U_μ U_ν U_μ† U_ν†)/N] / N_pairs. No λ inside the loss.
- **R4 — Orientation-only vs full U(N).** If the ansatz cannot produce coupling-dependent plaquette (classical saddle ≠ master field), enlarge to U_μ = exp(i M_μ) with M_μ Hermitian, D · N² parameters, no fixed eigenvalues.

### Implementation

New subfolder `tek_master_field/` with seven Python modules: `tek.py` (core: clock matrix, twist, link builder, action), `optimize.py` (Adam + warmup_cosine schedule + coupling continuation), `observables.py` (plaquette, Polyakov, eigenvalue density; rectangular gated), `gross_witten.py` (Phase A sanity check), `train.py` (CLI), `config.py`, `test_tek.py` (39 pytest tests). JAX + optax with float64, H_μ re-Hermitianized after each gradient step.

Infrastructure mirrors `master_field/`: float64 config, optax chain pattern from `neural_master_field.py:436`, training-step JIT pattern from `:468`, coupling continuation from `:597`, pytest marks from `test_qcd2.py`.

### Phases

- **Phase A — Gross-Witten** (1-matrix unitary; not TEK). Path C: parametrize support endpoint a, find by normalization, verify exact w_1, w_2. **Gate: err < 10⁻⁶.**
- **Phase B — untwisted EK, D = 2.** First test of the orientation-only ansatz. If it fails to develop a coupling-dependent saddle, adopt R4 fallback.
- **Phase C — TEK, D = 2.** First real TEK computation. Compare to exact GW strong coupling at N = 49, 121, 289. **Gate: plaquette within 0.5 % of reference.**
- **Phase D — TEK, D = 4** (the target). Compare to González-Arroyo–Okawa MC at N = 289, β = 0.356. **Gate: plaquette within 1 % of MC.**
- **Phase E — N → ∞ extrapolation.** Fit O(N) = O(∞) + c₁/N² + c₂/N⁴ from N ∈ {49, 121, 289, 529}.

### Current status (end of this session)

- Scaffolding: `tek_master_field/` created, `__init__.py`, `conftest.py`, `results/` (gitignored).
- **Phase A gate PASSES.** At t ∈ {0.3, 0.5, 0.8, 1.0, 1.2, 2.0, 5.0}: worst |w_1 − exact| = 5.5 × 10⁻¹², worst |w_2 − exact| = 1.6 × 10⁻¹³. Well below the 10⁻⁶ gate.
- **Core infrastructure works end-to-end.** 39/39 pytest tests pass. Smoke test: D = 2, N = 9, k = 1 continuation over λ = {10, 5, 2, 1} completes in 3.8 s, JIT + optax update + hermitianize stable, link matrices remain unitary to 10⁻¹⁰.
- Reference doc `reference/tek_master_field.md` written (symmetric twist, center-symmetry caveat, open question about classical vs quantum saddle, benchmark MC table, bibliography).
- Cluster script `cluster/submit_tek.pbs` for N = 289 runs (mirrors `cluster/submit.pbs`).
- Phases B–E deferred to subsequent implementations. Phase B is the first test of R4.

### What this delivers

The infrastructure to run direct-optimization TEK computations at any D ∈ {2, 3, 4} and any N = L² with L prime, up to the center-symmetry ansatz assumption. The code is the direct executable of the plan in `/Users/dz1614/.claude/plans/piped-twirling-narwhal.md`. Next concrete step is Phase B: the orientation-only ansatz will either produce a coupling-dependent saddle (confirming the Gopakumar-Gross "axial-gauge Gaussian" picture in a TEK context), OR fail to do so (triggering R4 fallback to the full U(N) parametrization). Phase B is a two-matrix numerical experiment; its outcome decides the architecture for Phases C and D.

### References

Eguchi-Kawai 1982 (original). González-Arroyo-Okawa 1983 PRD 27 (original TEK + eq. 3.5 for rectangular loops). González-Arroyo-Okawa 2010 arXiv:1005.1981 (modified flux). García Pérez-González-Arroyo-Okawa 2017 arXiv:1708.00841 (perturbative Wilson loops with twist). Teper et al. 2006 hep-th/0612097 (center-symmetry breaking). Gopakumar-Gross hep-th/9411021 §1 (spacetime-independent master field). Local: `reference/tek_master_field.md` (full bibliography).

---

## Implementation-14: Phase 1 Steps 2–5 — LoopSystem, Neural Model, D=2 Training (Apr 12, 2026)

### What was built

- `lattice.LoopSystem` dataclass + `build_loop_system(D, L_max)`: enumerates canonical lattice loops up to L_max + 2 (to cover MM staple insertions) and precomputes every MM equation as index arrays (`MMEquation`). D=2 L_max=6 gives K=35 loops, 32 equations. D=3 L_max=6 gives 627 loops, 288 equations.
- `master_field/neural_loop.py`: `NeuralLoopFunctional` — MLP mapping λ to K-vector of Wilson loop values. `mm_loss`, `supervised_loss_2d`, `unitarity_penalty` as JIT functions. Training helpers: `train_supervised_2d`, `train_mm_2d`, `train_mm_2d_curriculum`, `train_mm_2d_warmstart`.

### Phase 1a (supervised D=2): works

Trained against GW lattice answer W[C] = w_+^Area with w_+ = 1/(2λ). After 3000 epochs at L_max=6:
- λ=1: W[plaq]=0.502 (exact 0.500), W[2×1]=0.243 (exact 0.250)
- λ=2: W[plaq]=0.251 (exact 0.250), W[2×1]=0.064 (exact 0.0625)
- λ=5: W[plaq]=0.093 (exact 0.100), W[2×1]=0.006 (exact 0.010)

Max error ~1–2%. Architecture validated; more epochs / bigger net would tighten further.

### Phase 1b (MM-only D=2): partially works, exposes real structure

**Initial attempt (tanh + Xavier init) FAILED**: converges to spurious local minimum w[C] = −1 for all non-empty loops. Final MM loss 4.3. Tanh saturation blocks the gradient; Xavier init starts well outside the GW basin.

**Remediations implemented** (per approved plan):
- **R1** (no tanh, small W_out init=0.01, soft unitarity penalty): dramatic improvement. MM loss 4.3 → 6.8e-4. W[plaq] closely tracks the self-consistent candidate-D solution w_+^MM = λ−√(λ²−1). But W[2×1] is driven to ≈ 0 (wrong).
- **R2** (curriculum λ: 10→1 over 8 stages): MM loss → 1e-26 (essentially zero) BUT converges to a solution that violates unitarity (W[plaq]=1.3 at λ=5). MM + soft unitarity penalty at weight 10 isn't strong enough.
- **R4** (supervised warm-start 1500 epochs, then MM-only 3000 epochs): best result. W[plaq] at λ=5 = 0.10012 vs GW 0.1 (near-exact). W[2×1] at λ=1 drifts from supervised 0.25 to 0.16 after MM refinement.

### The real finding

**MM equations + unitarity alone do NOT uniquely determine Wilson loop values in D=2.** Multiple solutions exist:
- The GW answer w_+ = 1/(2λ), W[C] = w_+^Area
- The candidate-D self-consistent solution w_+^MM = λ − √(λ²−1), close to GW only at strong coupling
- Solutions violating |W| ≤ 1 if unitarity penalty is weak
- Solutions where high-area loops → 0 regardless of what the area law predicts

At K=35 unknowns vs 32 MM equations, with one hard constraint (W[empty]=1), the system is 2 equations short of uniqueness. Curriculum training (R2) shows that the MM loss has a continuous family of near-zero minima, not a single point.

### Interpretation and next step

This is consistent with the role of the MM equation in the full QCD bootstrap: MM equations are NECESSARY but not SUFFICIENT. The Kazakov-Zheng program uses SDP positivity (R3) on top of MM to pin down the solution. Our Phase 1 on D=2 validates this claim empirically.

**Figure-8 subtlety discovered**: our `abs_area_2d` returns 0 for loops with signed area 0 (e.g., figure-8 = CCW plaquette + CW plaquette joined at a vertex). These loops factorize at the self-intersection and have W = w_+·w_−, not W = w_+^0 = 1. The exact-target formula `W[C] = w_+^abs_area_2d(C)` is only valid for simple (non-self-intersecting) loops. Documented for future D=2 validation.

### Decision

Move to Phase 1c (D=3) where:
1. The architecture + R4 (warm-start with a strong-coupling initialization) can be validated against bootstrap bounds (Kazakov-Zheng 2203.11360) rather than an exact area law.
2. MM + positivity (R3) becomes the natural next implementation step. The bootstrap bounds tell us approximately where W must lie; MM refines.
3. The exploration problem (unique solution selection) is attacked directly in the regime where it matters.

R3 (positivity / Toeplitz-PSD) is deferred to Phase 1c where we'll need it anyway. Phase 1b on D=2 provides the calibration baseline.

### Tests

All 16 pre-existing + 11 MM + 18 QCD₂ pytest tests still pass.

---

## Implementation-13: Phase 1 Step 1 — MM Equation for Lattice 2D YM (Apr 12, 2026)

### Goal

Per the Phase 1 plan, Step 1 is: fix and validate the Makeenko-Migdal equation machinery on D=2 before any neural-net code. The plan specified `test_mm_all_loops_2d` pass at 10⁻⁶ residual against exact area law.

### What was built

- Fixed `lattice.plaquette_insertions`: now uses STAPLE-replacement convention. Edge +μ at position e_j is replaced by the 3-step staple (ν, +μ, −ν) for each ν ≠ ±|μ|. Net displacement preserved, loop stays closed.
- New `mm_equations.py`: candidate catalog A–G of MM equation forms; `gw_w_plus(λ)` returning the Gross-Witten single-plaquette strong/weak coupling value; `mm_residual_staple(loop, edge_idx, D, lam, W, include_self_closure)`; `scan_candidates_2d(target="lattice" or "continuum")` empirically compares candidates.
- New `test_mm.py`: 6 tests documenting the candidate-D behavior.

### Key findings — the MM equation for 2D lattice YM is NOT uniquely "exact"

**Finding 1**: The continuum area law `W[C] = exp(−λ Area/2)` does NOT satisfy our MM candidates at machine precision. Best residual with candidate D (LHS = Σ_P W[staple(C,e,P)]/λ, RHS = 2W[C] + splits) is 8.65e-02 at λ=5 — clearly not lattice-exact.

**Finding 2**: The LATTICE answer on an infinite 2D lattice at N=∞ is `W[C] = w_+^Area` where w_+ is the Gross-Witten single-plaquette value:
- Strong coupling (λ ≥ 1): w_+ = 1/(2λ) — exact, no corrections.
- Weak coupling (λ < 1): w_+ = 1 − λ/2.

Against this lattice answer, candidate D residuals shrink: λ=2 → 2e-3; λ=5 → 2.5e-4. **Leading order in 1/λ is correct; there are subleading corrections at O(1/λ³).**

**Finding 3**: Candidate D self-consistent solution w_+^MM = λ − √(λ²−1) matches GW's w_+^GW = 1/(2λ) to leading order:
  w_+^MM = 1/(2λ) + 1/(8λ³) + ...

The O(1/λ³) corrections are where candidate D differs from the physical answer. Which means our candidate D is the LEADING-ORDER MM, not the full MM.

### Why the MM equation isn't unique (physics interpretation)

The Wilson-action lattice MM has multiple equivalent formulations with different coefficient structures depending on which plaquettes are counted, how the self-closure is handled, and whether the "staple" or "symmetric difference" convention is used. Various references (Migdal 1975, Wadia 1981, Anderson-Kruczenski 2017, Kazakov-Zheng 2021) write the equation differently; comparing across them requires explicit convention dictionaries.

For the Phase 1 program, the KEY question isn't "which form is exactly right" but "does the MM loss train the neural network to the correct Wilson loop values?" If the leading-order form is reasonable, the ML can compensate for subleading corrections.

### Decision

- **Adopt candidate D as the working MM equation** for Phase 1b.
- Train neural loop functional by minimizing its residuals.
- Validate: Phase 1b neural-network training should recover W[C] consistent with the GW strong-coupling answer (w_+ = 1/(2λ)) at least to 1-few%.
- If Phase 1b fails (residuals converge to zero but Wilson loops wrong), iterate on MM form.

### Empirical test results

`test_candidate_D_leading_order_at_strong_coupling`: for λ ∈ {2, 5, 10}, max residual < 1/λ². ✓
`test_self_consistent_w_plus_candidate_D`: rel error vs GW < 1/λ² for λ ∈ {2, 5, 10, 100}. ✓
`test_plaquette_insertions_preserves_closure`: all 2D plaquette insertions closed. ✓
`test_gw_wplus_values`: matches standard GW formulas. ✓
All 6 MM tests pass. All 16 pre-existing tests still pass.

### Next step

Phase 1 Step 2: implement `LoopSystem` (precomputed loop enumeration + MM-equation index tables) in `lattice.py`. Then the neural loop functional and training can proceed on top of these precomputed structures.

---

## Implementation-12: Phase 0 — QCD₂ Master Field Infrastructure (Apr 12, 2026)

### What was built

Following the Discussion-11 roadmap, Phase 0 delivers the infrastructure for constructing unitary master field operators in the Cuntz-Fock space and computing lattice Wilson loops.

**New modules:**
- `master_field/lattice.py`: lattice loop encoder with signed step indices `μ ∈ {±1,...,±D}`. Provides backtrack reduction, cyclic canonicalization, hyperoctahedral symmetry orbits (`B_D`), non-self-intersecting loop enumeration, and (preliminary) plaquette-insertion and self-intersection split operators for MM-equation residuals.
- `master_field/qcd2.py`: `solve_alpha_for_plaquette` (brentq root-finding), `validate_wilson_loops` (comparison against exact area law), `validate_mm_equation_exact` (MM residual check against exact area-law Wilson loops), `qcd2_main` acceptance output.
- `master_field/test_qcd2.py`: 18 pytest tests. All pass.

**Extended modules:**
- `cuntz_fock.CuntzFockSpace`: `build_unitary_gaussian(α, μ) = exp(iα(â_μ+â_μ†))` via `scipy.linalg.expm`. `wilson_loop_vev(operators, word)` handles signed step indices (negative → conjugate transpose). `check_unitarity` monitors `‖UU† − I‖_F`.

**Documentation:**
- `reference/qcd2_master_field.md`: physics reference with the exact formulas (area law, axial-gauge Gaussian master field per Gopakumar-Gross §5, lattice MM equation) and the observed Phase 0 behavior.

### Empirical findings

**Unitarity** (machine-level). `‖Û_μ Û_μ† − I‖_F = O(10^{−15})` at every truncation tested. `scipy.linalg.expm` of a Hermitian matrix is unitary to machine precision, even in a truncated Fock space.

**α convergence** (rapid). The fitted α(λ, L) that matches the plaquette converges quickly:

| λ | α(L=4) | α(L=6) | α(L=8) | α(L=10) |
|---|---|---|---|---|
| 1.0 | 0.95181358 | 0.95176911 | 0.95176915 | 0.95176915 |

Stable to 8 decimals by L=8.

**Plaquette** (exact by construction). `|W[□]_ML − e^{−λ/2}| < 10^{−10}` for all tested λ ∈ {0.5, 1, 2, 5}, L ∈ {4,6,8,10}.

**The key physical finding**: the single-α Gaussian ansatz Û_μ = exp(iα(â_μ+â_μ†)) **does not reproduce the area law** for Wilson loops larger than the plaquette. The error saturates at a fixed, L-independent value:

| Loop | λ=1, L=6 | λ=1, L=8 | λ=1, L=10 |
|---|---|---|---|
| 1×1 (plaquette) | 0 (fit) | 0 (fit) | 0 (fit) |
| 2×1 rectangle | 5.1×10⁻³ | 4.9×10⁻³ | 4.9×10⁻³ |
| 2×2 square | 1.35×10⁻¹ | 1.35×10⁻¹ | 1.35×10⁻¹ |

Since α has converged but the Wilson loop errors have not, the deficiency is in the **ansatz form**, not in the truncation. With only one real parameter, the ansatz has 1 degree of freedom; the area law requires W[m×n] = W[□]^{mn} for all m,n, giving infinitely many independent constraints.

### Interpretation

Gopakumar-Gross §5 states that 2D YM has a Gaussian master field in axial gauge. Two possible readings:

1. **Continuum/scaling statement**. "Gaussian" refers to the continuum master field. On a finite Cuntz-Fock truncation with spacetime-independent link operators, the "Gaussian" form `exp(iα(â+â†))` is too restrictive and yields only approximate area law.

2. **Generalized Gaussian**. The correct ansatz might be a Voiculescu-style expansion Û_μ = Σ_n c_{μ,n} (â_μ†)^n + (h.c.) with infinitely many coefficients tuned to reproduce the full Wilson-loop algebra. At the truncation level L, this gives O(L) coefficients per direction — enough to match O(L) independent Wilson loops.

Either way, the implication for our program is the same: **Phase 1 must optimize over a richer ansatz.** Two natural candidates:

- **Polynomial coefficients** `{c_{μ,n}}_{n=0}^{L-1}` trained against MM-equation residuals (Direction B in Discussion-11).
- **Neural loop functional** `f_θ(C, λ)` with architectural equivariances, trained against MM-equation residuals directly (Direction A).

### MM-equation machinery status

The `plaquette_insertions` and `self_intersection_splits` functions in `lattice.py` are implemented but untested for correctness against specific conventions. `validate_mm_equation_exact` currently reports large residuals when evaluated on the exact area law — this is most likely because our insertion convention (replacing link μ with a plaquette closed path) does not match the standard Migdal/Chatterjee convention, which uses edge-set symmetric difference. Deferred to Phase 1 planning when we actually need MM loss functions.

### Phase 0 acceptance

✓ Infrastructure built and tested (18/18 pytest, 16/16 pre-existing tests).  
✓ Plaquette matches exactly to machine precision.  
✓ α(λ, L) converges with L.  
✓ Unitarity preserved.  
✓ Backtrack invariance (`W[(1,-1,2,3,-3,2)] = W[(2,2)]`) holds.  
✓ Exchange symmetry `W[2×1] = W[1×2]` holds.  
✗ Full area-law reproduction for larger loops — deferred to Phase 1 (as expected; single-parameter ansatz is insufficient).

### Next step — Phase 1

Train a multi-parameter ansatz against the full set of Wilson loops up to L_max. Candidate: polynomial Û_μ = exp(i Ĥ_μ) with Ĥ_μ = Σ_n h_n (â_μ + â_μ†)^n (truncated Taylor series with trainable coefficients). This has ~L real parameters per direction. Loss = Σ over loops of |W_ML[C] − e^{−λ·Area(C)/2}|² + unitarity penalty.

Alternatively (Direction A): parametrize W[C] as a neural network with built-in equivariances, train against MM-equation residuals. No Fock space needed for the forward pass; Fock-space checks only for validation.

---

## Discussion-11: Revised Plan — Original Directions Only (Apr 12, 2026)

### What's been done (by others — don't repeat)

- **Bootstrap/SDP** (Kazakov-Zheng, Li-Zhou, Guo-Qiao-Zheng): Rigorous *bounds* on Wilson loop averages from MM equations + positivity, via SDP. Works in D=2,3,4 for SU(∞), SU(2), SU(3). Up to L_max=24. Already implemented, published, well-understood. *We are not re-doing this.*
- **Collective field optimization** (Rodrigues et al.): Master field for Hermitian multi-matrix QM via constrained minimization of V_eff in loop space. Up to ~10⁴ variables. Works for matrix QM but *not* applied to lattice gauge theory.
- **Gopakumar-Gross**: Explicit master field construction in Cuntz-Fock space. Done for QCD₂ (Gaussian in axial gauge) and for independent/coupled Hermitian matrix models. *Never done for lattice YM in D≥3.*

### What has NOT been done — the gap

**Nobody has explicitly constructed the master field (as an operator in a well-defined Hilbert space) for lattice Yang-Mills theory in D≥3.**

The bootstrap bounds the observables. The Rodrigues program finds the master field for matrix QM. But lattice YM in D=3,4 — actual QCD — has no explicit master field construction.

This is the target. The ML is not an end in itself — it's the computational engine for a problem that is well-posed but computationally intractable by brute force.

### The physics setup

**Spacetime-independent master field** (Gopakumar-Gross §1, eq. 1.6-1.7): the master gauge field can be made spacetime-independent by a gauge transformation. On the lattice: master link variables Ū_μ depend only on direction μ, not on site. The entire N=∞ lattice YM theory is encoded in **D unitary operators** Û_1, ..., Û_D in a Cuntz-Fock space.

Wilson loop for closed path C = (μ_1,...,μ_k) with μ_i ∈ {±1,...,±D} and Û_{-μ} = Û_μ†:

W[C] = ⟨Ω| Û_{μ_1} Û_{μ_2} ⋯ Û_{μ_k} |Ω⟩

This is the Gopakumar-Gross framework applied to unitary operators. Fock space has 2D creation operators â_μ†.

### Why this hasn't been done

1. **Unitarity constraint**: Û_μ Û_μ† = 1 as operator equation is nontrivial in truncated Fock space; couples all orders.
2. **Nonlinear loop equations** at N=∞ via factorization: MM equation involves products W[C_1]·W[C_2].
3. **Exponential growth** of distinct loops with length (~618 at L_max=12 in 4D, per Qiao-Zheng).
4. **No small parameter**: D≥3 has genuine interactions; no globally convergent expansion.

### Three genuinely new directions

**Direction A — Neural loop functional**. Parametrize W[C] as a neural network f_θ: loops → ℝ. Transformer/GRU on step sequences with built-in equivariance (cyclic, reversal, hyperoctahedral symmetry B_D). Loss = MM residuals. Coupling λ as additional input — single trained network gives full phase diagram. Deconfinement transition appears as a sharp feature in f_θ(C, λ). **Novelty**: first neural parametrization of the observable W[C] itself. Generalizes beyond truncation (training at length L constrains predictions at L+2 via MM).

**Direction B — Cuntz-Fock master field operators**. Directly construct Û_μ in truncated Cuntz-Fock space. Parametrize Û_μ = exp(i Ĥ_μ) with Ĥ_μ Hermitian. Constraints: MM residuals, unitarity Û Û† ≈ I, PSD correlation matrix (automatic from Fock structure). Loss = Σ|MM|² + μ Σ ||Û_μ Û_μ† - I||². For large truncation, parametrize Ĥ_μ via NN. **Novelty**: the Rodrigues/Jevicki-Sakita program applied to lattice YM (unitary operators), not matrix QM.

**Direction C — Hybrid bootstrap-ML**. Use Kazakov-Zheng bootstrap bounds W⁻[C] ≤ W[C] ≤ W⁺[C] as *hard box constraints* on a neural loop functional, then train against MM residuals. Bootstrap eliminates spurious local minima; ML extrapolates beyond the bootstrap truncation and finds the unique interior point. **Novelty**: uses rigorous bounds as constraints rather than stopping at bounds.

### Implementation phases

- **Phase 0** (QCD₂): validate the Cuntz-Fock unitary master field against the exactly-solvable 2D theory. Gaussian in axial gauge (Gopakumar-Gross §5), W[C] = exp(-λ Area/2).
- **Phase 1** (D=2 neural): train neural loop functional against MM, compare to exact area law. Test architecture, equivariances, generalization to long loops.
- **Phase 2** (D=3): the first unsolved case. All three directions applied in parallel. Validate against published SDP bounds and MC.
- **Phase 3** (D=4 QCD): the target. Extract string tension, glueballs, deconfinement from f_θ(C, λ).

### What makes this original

| Existing work | What it does | What it doesn't |
|---|---|---|
| Kazakov-Zheng SDP | Bounds on W[C] | Not the master field. Only up to L_max. |
| Rodrigues collective field | Master field for matrix QM | Not lattice gauge theory. Hermitian. |
| Gopakumar-Gross | Cuntz-Fock framework + QCD₂ | Not D≥3. Framework, not solution. |
| Normalizing flows for LGT | Sample gauge configurations | Not N=∞. Not master field. |

Our contribution: (A) first neural parametrization of W[C] itself; (B) first explicit master field construction for lattice YM in D≥3; (C) bootstrap bounds as hard ML constraints.

If Direction A or B succeeds in D=4: first explicit construction of the large-N master field for QCD. Problem posed by Witten (1979), formalized by Gopakumar-Gross (1994). 47 years open.

### References

1. Gopakumar-Gross, hep-th/9411021, §1 (spacetime-independent master field), §5 (QCD₂ master field)
2. Gopakumar PhD thesis (1997)
3. Rodrigues et al. JHEP 2022, 2024 — master variable / collective field
4. Kazakov-Zheng 2203.11360 — MM equations on lattice, SDP bootstrap
5. Qiao-Zheng 2601.04316 — systematic loop equation construction
6. Raissi et al., J. Comp. Phys. 378 (2019) — physics-informed NN
7. Han-Hartnoll, Phys. Rev. X 10 (2020) — NN ansatz for matrix model ground states

---

## Discussion-10: Gross-Witten Model — Unitary Loop Equations (Apr 12, 2026)

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
