# Continuation selects the large-N master field: finite-cutoff loop equations plus positivity under-determine it

**Status:** methods note (markdown draft, not LaTeX). Branch `matrix-master-field`.
Most numerical claims below cite the run and file that produced them. A few matrix-side
cross-checks — the cold-BM multistart spread (§3.1), the HMC m[A²] (§2.2/§5.1), and the
RP loss floor (§8.1) — are reproducible from the named script but their run output is not
persisted; each is flagged inline. Where a brief and a repo doc disagree the discrepancy
is flagged inline where it occurs rather than silently reconciled.

---

## 1. Introduction and claims

Constructing the large-N master field (the N=∞ saddle point) by minimizing the
residual of the planar loop equations — Schwinger–Dyson for matrix models,
Makeenko–Migdal for gauge theory — over a positive, tracial ansatz is an established
program. The idea and most of its ingredients are prior art:

| Ingredient | Prior art | Status here |
|---|---|---|
| Master field as N=∞ saddle | Gopakumar–Gross, hep-th/9411021 | used, not novel |
| Cuntz/Fock operator representation | Gopakumar–Gross hep-th/9411021; Douglas hep-th/9411025 | used, not novel |
| Loop-residual minimization to build the master field | Maeta arXiv:2605.10720 (one/two-matrix, finite-dim regularization, no positivity); de Mello Koch–Jevicki et al. arXiv:2108.08803, arXiv:2306.00935 (collective variational + SDP); Engelhardt–Levit hep-th/9609216 | used, not novel |
| "Construction gives points where the bootstrap gives bounds" framing | Maeta arXiv:2605.10720 | used, not novel |
| Rigorous existence of the master field | Guionnet–Shlyakhtenko arXiv:1204.2182 (matrix); Lévy arXiv:1112.2452 (2D gauge) | context |

What is **not** settled is uniqueness in practice. This note reports a measured fact
and a selection method:

1. **Under-determination is real and measurable.** At any finite constraint cutoff the
   loop-equation variety, with positivity switched off, is a *continuum*: we measure its
   dimension in a two-matrix quantum mechanics (3, 8, 22 at truncation levels 4, 6, 8,
   computed here as nV − rank J). Restoring positivity does not collapse it to a point —
   the feasible set keeps positive extent (the m[A²] projection is an interval, §3.1) —
   and the loop equations admit spurious solutions, including an explicit machine-zero
   solution of the 2D lattice Yang–Mills loop equations with negative Wilson loops
   (residual 1.3e-14).
2. **Continuation from a solvable anchor is the decisive selection lever — jointly with
   the constraint set.** In the two-matrix model, cold start lands on a spurious branch
   (m[A²]=0.343) while continuation from the exact h=0 anchor tracks the physical one; on
   the gauge side continuation drives the residual orders deeper and improves the
   plaquette, but there branch discrimination is *shared* with the constraint set (§4.2),
   not continuation's alone.
3. **Reach ≈ the constraint cutoff; no free large observables — matrix-demonstrated,
   gauge-directional.** A provably-wrong long-moment prediction (matrix side) is caught by
   a rigorous bootstrap bracket plus independent HMC; the area-vs-perimeter correction
   limits transfer to gauge, where the reach evidence is directional only (§5.2).
4. **A truth-free reach diagnostic (proposed).** Agreement across independent machine-zero
   solutions of the same objective is proposed — and illustrated on one prototype (§6) —
   as a reach proxy where no exact answer exists; it bounds solver scatter, not shared
   truncation bias.

The claims are demonstrated on two testbeds with known answers (a two-matrix model and
2D lattice Yang–Mills), chosen because the physical solution is available for
falsification. The intended discovery arena is D=3, discussed in §8.

---

## 2. Setup: operator master field and the two testbeds

### 2.1 Operator master field

Matrices (or gauge links) are represented as functions of free generators on a
truncated Cuntz–Fock space. The free vacuum is the tracial state (Voiculescu), so
positivity and traciality of τ hold automatically to all orders — not only up to a
moment cutoff as in the convex bootstrap. (This is exact for the ideal operator field and
for the Hermitian two-matrix case; the unitary gauge realization Û=exp(iĤ) inherits it
only while the finite-order expm stays accurate — outside that envelope unitarity is
numerically violated, §8.2.4.) Two concrete parametrizations are used:

- **Two-matrix (Hermitian).** A, B are self-adjoint polynomials in two free semicircular
  generators x₀, x₁; symmetry (Z₂×Z₂: A odd in x₀, even in x₁; exchange: B(x₀,x₁)=A(x₁,x₀))
  is imposed by construction. One fits the ansatz coefficients by minimizing the planar
  loop residual (docs `2026-07-01-two-matrix-validation.md`).
- **Gauge links (unitary).** Û_μ = exp(iĤ_μ) with Ĥ_μ a Hermitian polynomial in
  creation/annihilation operators, Fock truncation L_trunc.

### 2.2 Testbed A — two-matrix model (Kazakov–Zheng)

Kazakov–Zheng, arXiv:2108.04830 eq. 6:
S = N tr[½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]²], Hermitian A, B, large-N, g=h=1
unless stated. m[W] = single-trace normalized moments; the observable family is
m[A^{2k}]. The h=0 limit decouples into two quartic one-matrix models with the exact
anchor m[A²]=a⁴(ga²+1)/16, a²+¾ga⁴=4, giving m[A²]=0.5161 at g=1 (this is the
continuation anchor). The rigorous convex-bootstrap island at cutoff 12 is
m[A²] ∈ [0.4204, 0.4224]; independent HMC (N=48) gives m[A²]=0.42133(36) (`scratch/kz_hmc.py`;
run output not persisted — the length-12 HMC value of §5.1 is, in the vehicle-B doc).

A companion under-determination measurement (§3.1) is run on a related but distinct
model — the massive D=2 Lin–Zheng two-matrix quantum mechanics (arXiv:2507.21007),
H = Tr[½(Π_I²+M²X_I²) − ¼[X₁,X₂]²], M²=1 — because its free-parameter counts (Table I:
14/94/614) give an independent check on the measured variety dimension. This distinction
is noted again at §3.1.

### 2.3 Testbed B — 2D lattice Yang–Mills

SU(N) at N=∞, Wilson action, λ = 't Hooft coupling, strong-coupling phase λ ≥ 1. The
exact single-plaquette expectation is w₊ = 1/(2λ) (Gross–Witten, Phys. Rev. D21 (1980)
446). 2D lattice gauge theory factorizes into independent plaquettes, so for a simple
loop enclosing area A one has ⟨W(C)⟩ = w₊^Area *exactly* — the root of exact 2D
solubility (Migdal, Sov. Phys. JETP 42 (1975) 413; not 't Hooft 1974). A backtracking
spur (U U† = 𝟙) encloses zero area and contributes a factor 1 (Kazakov–Kostov, Nucl.
Phys. B176 (1980) 199). A rigorous 2D master field exists (Lévy, arXiv:1112.2452), which
is why 2D is a validation arena rather than a discovery target. **The area law is used
only as ground truth for scoring; it is never imposed in the loss.**

Conventions in force: large-N ('t Hooft) limit, coupling λ (strong-coupling λ ≥ 1),
Wilson lattice action. These are pinned by the testbed papers cited above.

---

## 3. Under-determination results

### 3.1 Matrix side — the variety dimension is measured

Code `matrix_master_field/bfss/relaxation_selection.py`, data
`results/selection_branches.csv`. On the massive D=2 Lin–Zheng QM (M²=1), strip
positivity, characterize the exact **factorized** loop variety (double-traces reduced to
products of single-trace moments), and measure its dimension as nV − rank(Jacobian):

| level L | nV | # loop eqs | variety dim = nV − rank(J) | positive-E extent (SQP min/max) | point or set |
|---|---|---|---|---|---|
| 4 | 14 | 48 | **3** | [0.86603, 9.056] | SET |
| 6 | 94 | 326 | **8** | [0.95578, 2.970] | SET |
| 8 | 614 | 2118 | **22** | [1.16967, —] | SET |

(L=8 upper edge did not converge in double precision — positivity is weak on the upper
side; honest limit.) The variable counts 14/94/614 match Lin–Zheng arXiv:2507.21007
Table I (validating the assembly); the variety dimensions 3, 8, 22 = nV − rank(J) are
computed here, not read from Table I. Two facts follow:

- **Positivity does not collapse the variety to a point at finite truncation** — it
  leaves a positive-dimensional *set* (a bracket), width 8.19 → 2.01 → shrinking. The
  lower edge (a rigorous lower bound) rises 0.86603 → 0.95578 → 1.16967 toward the
  rigorous island lower edge 1.172098376 (gap ≈ 0.2% at L=8). The physical point lies
  inside the positive set at every level.
- **The relaxation gap is not flatness-controlled at reachable levels.** The moment-matrix
  rank climbs 2 → 12 → 38 through L=4,6,8 with no flat extension
  (`results/relaxation_flatness.csv`), so the truncated moment problem is not yet flat and
  the representing state is not pinned by flatness. "Relax and hope the island is tight" is
  not asymptotically free here; the load-bearing route is the converging bracket edge, i.e.
  level-growth / continuation.

On the KZ model itself (§2.2), the same under-determination shows up in the optimizer
spread. Cold multistart Burer–Monteiro (`matrix_master_field/kz/kz_bm.py`, 10 restarts,
g=1, L=8) samples a *spread* of feasible points m[A²] ∈ [0.395, 0.426] — essentially the
convex bracket width, not a point (reproducible via `kz_bm.py`; the multistart output is
not persisted). With continuation from the solvable h=0 anchor, the optimizers instead
converge to the *neighborhood* of the island m[A²] ∈ [0.4204, 0.4224], straddling it:
moment-ALM 0.42292 and SQP 0.42298 land just above, the operator field 0.41968 just below
(doc `2026-07-01-two-matrix-validation.md`); the Burer–Monteiro continuation value (≈0.418)
is likewise reproducible via `kz_bm.py` but unpersisted.

### 3.2 Gauge side — a spurious machine-zero solution, and two-solution disagreement

Code `cluster/qcd2_area_gate.py`, `cluster/qcd2_area_gate_matfree.py`; data
`results/qcd2_gate_{cold,cont,cont5,mf_val5,seeds}.json`; figure
`results/qcd2_gate_reach.png`. The loss is the exact D=2 lattice loop equation on a loop
set (detour staples evaluated directly; contact terms via base-point factorization
W(plaq)·W(C), a generic N=∞ identity — large-N factorization ⟨tr A·tr B⟩ =
⟨tr A⟩⟨tr B⟩ + O(1/N²), Yaffe Rev. Mod. Phys. 54 (1982) 407 — **not** the area law) plus
cyclicity, reflection
positivity, and B_D symmetry. The area law is not imposed. All runs at λ=2 final
(w₊=0.25) unless noted.

| run | final loss | W_plaq (err) | 2×1 err | 3×1 err | 2×2 err |
|---|---|---|---|---|---|
| cold, L_trunc=4 | 3.7e-05 | 0.2532 (1.3%) | 23% | 182% | 888% |
| continuation, L_trunc=4 | 1.9e-09 | 0.2479 (0.85%) | 15% | 57% | 623% |
| continuation, L_trunc=5 (dense) | 6.8e-16 | 0.2498 (0.08%) | 1.4% | 68% | 84% |
| continuation, L_trunc=5 (matrix-free) | 1.3e-18 | 0.2493 (0.27%) | 4.9% | 78% | 173% |

Two readings establish under-determination on the gauge side:

**(a) Machine-zero residual does not imply the master field.** A minimal-constraint cold
run (data `results/qcd2_gate_smoke.json`) drives the loop-equation residual to 1.3e-14 on
an *unphysical* solution — negative Wilson loops, and 2×1 ≠ 1×2 (rotational symmetry
violated). Zero truncated residual is necessary but not sufficient.

**(b) Two independent machine-zero solutions disagree at large area.** The dense and
matrix-free continuation runs (last two table rows) are two independent machine-zero
solutions of the *same* objective. They agree on the plaquette (≈0.2%) and area-2
(≈4% spread) but diverge wildly at area ≥ 3. The run-to-run spread *is* the reach — this
is the truth-free diagnostic of §6.

---

## 4. Selection results (continuation)

### 4.1 Continuation selects the physical branch

- **Matrix side.** Cold start of the operator field at test-word cutoff W=2 drives the
  loop residual to ~1e-31 yet lands on a spurious solution m[A²]=0.343. Ramping the
  commutator coupling h: 0→1 with warm starts from the exact h=0 anchor tracks the
  physical branch monotonically to m[A²]=0.41898 at W=2 and 0.41968 at W=3 (dim 2047) —
  the same-cutoff W=2 comparison (cold 0.343 vs continuation 0.41898) isolates the
  continuation effect from the W=2→3 change — landing ≈0.17% below the certified island
  lower bound, in seconds (doc `2026-07-01-two-matrix-validation.md`).
  min/max bracketing (ALM) does *not* select: extremizing m[A²] over the whole feasible
  set returns the spurious collapsed extremes ([0, 0.618]), because m[A²]=0 is feasible
  when a large m[A⁴] balances the loop equation. Continuation ignores the extremes and
  follows the saddle.
- **Gauge side.** Continuation reaches ≈4 orders deeper in residual than cold at fixed
  L_trunc=4 (1.9e-09 vs 3.7e-05), and deeper still as the truncation grows (6.8e-16 at
  L_trunc=5); the plaquette error improves from 1.3% (cold) to 0.08–0.27% (continuation).

### 4.2 Five-seed sweep and the nuance

Data `results/qcd2_gate_seeds.json`, L_trunc=4:

- Continuation reaches 2–6 orders deeper residual than cold in 5/5 seeds
  (cont 1.1e-5 → 6.1e-9 vs cold 6.3e-4 → 3.7e-3).
- W_plaq is robust to ≈1% in ALL 10 runs.
- Area-2 is systematically −15% biased at L_trunc=4 under continuation, improving to −3±2%
  at L_trunc=5 — a truncation bias, not scatter.
- Area ≥ 3 is unpinned in both protocols.

**Nuance (must appear).** With the FULL constraint set, cold start is *not* catastrophic:
it lands in the physical neighborhood but converges orders of magnitude shallower. The
spurious-branch catastrophe of §3.2(a) appears when the constraint set is *weakened*.
Selection is done JOINTLY by the constraint set and continuation — neither alone.

---

## 5. Reach and cost

### 5.1 The provably-wrong long moment (matrix side)

Code `matrix_master_field/kz/opfield_general.py`, `kz_large_observables.py`; data
`results/kz_reach_vs_degree.csv`. The reliable reach is ≈ 2W where W is the
loop-equation cutoff; the basis degree is NOT the knob (degree-7 at W=3 does not help,
and is in fact worse at high moments). At W=5 the operator field lands inside every
rigorous bootstrap bracket through m[A¹⁰]. At length 12 it fails:

- master field m[A¹²] = 0.4932
- rigorous MOSEK bracket at cutoff 12: [0.4598, 0.4795] (52 s solve)
- independent HMC (N=48) truth: m[A¹²] = 0.46574(267), inside the bracket.

The master field value is above the rigorous bracket — **provably wrong**. HMC also
validates the island: m[A²] = 0.42133(36).

**Caveat (must appear).** An earlier claim that the bootstrap "cannot reach length 12" was
an artifact of hardcoded cutoffs (`cutoffs=[4,6,8,10]` in `kz_large_observables.py`) —
corrected 2026-07-03 (correction block in doc
`2026-07-01-vehicleB-large-observables.md`). With the cutoff raised to 12 the bootstrap
brackets m[A¹²] as above, and the master field does not beat it — it matches it up to 2W,
then fails.

**Caveat (must appear).** For GAUGE theory, "large" means enclosed AREA, not word length.
A backtrack collapses (U U† = 𝟙 → enclosed area 0 → factor 1; Kazakov–Kostov NPB176
(1980) 199), so the Hermitian long-word result does NOT transfer to Wilson loops. This is
why the reach test is repeated on the gauge side (§5.2) along the area axis.

### 5.2 Directional cutoff result (gauge side)

The gauge reach question — does adding a loop's own equation to the constraint set pull
that loop's value toward truth? — has **directional** support, a well-posedness check, and
one failed attempt at a fully-converged higher-cutoff point. Three runs, kept separate:

**(a) Directional evidence (L_trunc=6, all base edges).** Data
`results/qcd2_gate_reach6_salvage.log` (the λ=3 stage of a run killed at ~85%; 8-loop set
through area 9, `all_edges=True`, rp_cut=3). Once the 3×1 loop's own equation is inside
the loss, its area-3 error is **6.73%** (model 0.00432 vs exact 0.00463), versus 57–182%
in the area-2-cutoff runs where 3×1 sits *outside* the constraint set (§3.2 table). That is
directional evidence that reach follows the loop cutoff. But the run is under-converged
(loss 1.5e-5; the area-2 loops themselves degraded to 22–30% and larger loops went
negative), and it mixes cutoff-raising with truncation-raising (L4/L5 unconstrained →
L6 constrained), so it is **directional only**.

**(b) Well-posedness at L_trunc=5 (extended loop set).** Data
`results/qcd2_gate_scanhi.json`. The same continuation with the loop set extended to area 6
converges cleanly at L_trunc=5 (final loss 9.3e-8, plaquette 0.24934 / 0.26% off), showing
the extended-loop-set optimization is well-posed at L5. Per-loop area errors were not
persisted for this run, so it certifies convergence, not a specific area-3 number.

**(c) The clean L_trunc=6 measurement was attempted and is INVALID.** Data
`results/qcd2_gate_mf_reach6lite.json`, `results/qcd2_gate_reach6lite.log`. A matrix-free,
cost-reduced continuation (`edge_mode="reps"`, 2 edges/loop) ran to completion (34.6 h
wall) and reached loss 5.7e-8 at λ=2 — but on a numerically corrupted, unphysical field,
flagged by the run's own diagnostics:

- **‖H‖ left the expm's validity window.** The final Hermitian generators have
  ‖H‖ ≈ 10.35 and 10.33, far outside the order-30 Taylor `expm_iH_v` range (‖H‖ ≲ 3–5);
  the links are then non-unitary (|‖u‖−1| ≈ 7e-4) and the Taylor tail is unconverged
  (≈ 6e-3). Both directions carry a `[WARN]`.
- **The Fock cross-check fails 40×.** The same parameters give plaquette **0.2385 at
  L_trunc=6 but 0.0059 at L_trunc=7** (`W_plaq_gap = 0.233`). A genuine master field is
  cutoff-independent; a 40× swing means this field is not Fock-converged.
- **Unphysical loop spectrum.** At λ=2: 2×1 and 1×2 ≈ 80% off, 3×1 94% off, and
  2×2/3×2/3×3 come out *negative* — inconsistent with the known-positive 2D
  strong-coupling area-law values (every w₊^Area > 0).

The low residual was reached on a large-‖H‖, Fock-unconverged spurious branch. Leading
hypothesis for the cause: the `reps` edge reduction (12 residuals vs 40 all-edges)
under-constrained the fit, so as λ fell (links moving far from identity, needing larger
‖H‖) the optimizer was free to escape the numerically-valid envelope; the all-edges L5
runs (§3.2) never did. This is not fully isolated from the low-λ effect. The point of
record: the ‖H‖/expm-validity monitor and the L→L+1 Fock cross-check — both added in the
adversarial audit — caught it; without them the run would have been misreported as the
converged L6 result.

**Net.** No converged higher-cutoff gauge point is in hand. The reach ≈ cutoff law is
*demonstrated* on the matrix side (§5.1) and supported *directionally* on the gauge side
(§5.2a, the salvage λ=3). Whether the L_trunc=6 barrier is a genuine Fock-horizon limit
or an artifact of the expm radius plus the `reps` constraint reduction is **undetermined**
— the invalid run (§5.2c) cannot distinguish them, because both would present as a
low-residual field that fails the Fock cross-check. Settling it requires the hardened-expm,
all-edges rerun described in §8.2.4; until then the gauge reach claim stays directional.

### 5.3 Cost (measured, single Apple laptop, CPU, JAX)

- Plaquette-only: seconds.
- Area-2 (L_trunc=5): ≈10 h dense; ≈15 h matrix-free with per-mode-once XLA compile
  (compile 208 s to 27 min depending on RP size; λ traced ⇒ one compile per mode).
- Area-3 attempt (L_trunc=6, 8 loops, all base edges): 33 s/step ⇒ ≈3 days, killed at 85%
  (λ=3 stage salvaged, §5.2a). A cheaper matrix-free reps-edge retry completed in 34.6 h but
  was invalidated by the ‖H‖/Fock diagnostics (§5.2c) — a converged L_trunc=6 point needs a
  hardened expm and stays a multi-day run.

Reach grows with the cutoff; cost grows roughly an order of magnitude per unit of
area-reach — comparable order-of-magnitude growth to the bootstrap's (on the two-to-three
cost points measured, the area-3 point among them not converged). The convex bootstrap
brackets
m[A¹²] in 52 s at cutoff 12 on the matrix side, and on the gauge side rigorously brackets
only the plaquette (Kazakov–Zheng arXiv:2203.11360, L_max ≤ 16). State plainly: **the
master field does not beat the bootstrap's scaling.** Its advantages are (i) points, not
brackets, within reach; (ii) one configuration → any observable; (iii) exact
positivity/traciality (for the ideal operator field; numerically envelope-limited for the
unitary realization, §8.2.4); (iv) branch selection where the bootstrap's relaxation cannot
distinguish branches.

---

## 6. A truth-free reach diagnostic

Where an exact answer exists, reliable reach is read off directly (the largest observable
still inside the rigorous bracket). Where none exists — the D=3 target — we propose using
**agreement across independent machine-zero solutions of the same objective** as a proxy.

The gauge two-solution comparison of §3.2(b) is the prototype: the dense and matrix-free
continuation runs agree on the plaquette to ≈0.2% and on area-2 to a ≈4% spread, and
disagree by hundreds of percent at area ≥ 3. Independently of any ground truth, this
locates the reliable horizon at area ≈ 2 for that cutoff. The construction: run two (or
more) independent solvers/parametrizations that reach machine-zero residual on the same
constraint set; trust an observable only out to the size where they still agree to
target tolerance. This is usable precisely where the bootstrap gives no bracket and no MC
is available.

**Caveat.** This is *proposed and illustrated once*, not validated. Crucially, cross-solver
agreement bounds solver *scatter*, not *shared* truncation bias: two independent solutions
of the same truncated objective can agree while both carry the same systematic error —
precisely the area-2 −15% bias of §4.2, which two agreeing solutions would report as
"reliable." The diagnostic flags where solutions diverge; it does not certify where they
agree. Validating it against a case with known shared bias is future work.

---

## 7. Related work and novelty

**Not novel (stated plainly):**

- The Cuntz/Fock operator representation of the master field (Gopakumar–Gross
  hep-th/9411021; Douglas hep-th/9411025).
- Loop-residual minimization to construct the master field (Maeta arXiv:2605.10720 —
  one/two-matrix, finite-dim regularization, no positivity; de Mello Koch et al.
  arXiv:2108.08803, arXiv:2306.00935 — collective variational + SDP; Engelhardt–Levit
  hep-th/9609216).
- The "construction gives points where the bootstrap gives bounds" framing (Maeta
  arXiv:2605.10720).

**Novel / defensible (claimed narrowly):**

1. **A quantified measurement of the under-determination** — the loop-variety dimensions
   3/8/22 = nV − rank(J) computed here, and an explicit spurious machine-zero *gauge*
   solution. The *concept* of non-uniqueness is not new (Maeta arXiv:2605.10720 documents
   it — his Table 2: 3/15 random seeds physical); what is added is the dimension count and
   the gauge instance.
2. **Continuation from a *solvable* limit as the branch selector** — clean on the matrix
   side (cold → spurious 0.343 vs continuation → physical); on the gauge side it is *joint*
   with the constraint set (§4.2) and the anchor is only half-cold (§8.2.2), so the gauge
   claim is partial. Tracking from a known-solvable point differs from Maeta's random-seed
   initialization heuristic, but we are not aware of a prior branch-selection claim to weigh
   it against.
3. **A truth-free reach proxy (proposed)** — cross-solver agreement as a reach estimate
   where no bracket or MC exists (§6); proposed and illustrated once, with the shared-bias
   caveat noted there.

Two further items are honest *findings*, not novelties: the reflection-positivity /
truncation rule 2·rp_cut ≤ L_trunc (an implementation constraint, §8.1), and the reach/cost
characterization including the provably-wrong length-12 prediction caught by bootstrap + HMC
(§5.1) — which re-confirms, rather than overturns, the cutoff-limited character these
constructions are already understood to have.

---

## 8. Limitations and outlook

### 8.1 The reflection-positivity truncation rule

A failed sweep produced a reusable rule. Reflection-positivity paths of length ℓ create
Gram entries equal to Wilson values of words of length 2ℓ; if 2ℓ > L_trunc the Gram is
truncation-corrupted and its negative eigenvalues cannot be optimized away (an
un-optimizable loss floor, measured ≈5e-2 across the rp=4, L_trunc=4 seed runs; the run
logs are not persisted). **Rule: 2·rp_cut ≤ L_trunc.** Corollary: RP can
only protect loops up to half-perimeter L_trunc/2; negativity of larger loops is part of
the reach limit.

### 8.2 Limitations (verbatim content required)

1. **The gauge loss equation is verified but partly reverse-engineered.** The exact D=2
   lattice loop equation used in the gauge loss was verified machine-zero against the exact
   area-law solution (all 324 simple loops L≤12, every base edge, λ≥1;
   `cuntz_bootstrap/lattice_loop_eq.py`) and reduces to the known plaquette equation — but
   its contact/staple structure was partly reverse-engineered from that solution. A
   first-principles derivation from the Wilson action (or transcription from
   Anderson–Kruczenski arXiv:1612.08140) is still open. So the gate demonstrates SELECTION
   among solutions of a constraint set known to admit the physical solution — not an
   ab-initio derivation of the area law.
2. **The anchor is only half a genuine anchor.** The "continuation anchor" at λ=8 starts
   from near-identity links (weak-coupling-like init), so the first stage is effectively a
   cold solve at strong coupling; only the subsequent λ-sweep is genuine branch tracking. A
   cleaner anchor (free-Haar λ→∞ or the exact single-plaquette Gross–Witten field) is
   future work.
3. **Indicative, not optimized, and 2D is validation not discovery.** Single machine, JAX
   CPU; wall-times are indicative not optimized. 2D is a validation arena (exact answers;
   rigorous master field exists — Lévy arXiv:1112.2452); the discovery arena is D=3, where
   the same dimension-general loop equation applies and the referees are the SU(∞)
   plaquette bootstrap (arXiv:2203.11360) and TEK Monte Carlo with √N-scaled twist
   (González-Arroyo–Okawa arXiv:1005.1981, arXiv:1410.6405).
4. **The unitary parametrization has a numerical validity envelope that binds at
   L_trunc ≥ 6.** Û = exp(iĤ) is evaluated with an order-30 Taylor `expm_iH_v`, accurate
   only for ‖Ĥ‖ ≲ 3–5. The required ‖Ĥ‖ grows as λ decreases (links moving away from
   identity), and at L_trunc=6, λ=2 it reaches ≈10 — outside both the expm radius and the
   Fock horizon (§5.2c). Two consequences: (i) reaching L_trunc ≥ 6 at strong coupling needs
   a hardened expm (scaling-and-squaring) plus an L→L+1 Fock-convergence acceptance gate,
   not just more steps; (ii) the `edge_mode="reps"` constraint reduction is **unsafe** — it
   under-constrains the fit enough to let the optimizer escape into the large-‖Ĥ‖ spurious
   region while still driving the reduced residual to ~1e-8. The audit-added ‖Ĥ‖ monitor and
   Fock cross-check are what caught this; they are now mandatory acceptance checks, not
   diagnostics.

### 8.3 Outlook — the D=3 program

The validated vehicle is the matrix-free operator master field plus continuation. The
D=3 plan: continue from the Gross–Witten / QCD₂ anchor into D=3 using the same
dimension-general lattice loop equation; measure reliable reach along the area axis via
the two-solution diagnostic (§6); and cross-check against the SU(∞) plaquette bootstrap
(arXiv:2203.11360) and TEK Monte Carlo (arXiv:1005.1981, arXiv:1410.6405), which are the
only available referees at N=∞ in D=3. The honest expectation, set by §5, is that reach
is bounded by the affordable area cutoff and that the deliverable is a *point* per
observable within that reach, not a bound and not unbounded extrapolation.

---

## 9. Repository cross-reference

- §3.1 variety dimensions: `matrix_master_field/bfss/relaxation_selection.py`,
  `results/selection_branches.csv`, `results/relaxation_flatness.csv`; doc
  `docs/superpowers/results/2026-07-01-relaxation-selection.md`.
- §3.1 KZ optimizer spread / island: `matrix_master_field/kz/kz_bm.py`,
  `kz_moment_methods.py`, `kz_sqp.py`; doc
  `docs/superpowers/results/2026-07-01-two-matrix-validation.md`.
- §3.2, §4.2, §5.2, §5.3, §8.1 gauge runs: `cluster/qcd2_area_gate.py`,
  `cluster/qcd2_area_gate_matfree.py`;
  `results/qcd2_gate_{cold,cont,cont5,mf_val5,seeds,scanhi,smoke}.json`;
  `results/qcd2_gate_reach6_salvage.log` (§5.2a), `results/qcd2_gate_mf_reach6lite.json`
  with `results/qcd2_gate_reach6lite.log` (§5.2c, invalid run);
  figure `results/qcd2_gate_reach.png`.
- §5.1 large-observable reach: `matrix_master_field/kz/opfield_general.py`,
  `kz_large_observables.py`; `results/kz_reach_vs_degree.csv`,
  `results/kz_large_observables.csv`; doc
  `docs/superpowers/results/2026-07-01-vehicleB-large-observables.md`.
- Verified lattice loop equation: `cuntz_bootstrap/lattice_loop_eq.py`; doc
  `docs/superpowers/results/2026-07-01-lattice-loop-equation.md`.
- Citations: `REFERENCES.bib`.
```
