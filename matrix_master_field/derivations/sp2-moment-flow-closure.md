# SP2 — Moment-flow closure (S1) and truncation rigor (S2): go/no-go for the pure-large-N master field

**Date:** 2026-06-27
**Branch:** `matrix-master-field`
**Verification script:** `matrix_master_field/derivations/sp2_flow_test.py`
(run: `uv run --no-project --with numpy --with scipy python sp2_flow_test.py`; all five tests pass)
**Numerics reuse:** the bosonic-Fock construction (Hermitian mode basis + occupation-truncated
ladder operators) of `matrix_master_field/exact_diag.py` and `matrix_master_field/qm_fock.py`.

> **Scope.** This file settles the two seams an adversarial audit flagged in the variational
> master-field design `|ψ⟩=U|G⟩`, `U=exp(iÂ)`, Â a Hermitian single-trace polynomial, energy via a
> large-N moment-flow:
> - **(S1) Closure** — does the single-trace moment-flow close at large N?
> - **(S2) Truncation rigor** — is the L-truncated flow energy a rigorous bound?
>
> Verdict up front: **S1 closes** (single-trace sector, infinite L-truncated hierarchy, closed by
> large-N factorization — DERIVED and verified to machine precision). **S2 fails** for the pure-large-N
> truncated flow (not a certifiable bound; biased *from below*, M5c-style). **Recommendation: (B)
> finite-N staging.**

---

## Conventions (pinned — used verbatim, not re-guessed)

- Model (HHK arXiv:2004.10212, Eq. 17): `H = Tr(P_X² + P_Y² + m²(X²+Y²) − g²[X,Y]²)`, `X,Y` Hermitian
  `N×N`, `ℏ=1`, `m=1`. Matrix CCR `[X_ij, P_X,kl] = i δ_il δ_jk` — the **matrix** commutator `[X,P]` is
  an OPERATOR, not a c-number `iN·𝟙`; only its trace is fixed.
- 't Hooft scaled variables: `X̃=X/√N`, `P̃=P/√N` ⟹ `[X̃_ab, P̃_cd] = (i/N) δ_ad δ_bc`. Normalized
  trace `τ(·)=(1/N)Tr(·)`; single-trace moments `m[w]=lim_{N→∞} τ(w)`. Coupling `λ=Ng²` fixed.
- Heisenberg flow of the trial: `O(s)=e^{-isÂ} O e^{isÂ}`, so `dO/ds = i[O,Â]`, integrated `s:0→1`,
  evaluated in the fixed Gaussian reference `|G⟩` (the `g=0` ground state of `Tr(P̃²+X̃²)`, frequency `m=1`).

All numerics below are in the **single-Hermitian-matrix sector** (`X̃`, `P̃`; words over `{0=X̃,1=P̃}`),
which is sufficient to exercise the matrix CCR, the generator scaling, closure, factorization, and
truncation. The two-matrix structure adds only the `Ỹ` sector and the commutator interaction; neither
changes the closure mechanism (the interaction `−λτ([X̃,Ỹ]²)` is a single-trace observable read off at
`s=1`, and `[X̃,P̃_X]`, `[Ỹ,P̃_Y]` are independent canonical pairs).

---

## Part 1 — Pinning the algebra (fixes the audit's C1/C2/C4/I2)

### 1.1 The scaled CCR with indices, and `τ([X̃,P̃_X])=i`

From `[X_ab,P_{X,cd}]=iδ_ad δ_bc` and `X̃=X/√N`, `P̃=P/√N`:

```
[X̃_ab, P̃_{X,cd}] = (1/N)[X_ab,P_{X,cd}] = (i/N) δ_ad δ_bc.                          (1)
```

Trace of the matrix commutator (an operator identity; the entries are operators):

```
Tr[X̃,P̃_X] = Σ_a [X̃,P̃_X]_aa = Σ_{a,k} (X̃_ak P̃_{X,ka} − P̃_{X,ak} X̃_ka)
           = Σ_{a,k} [X̃_ak, P̃_{X,ka}] = Σ_{a,k} (i/N) δ_aa δ_kk            (a=a,b=k,c=k,d=a)
           = (i/N) Σ_{a,k} 1 = (i/N) N² = iN.                                          (2)
```

Hence `τ([X̃,P̃_X]) = (1/N)·iN = i.` ✓ (Cross-check, unscaled: `Tr[X,P]=Σ_{ak}iδ_aaδ_kk=iN²`, and
`Tr[X̃,P̃]=(1/N)Tr[X,P]=iN`, consistent.) **Numeric:** `sp2_flow_test.py` measures
`τ(X̃P̃)−τ(P̃X̃)=+1.0000i` **exactly at finite N** (Test E header / design probe), for N=2,3,4 — this is the
matrix CCR / Gauss-law content, exact at every `N`, not a large-N statement.

### 1.2 No cyclicity for words containing momenta — the correction term

Trace cyclicity `Tr(O₁O₂)=Tr(O₂O₁)` is an identity for *c-number* entries. For operator-valued entries it
**fails**, and the failure is exactly the canonical commutator:

```
Tr(O₁O₂) − Tr(O₂O₁) = Σ_{ij} [ (O₁)_ij (O₂)_ji − (O₂)_ij (O₁)_ji ]
                     = Σ_{ij} [ (O₁)_ij , (O₂)_ji ].                                    (3)
```

With `O₁=X̃`, `O₂=P̃_X`: `Σ_{ij}[X̃_ij,P̃_{X,ji}] = Σ_{ij}(i/N)δ_ijδ_ij = (i/N)N = i`, i.e.
`τ(X̃P̃_X) − τ(P̃_X X̃) = i` (the `+i/2`, `−i/2` two-point split of `tm_qm_relations.py`). **Rule:** a cyclic
move that carries a letter `a` across a non-commuting letter `b` picks up `Σ_{ij}[a_ij,b_ji]`; for adjacent
canonical letters this is `±i`. The verification script therefore **never uses cyclicity** — every word is
multiplied in order along the matrix index (`word_mat`), with operator (Fock) products kept ordered.

### 1.3 Generator normalization: Â must carry an explicit factor of N

The audit noted single-trace commutators are `O(1/N)`; we derive the explicit power. Take Â `= κ·Tr(â)`
for a single-trace word `â` and coupling `κ`. The flow of a normalized moment is

```
dτ(w)/ds = i⟨[τ(w), Â]⟩ = i (κ/N) ⟨[Tr w, Tr â]⟩.                                      (4)
```

**Lemma (planar join).** For one canonical contraction between `Tr(u X̃ u')` and `Tr(v P̃ v')`,

```
[Tr(uX̃u'), Tr(vP̃v')]│_one-contraction = (i/N) Tr(u'u v'v).                            (★)
```

*Proof.* `Tr(uX̃u')=Σ_{iab}u_{ia}X̃_{ab}u'_{bi}`, `Tr(vP̃v')=Σ_{jcd}v_{jc}P̃_{cd}v'_{dj}`; the only
non-commuting pieces are `X̃,P̃`, with `[X̃_ab,P̃_cd]=(i/N)δ_adδ_bc`. Summing,
`(i/N)Σ u_{ia}u'_{bi}v_{jb}v'_{aj} = (i/N)Σ_{ab}(u'u)_{ba}(v'v)_{ab} = (i/N)Tr(u'uv'v)`. ∎

Because `(★)` produces **one** trace, `⟨[Tr w, Tr â]⟩ = (i/N)⟨Tr(join)⟩ = (i/N)(N·m[join]) = i·m[join] = O(1)`.
Substituting into (4): `dτ(w)/ds = i(κ/N)·O(1)`. For this to be `O(1)` we need **κ = N**:

```
Â = N · Tr(â) = N² · τ(â).                                                              (5)
```

This is the 't Hooft normalization of a single-trace deformation (coupling `~N`, like the action `S=N Tr[…]`).

**Numeric (Test E), decisive.** With `Â=N Tr(â)`, `dτ(w)/ds│₀ = i⟨[Tr w, Tr â]⟩` is `O(1)` AND
**N-independent** to `~1e-14` for every probed `(w,â)` across `N=2,3,4,5`:

| `(w; â)` | N=2 | N=3 | N=4 | N=5 | planar |
|---|---|---|---|---|---|
| `(X̃; P̃)` | −1.0000 | −1.0000 | −1.0000 | −1.0000 | −1 |
| `(P̃; X̃³)` | +1.5000 | +1.5000 | +1.5000 | +1.5000 | +3/2 |
| `(X̃²; X̃P̃)` | −1.0000 | −1.0000 | −1.0000 | −1.0000 | −1 |
| `(P̃²; X̃P̃)` | +1.0000 | +1.0000 | +1.0000 | +1.0000 | +1 |

(With the *un*-scaled `Â=Tr(â)`, the same derivative scales as `1/N` → 0; that mis-scaling was an artifact
of an early probe, now corrected.)

---

## Part 2 — The planar moment-flow for concrete generators, and the closure structure

We work two genuine non-trivial cases (`w ≠ Â`), one linear and one nonlinear, deriving the matrix EOM
rigorously (finite N, exact), then the moment-flow.

### 2.1 Test A — `Â = N Tr(X̃³)` (position-only; exact affine flow)

`[X̃_ab, Tr X̃³]=0`, so **`dX̃/ds=0`** (X̃ frozen). For `P̃`, with `[P̃_ab,X̃_cd]=−(i/N)δ_cbδ_da`:

```
[P̃_ab, Tr X̃³] = Σ_{pqr}([P̃_ab,X̃_pq]X̃_qrX̃_rp + X̃_pq[P̃_ab,X̃_qr]X̃_rp + X̃_pqX̃_qr[P̃_ab,X̃_rp])
              = −(i/N)·3 (X̃²)_ab,    ⟹   dP̃_ab/ds = iN·(−(i/N)·3(X̃²)_ab) = 3(X̃²)_ab.            (6)
```

So `P̃(s) = P̃₀ + 3s X̃²` (matrix, exact at finite N — the analog of `p→p+s V'(x)`, `V=x³`). Therefore

```
τ(P̃²)(s) = τ(P̃₀²) + 3s[τ(P̃X̃²)+τ(X̃²P̃)] + 9s² τ(X̃⁴).                                  (7)
```

This is **closed**, **purely single-trace**, and (because the substitution is affine and `Tr(P̃²)` is only
quadratic in `P̃`) **exact at every N** — there is no factorization correction. It validates the generator
normalization (5), the sign, and the machinery.

**Numeric (Test A + Test D).** The exact Fock flow matches (7) up to a pure Fock-truncation error that
vanishes in `K`:

| N | K | D | `max|exact τ(P̃²) − ODE(7)|` |
|---|---|---|---|
| 2 | 4 | 70 | 2.2e−2 |
| 2 | 6 | 210 | 2.6e−3 |
| 2 | 10 | 1001 | 3.8e−4 |
| 3 | 4 | 715 | 2.9e−2 |
| 3 | 5 | 2002 | 1.4e−2 |
| 3 | 6 | 5005 | 4.7e−3 |

The ODE value is K-independent; the residual is monotone-decreasing truncation error (X̃ "frozen" check:
`|Δτ(X̃²)|` falls from `7.7e−3` at N=3,K=4 toward 0 as K grows). **The gap is truncation, not a closure
failure.**

### 2.2 Test B — `Â = (N/2) Tr(X̃²P̃ + P̃X̃²)` (nonlinear canonical flow)

`[X̃_ab, Tr(X̃²P̃)] = (i/N)(X̃²)_ab` (only `P̃` contracts), and likewise for `Tr(P̃X̃²)`, so

```
dX̃_ab/ds = i[X̃_ab,Â] = i·(N/2)·2·(i/N)(X̃²)_ab = −(X̃²)_ab     ⟹     dX̃/ds = −X̃²       (8)
```

a **matrix Riccati equation**, `X̃(s)=X̃₀(1+sX̃₀)⁻¹` — a genuinely nonlinear canonical flow. The pure-X̃
moment hierarchy is

```
dτ(X̃^k)/ds = (k/N)Tr(X̃^{k-1}·(−X̃²)) = −k τ(X̃^{k+1}),                                  (9)
```

closed on single traces, **linear in the moments**, raising word length by one (an infinite,
L-truncated hierarchy). **Numeric (Test B):** integrating (9) (RK4, `kmax=12`) reproduces the exact
finite-N flow of `τ(X̃²)`: `max|exact−ODE| = 7.7e−4` (N=2,K=6) → `6.2e−5` (N=2,K=8); `1.9e−3` (N=3,K=4).

### 2.3 Where double-traces arise, and how closure works (the crux of S1)

At the level of the operator identity `dτ(w)/ds=i⟨[Tr w, Tr â]⟩`, a **single** CCR firing gives one trace
(★). A **second simultaneous** firing (`â` of degree ≥2 meeting ≥2 letters of `w`) can leave **two** index
loops, producing a genuine **double-trace operator** `Tr(u)Tr(v)` at order `(i/N)²·(N²)·(τ·τ) = O(1)`. So
the flow RHS is, in general,

```
dm[w]/ds = Σ_r (single-trace m[join_r])  +  Σ_{r'} (double-trace ⟨τ(u_{r'}) τ(v_{r'})⟩).   (10)
```

**Closure holds because of large-N factorization:**

```
⟨τ(u) τ(v)⟩ = ⟨τ(u)⟩ ⟨τ(v)⟩ + O(1/N²).                                                 (11)
```

So the double-trace expectation factorizes into a **product of single-trace moments**, and (10) closes on
single-trace moments (and their products). The hierarchy is therefore **closed on the single-trace sector**,
generally **nonlinear** (because of the factorized products), and **infinite** in word length (truncated at
L). This is the precise content of S1.

> The audit's "concrete counterexample" `[Tr(XPX), Tr(XPX)]` is `[W,W]=0` (anything commutes with itself)
> and tests nothing. The genuine tests above (`w ≠ Â`) show closure explicitly.

**Numeric (Test C), decisive — the closure mechanism measured directly.** Connected double traces in `|G⟩`,
`C_N(w₁,w₂)=⟨τ(w₁)τ(w₂)⟩−⟨τ(w₁)⟩⟨τ(w₂)⟩`, scale as `1/N²`:

| `C_N · N²` | `τ(X̃²)τ(X̃²)` | `τ(X̃²)τ(X̃⁴)` | `τ(P̃²)τ(P̃²)` |
|---|---|---|---|
| N=2 | +0.500 | +1.125 | +0.500 |
| N=3 | +0.500 | +1.056 | +0.500 |
| N=4 | +0.500 | +1.031 | +0.500 |

`C_N·N²` is **flat** (`τ(X̃²)τ(X̃²)` and `τ(P̃²)τ(P̃²)` are exactly `0.500` for all N; `τ(X̃²)τ(X̃⁴)` →
`~1.0` from above, the residual being K-truncation at the smaller K for N=3,4). Hence `C_N = O(1/N²)`,
i.e. (11) holds with the textbook `1/N²` rate. **This is what makes the moment-flow RHS close.**

### 2.4 Literature support (with specifics)

- **Factorization (11) and the closed loop equations.** That connected multi-trace ("multi-loop")
  correlators are `1/N²`-suppressed, so the Schwinger–Dyson / loop equations close on single-trace
  ("single-loop") correlators, is the content of the Makeenko–Migdal loop equations: Makeenko–Migdal,
  *Exact equation for the loop average in multicolor QCD*, Phys. Lett. B **88** (1979) 135. Textbook
  treatment of large-N factorization `⟨W₁W₂⟩=⟨W₁⟩⟨W₂⟩+O(1/N²)` and the resulting closed loop equations:
  Makeenko, *Methods of Contemporary Gauge Theory* (CUP 2002), large-N / loop-equation chapters. Our
  Test C is a direct numerical instance of this factorization (the `C_N=O(1/N²)` rate).
- **Closure on the single-trace sector in matrix QM.** The collective-field Hamiltonian of matrix quantum
  mechanics is written entirely in terms of the single-trace (loop / density) variables at large N:
  Jevicki–Sakita, *The quantum collective field method and its application to the planar limit*, Nucl.
  Phys. B **165** (1980) 511; Jevicki–Sakita, Nucl. Phys. B **185** (1981) 89. This is the same
  single-trace closure exploited by `tm_qm_relations.py` (the T5 loop equations / T2 Gauss law, verified
  to `<1e-10` on the `g=0` Gaussian).
- **Conjugate-momentum / time-dependent matrix QM closing on single traces.** The Heisenberg evolution of
  single-trace operators in matrix QM stays (at large N) in the single-trace sector — the basis of the
  collective-field and of the bootstrap loop equations already used here (`tm_qm_relations.py`, T5/T2,
  verified to `<1e-10` on the `g=0` Gaussian).

**S1 verdict:** the single-trace moment-flow **closes** at large N — on the single-trace sector, as an
infinite (L-truncated), generally nonlinear hierarchy, closed by factorization (11). Established by
derivation (Part 1–2) and verified to machine precision (Tests A,B,C,D,E).

---

## Part 3 — S2: is the L-truncated flow energy a rigorous bound?

This is the deep seam, and it is where the design **fails**.

### 3.1 Two different objects

- **(a) `|ψ⟩=e^{iÂ}|G⟩` at fixed finite N is a literal state** (U unitary, `|G⟩` unit-norm). Its energy
  `⟨ψ|H|ψ⟩` is, by Rayleigh–Ritz, an **exact variational upper bound** on `E₀(N)`. *No truncation issue
  here* — this is option (B).
- **(b) The design does NOT compute `⟨ψ|H|ψ⟩`.** It computes the **large-N** moment-flow with the hierarchy
  **truncated at word length L**. Truncation does two damaging things: (i) it **drops** the longer
  single-trace moments the flow feeds into — the hierarchies (7),(9),(10) raise word length, so a length-L
  cut discards real contributions; and (ii) it uses **factorization (11)**, exact only at `N=∞`. The
  resulting number is **not** `⟨ψ|H|ψ⟩` for any actual `ψ`.

### 3.2 Can bootstrap positivity certify "these truncated moments are a state"?

The M5b/c bootstrap (`bootstrap_two_matrix_qm` in `bootstrap_sdp.py`) imposes moment-matrix Gram PSD +
stationarity loop equations + SU(N) Gauss law and **minimizes** `E/N²`. This is the right machinery but it
delivers the **opposite** of what the design needs:

- The feasible set {PSD + linear constraints} is a **relaxation** (a superset) of genuine-state moment
  sequences. Minimizing over it gives `min ≤ E₀` — a **certified LOWER bound**. (Verified: with a trusted
  CLARABEL solver, `bootstrap_two_matrix_qm(m=1,λ=0,L=4)=2.000` exact; `λ=1,L=4` gives `≥2.000`, a *loose*
  bound — the `L=4`/`L=6` relaxation is flat at the free value `2m`, as MEMORY records.)
- PSD-feasibility certifies only "**consistent with some state** up to the imposed constraints", never
  "**is** the moment sequence of `e^{iÂ}|G⟩`", and never that the truncated-flow value is `≥ E₀`. To get an
  **upper** bound you need an **actual** state and its **exact** energy — which is (a), i.e. finite N.

**Crucial asymmetry:** a moment-matrix relaxation is intrinsically **one-sided (lower)**. It cannot convert
an approximately-feasible trial into a rigorous **upper** bound. So positivity gives the *lower* side of the
sandwich (which does not even use the master-field trial — it is the SDP we already have), and the
pure-large-N **trial** energy from the truncated flow remains **uncertifiable** as a bound. "Certify these
L-truncated moments are exactly some state's moments" is, in the worst case, as hard as the original
problem (it is the truncated moment problem plus the dynamics), and even when feasible it yields the
relaxation value (lower side), not the trial's upper bound.

### 3.3 Bias direction of the truncation error — unsafe under minimization

**Correction (controller re-verification).** The SP2 design computes `⟨P̃²⟩` **directly** via the
moment-flow — it does NOT use the from-below free-Fisher `¼Φ*` (replacing `Φ*` was the entire point of
approach C). So a `Φ*` sup-characterization argument does **not** apply here, and the per-`Â` truncation
error of the moment-flow is, by itself, **uncertain-sign** (an ODE / word-length truncation error, not a
one-sided Voiculescu sup).

The danger is the **optimization**, not a per-step sign. Minimizing the truncated-flow energy over `Â`
**selects** the generator for which the truncation most under-estimates the true `⟨ψ|H|ψ⟩` — the optimizer
exploits whichever `Â` makes the computed number small, i.e. it preferentially picks the negative-error
direction. Under minimization the reported energy therefore drifts **below** the true variational value and
can dip under `E₀`. This is the same **selection** mechanism that sank M5c (there it was *amplified* by `Φ*`
being one-sided from below; here it arises purely from optimizing a quantity that is not a bound), and it is
the unsafe direction: the result is neither a guaranteed upper nor a guaranteed lower bound on `E₀`.

The M5c precedent (illustrating the selection mechanism): a truncated `¼Φ*=¼bᵀG⁻¹b` masqueraded as a bound;
the "degree-3 beats the Gaussian (2.322<2.365)" claim did **not** survive re-evaluation at a larger basis
(`max_word_len 3→4`: traciality residual jumped `1e-32 → 7.7e-5`, energy far higher). See
`qm_master_field.fisher_master_field` docstring and `tests/test_qm_master_field.py:113-117`.

**S2 verdict:** the pure-large-N L-truncated flow energy is **NOT** a rigorous bound. Positivity-certification
yields the **lower** side only (the existing SDP) and cannot upper-bound `E₀` from the trial; and minimizing a
non-bound over `Â` drives the result **unsafely downward** (the M5c selection mechanism). Not a variational
bound.

---

## Part 4 — Conclusion and recommendation

**Recommendation: (B) — pivot to finite-N staging.** Decisively, for two reasons:

1. **S2 is intractable for the pure-large-N trial** (Part 3): the truncated flow energy is not certifiable
   as a bound, and its error is biased *from below* — the precise way M5c already failed. Positivity gives
   only the lower side (which doesn't need the trial at all). There is no positivity route that turns the
   truncated trial into a rigorous *upper* bound, because moment-matrix relaxations are one-sided.

2. **Finite-N staging makes the bound exact by construction** (Part 3.1(a)): at fixed finite N,
   `|ψ⟩=e^{iÂ}|G⟩` is a literal unit vector, so `⟨ψ|H|ψ⟩` is an exact variational **upper** bound on `E₀(N)`
   (Rayleigh–Ritz), with the only error the *controlled, monotone, one-sided* Fock-truncation `K` already
   used by `exact_diag.py` (Test D shows it converging). The master field is then obtained by `N→∞`
   extrapolation of the finite-N bounds, sandwiched below by the existing bootstrap SDP
   (`bootstrap_two_matrix_qm`).

**What S1 buys us (and it is real):** the closure result (proven and machine-verified here) guarantees the
`N→∞` limit of the single-trace flow **exists and is single-trace-closed**, with factorization (11) the
mechanism. So the large-N moment-flow remains a **sound, cheap surrogate / initializer** for the finite-N
variational search (it predicts the planar energy landscape over Â without building the Fock space), and a
**consistency check** on the extrapolation — but the **certified number must come from the finite-N
expectation**, not from the truncated planar flow. Pure large-N is **not** viable as a *bound-producing*
method; it is viable only as a heuristic on top of finite-N staging.

---

## Appendix — verification script map (`sp2_flow_test.py`)

| Test | What it checks | Headline result |
|---|---|---|
| **E** | Generator normalization `Â=N·Tr(â)` ⟹ `dτ[w]/ds=O(1)` | N-independent to `~1e-14`, N=2..5 |
| **A** | Single-trace closure, closed-form ODE (7), sign, normalization | exact↔ODE = K-truncation only |
| **D** | K-convergence at fixed N (isolates truncation from finite-N) | N=2: 2.2e−2→3.8e−4; N=3: →4.7e−3 |
| **B** | Nonlinear flow `dX̃/ds=−X̃²`, single-trace ODE (9) | gap `6e−5` (N=2,K=8) |
| **C** | Factorization (11), the closure mechanism | `C_N·N²` flat = 0.500 (exact `1/N²`) |

`τ([X̃,P̃_X])=i` (exact at finite N) is confirmed in the design probe / Test E header. All operator products
are kept ordered (no cyclicity); the Fock construction reuses `exact_diag.py` / `qm_fock.py`.
