# M5b result — single-matrix QM master field (collective field + free fermions), with the certified SDP as a documented open item

**Date:** 2026-06-25. **Model** (Han–Hartnoll–Kruthoff, arXiv:2004.10212, Eq 8):
$$H = \mathrm{Tr}\,P^2 + \mathrm{Tr}\,X^2 + \tfrac{g}{N}\,\mathrm{Tr}\,X^4,\qquad \hbar=1,\quad [X_{ij},P_{kl}]=i\,\delta_{il}\delta_{jk},\quad g\ \text{fixed at large }N.$$
Second rung of M5: the step from single-particle QM (M5a) to a genuine **large-N matrix** QM,
with the SU(N) Gauss law and 't Hooft scaling. The single matrix is "solvable" (N free
fermions) — the validation rung before M5c (two-matrix QM, the unsolvable target).

## Conventions pinned by derivation (not guessed)

- 't Hooft scaling `X=√N X̃`, `P=√N P̃`; normalized moments `m[w]=(1/N)⟨Tr w⟩` (O(1)); energy
  density `E/N² = m[P̃²] + m[X̃²] + g·m[X̃⁴]`.
- Matrix commutator (derived): `[X,P]=iN·𝟙`, so `[X̃,P̃]=i·𝟙`.
- SU(N) Gauss law (HHK Eq 11–13), re-derived: `Tr XP = Tr PX + Σ_ij[X_ij,P_ji]`,
  `Σ_ij[X_ij,P_ji]=iN²`, with `⟨Tr XP⟩` imaginary ⟹ `⟨Tr XP⟩ = iN²/2`.
- `g=0` anchor (exact, all N): `H=Tr P²+Tr X²` → N fermions in `−∂²+λ²` (levels `2n+1`),
  `E = Σ_{n=0}^{N-1}(2n+1) = N²` ⟹ **`E/N²=1`** (the large-N analog of M5a's `E₀=1`).

## What was built and validated — the operator master field at large N

The large-N singlet master field is the **rescaled eigenvalue density** `σ(y)` (`y=λ/√N`),
the Jevicki–Sakita collective field, minimizing the convex functional
$$\frac{E}{N^2}[\sigma]=\int\!\Big[\tfrac{\pi^2}{3}\sigma^3+(y^2+g\,y^4)\,\sigma\Big]dy,\qquad \int\sigma=1,\ \sigma\ge0,$$
with analytic minimizer `σ=(1/π)√(μ−y²−g y⁴)`. Two constructions + an independent referee, **all
agreeing** (`matrix_master_field/qm_collective.py`):

1. **Analytic collective** minimizer `σ(y)` → `E/N²`, moments.
2. **Variational** (operator-master-field-by-minimization): minimize `E/N²[σ_θ]` over a positive,
   normalized density ansatz (JAX, softmax parametrization) → upper bound at/above the exact min.
3. **Free-fermion referee**: finite-N level-filling of `h=−∂²_λ+λ²+(g/N)λ⁴` (= the M5a `qm_fock`
   Hamiltonian with coupling `g/N`), converging to the collective value; `g=0→1` exact at any N.

| g | `E/N²` (exact) | `⟨X²⟩/N²` | `⟨X⁴⟩/N³` |
|---|---|---|---|
| 0.0 | **1.00000** | 0.50000 | 0.50000 |
| 0.5 | 1.18049 | 0.37943 | 0.28110 |
| 1.0 | 1.30190 | 0.33143 | 0.21301 |
| 2.0 | 1.48047 | 0.28161 | 0.15288 |

Tests: `tests/test_qm_collective.py` (collective matches the table; `g=0` exact; free-fermion
converges; variational upper bound). All green.

## The certified SDP lower bound — documented open item (the honest finding)

The planned sandwich's lower half — a certified SDP bound on `E/N²` — is **not delivered**. The
attempt and its instructive failure are recorded here so the next session starts informed:

- A bootstrap over **single-trace** moments of words in `X̃,P̃` with the canonical algebra
  `[X̃,P̃]=i𝟙` is, rigorously, **a non-tracial state on the one-pair Heisenberg algebra — i.e. one
  quantum particle**. Empirically it converged (MOSEK-certified) to the M5a single-particle
  `E₀(g)` (e.g. `1.3923` at g=1, `1.6075` at g=2), **not** the matrix `E/N²` (`1.302`, `1.480`).
- The matrix model is N fermions at coupling `g/N`; distinguishing it requires genuine large-N
  **multi-trace / factorization** structure (HHK's Eq 14 carries `⟨Tr⟩⟨Tr⟩` terms). The
  single-trace shortcut discards it — the spec's original instinct (factorization needed) was right.
- Deriving the corrected loop equations by hand proved **error-prone**: three cross-checks
  disagreed — a naive Wick script gave `m[X̃P̃X̃P̃]=0`, the exact operator identity
  `X̃P̃X̃P̃=X̃²P̃²−iX̃P̃` forces `=1/2`, and the loop equation then implied `t₄=3/2` vs the true
  `0.5`. The gap is Weyl-ordering / O(N) bookkeeping. **No bootstrap was committed** on this basis.

**Path for the fresh session:** derive the multi-trace matrix-QM loop equations with full Moyal /
large-N care, and verify *every* relation against (a) the `g=0` Gaussian matrix-oscillator moments
(exact by Wick, done carefully) and (b) the free-fermion phase-space droplet Weyl moments
`∫∫_{q²+y²+gy⁴≤μ} y^a q^b dy dq/(2π)`, **before** trusting the SDP. Obtaining HHK's precise Eq 9–15
(read directly, not via the lossy fetch) would de-risk this.

## Status
M5b delivers the **validated large-N master field** (collective + free-fermion, the operator-field
half of the sandwich). The certified SDP lower bound is a clean, well-scoped open item. Next:
either close the SDP (fresh session) or proceed to **M5c** (two-matrix QM, the unsolvable target).
