# M5b result — single-matrix QM as a certified bootstrap / collective-field sandwich

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

## The certified SDP lower bound — delivered (the correct matrix-QM bootstrap)

The sandwich's lower half is now built and certified. Getting it right required a correction of a
subtle error, recorded here because it is the crux of the matrix-QM bootstrap:

- **The trap:** treating `[X̃,P̃]=i𝟙` as a c-number and *reducing* moments makes the single-trace
  state a non-tracial state on the one-pair Heisenberg algebra — **one quantum particle**. It
  converged (MOSEK-certified) to the M5a single-particle `E₀(g)` (`1.3924` at g=1), not the matrix
  `E/N²`. **The matrix commutator `[X,P]` is NOT `iN·𝟙`** — it has operator parts (verified on an
  exact N=2 matrix oscillator: `Tr(XPXP)=0.5`, while `iN·𝟙` would force `3.5`).
- **The fix:** keep **ordered** single-trace moments `m[w]=⟨(1/N)Tr w⟩` as *independent variables*
  (no commutator reduction), constrained by: hermiticity + time-reversal reality; the **stationarity
  loop equations** `⟨[H,Tr w]⟩=0` (verified exactly on the N=2 ground state, e.g.
  `⟨Tr PX²P⟩+⟨Tr XPXP⟩+⟨Tr X²P²⟩=⟨Tr X⁴⟩`); the **SU(N) Gauss law**
  `⟨Tr([X,P]O)⟩=iN⟨Tr O⟩ ⟹ m[(0,1)+O]−m[(1,0)+O]=i·m[O]` (also verified on N=2) — this forces the
  phase-space area that prevents the trivial `E/N²≥0` collapse; and **Gram positivity**.
- Code: `bootstrap_sdp.bootstrap_single_matrix_qm(g, L)`; sandwich + fail-closed gate in
  `train.solve_single_matrix_qm` / `_sm_qm_gate`.

**Certified lower bound (`min E/N²`), bracketing the exact value and well below the single-particle
collapse (MOSEK-certified, L=4):**

| g | `E_lo` (SDP, certified) | `E/N²` (exact) | single-particle `E₀` (collapse signature) |
|---|---|---|---|
| 0.0 | **1.0000** | 1.00000 | 1.0 |
| 0.5 | 1.1034 | 1.18049 | 1.24185 |
| 1.0 | 1.1823 | 1.30190 | 1.39235 |

The full sandwich `E_lo ≤ E/N² ≤ E_var` holds with the collective variational `E_var` (≈ exact);
`validated=True` (`test_train_single_matrix_qm.py`). The lower bound is valid and certified;
tightening it (more constraints / the cyclicity-via-Gauss relations / higher L) is a future
refinement. Tests: `test_bootstrap_single_matrix_qm.py`, `test_train_single_matrix_qm.py`.

## Status
M5b is **complete**: the large-N master field (collective + free-fermion) **and** the certified
matrix-QM SDP lower bound, assembled into a validated sandwich. Next: **M5c** (two-matrix QM, the
unsolvable target).
