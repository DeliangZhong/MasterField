# Ansatz comparison — one-matrix master field (Milestone 2, Task 7)

**Date:** 2026-06-24. **Setup:** quartic one-matrix model V′=M+0.5M³, cutoff K=12,
Cuntz–Fock dim D=17. Engine: multi-restart Adam + L-BFGS polish on the exact
nonlinear loop-equation residual (positivity automatic). Metric: interior-moment
error (m₀…m₈) vs the exact moments; restart-robustness = fraction of single-restart
runs (5 seeds) reaching interior error ≤ 1e-2. Reproduce: `python matrix_master_field/validate.py`.

## Results

| Ansatz | #params | best interior err | median err | restart success |
|---|---|---|---|---|
| **monomial (deg 3)** | **6** | **1.04e-3** | 1.38e-3 | **100%** |
| dense-Hermitian | 289 | 4.68e-3 | 4.96e-3 | 100% |
| amortized monomial (hidden 64) | 4678 | 1.05e-2 | — | — (one net for g∈[0.1,0.9]) |

Amortized accuracy scales with capacity (probe: hidden 32 → 1.9e-2; hidden 96 →
1.7e-3 interior, held-out g=0.6 → 2.8e-3), i.e. competitive with the dedicated
solve at sufficient width.

## Findings

1. **Monomial ansatz is the clear winner for a single coupling**: 48× fewer
   parameters than dense, the best accuracy, and 100% restart-robustness. The
   bounded-degree creation/annihilation structure is an efficient, well-conditioned
   inductive bias.
2. **The dense (maximal-flexibility) ansatz also recovers the master field** — 100%
   restart success, no catastrophic over-fitting — but at ~5× the error and ~48× the
   parameters. So for the *one-matrix* case, maximal flexibility does **not** produce
   the spurious-solution failure mode; automatic positivity + one-matrix (Hamburger)
   determinacy keep it honest.
3. **No ansatz exhibited a spurious solution here.** The genuine spurious-solution
   risk (loop equations satisfied, moments wrong) is a **multi-matrix** phenomenon —
   to be stress-tested in Milestone 3 against the SDP island, not present at one matrix.
4. **Amortization works**: a single network represents M̂(g) across the whole quartic
   family and generalizes to held-out couplings — the vehicle for the headline-(ii)
   amortized master field.

## Decision for Milestone 3 (two-matrix)

- **Primary ansatz: monomial** (extended to multi-matrix words), for efficiency and
  conditioning; **amortized monomial** for the M̂(λ) family.
- **Keep the dense Hermitian ansatz as a flexibility fallback / cross-check** — useful
  precisely to *detect* multi-matrix spurious solutions (if monomial and dense disagree
  inside the SDP island, that flags trouble).
- Carry the engine unchanged (Adam + L-BFGS polish; interior-moment validation;
  model-appropriate cutoff).
