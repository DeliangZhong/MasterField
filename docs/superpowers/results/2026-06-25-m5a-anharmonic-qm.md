# M5a result — anharmonic-oscillator quantum mechanics as a bootstrap/operator-field sandwich

**Date:** 2026-06-25. **Model** (Han–Hartnoll–Kruthoff, arXiv:2004.10212, Eq 1):
$$H = p^2 + x^2 + g\,x^4,\qquad \hbar=1,\quad [x,p]=+i,\quad g\ge 0.$$
This is the first rung of M5 (matrix quantum mechanics → BFSS/BMN): the step from matrix
*integrals* (M1–M4) to matrix *quantum mechanics* — a Hamiltonian, an energy `E`, a
momentum with `[x,p]=i`. M5a builds and validates every new QM ingredient on an N=1 model
with an exact answer, before the large-N rungs (M5b single matrix, M5c two-matrix).

## The conceptual leap from M1–M4

| | M1–M4 (integral) | M5 (quantum mechanics) |
|---|---|---|
| unknowns | moments `τ(w)` | moments + **energy `E`** |
| closure | loop / SD equation | **stationarity `⟨[H,O]⟩=0`** |
| algebra | free Cuntz–Fock `ââ†=1` | **bosonic** oscillator `[â,â†]=1` ⟹ `[x,p]=i` |

## What was built (the sandwich)

```
   E_lo (SDP, certified)  ≤   E0(g)   ≤   E_var (operator field)
```

- **Operator master field — variational upper bound.** `x̂=(â+â†)/√2`, `p̂=−i(â−â†)/√2`
  on the truncated **bosonic** Fock space (`[â,â†]=1`, so `[x,p]=i` and positivity are
  automatic). The ground state is the lowest eigenvector of the exact Galerkin compression
  `Ĥ_trunc=P_K H P_K`; `E_var=λ_min` is a rigorous, monotone-decreasing upper bound.
  (`matrix_master_field/qm_fock.py`.)
- **Stationarity recursion — re-derived = HHK Eq 6.** From `[H,xᵗ]` and `[x,p]=i`,
  `4tE\,m_{t-1} + t(t-1)(t-2)m_{t-3} − 4(t+1)m_{t+1} − 4g(t+2)m_{t+3}=0`. Independent
  derivation in `derivations/m5a-anharmonic-qm.md`; numeric residual `<1e-8` and the
  energy relation D1 verified on exact-diag moments (`loss.qm_anharmonic_recursion_residual`,
  `tests/test_qm_recursion.py`).
- **SDP bootstrap — certified lower bound.** Variables `(E, m₂,…,m_{2K})`, the recursion as
  linear equalities (at fixed `E`), Hankel `⪰0`. Feasible energies form **narrow islands
  around the eigenvalues** (they shrink to the spectrum as `K` grows), so the lower bound is
  the left edge of the lowest island. Found robustly with a **max-min-eigenvalue margin**
  (`max t : Hankel ⪰ t·I`, always a trusted `optimal` solve) and an **anchored downward
  bisection** from the exact-diag `E0` (a feasible point at every `K`).
  (`bootstrap_sdp.bootstrap_qm_anharmonic`, `qm_anharmonic_feasibility`.)
- **Fail-closed gate.** `validated=True` only if a MOSEK/CLARABEL-certified `E_lo` and
  `⟨x²⟩` island bracket the exact-diag answer AND `E_exact ≤ E_var` — certifying the squeeze.
  (`train.solve_qm_anharmonic`, pure `train._qm_gate`.)

## Validation — the sandwich closes (K=24 operator field, K_sdp=6 SDP, MOSEK-certified)

| g | `E_lo` (SDP, certified) | `E0` (exact diag) | `E_var` (operator field) | `⟨x²⟩` island ∋ exact | validated |
|---|---|---|---|---|---|
| 0.0 | 0.83492 | **1.00000** | 1.00000 | [0.5000, 0.5000] ∋ 0.5000 | ✓ |
| 0.5 | 1.18791 | 1.24185 | 1.24185 | [0.3515, 0.3566] ∋ 0.3548 | ✓ |
| 1.0 | 1.36090 | 1.39235 | 1.39235 | [0.2980, 0.3134] ∋ 0.3058 | ✓ |
| 2.0 | 1.54501 | 1.60754 | 1.60754 | [0.2511, 0.2736] ∋ 0.2571 | ✓ |

Anchors all hit: `g=0` gives `E0=1` and `m₂=½` exactly (the ground state sits at the
untruncated Fock bottom; at `g=0` the recursion degenerates and pins the island to `m₂=½`);
`g=1` gives `E0=1.392352`, matching HHK. The variational bound is converged to `E_exact` at
`K=24`; the certified SDP lower bound sits ~3–4% below `E0` at `K_sdp=6` and tightens with `K`.

## Honest positioning

M5a's operator field *is* textbook variational diagonalization for a single particle — its
role is to validate the QM machinery (energy, stationarity, `[x,p]=i`, the oscillator Fock
space, the certified sandwich gate) against an exact answer. The genuine operator-master-field
novelty is the large-N rungs: **M5b** (single-matrix QM, HHK Eq 8 — reuses `qm_fock` + the QM
SDP, anchored on the free-fermion/collective-field exact solution) and **M5c** (two-matrix QM,
HHK Eq 17 — the unsolvable target, whose operator representation is the open research question).

## Tests

`test_qm_fock.py` (9), `test_qm_recursion.py` (3), `test_bootstrap_qm.py` (4),
`test_train_qm.py` (gate logic + `MMF_SLOW` certified sandwich). All green.
