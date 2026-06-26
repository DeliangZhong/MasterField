# M5c follow-up — finite-N exact diagonalization (ground truth for the two-matrix QM)

**Date:** 2026-06-26  **Status:** design (approved in brainstorming; spec under review)

**Goal.** Compute the *true* ground-state energy density `E/N²` of the two-matrix QM
$$H = \mathrm{Tr}\big(P_X^2+P_Y^2+m^2(X^2+Y^2)-g^2[X,Y]^2\big)\quad(\text{HHK arXiv:2004.10212 Eq 17})$$
by **exact diagonalization at small N (N=2 and N=3)** — an independent reference that validates every
M5c bound (SDP lower `2.0`, Gaussian upper `2.365`) and pins *where in that bracket the truth sits*.

**Why now (motivation).** The M5c dig showed the free-Fisher master field's "beats the Gaussian"
was a truncation artifact, undetected until hand-stress-tested — because we had **no ground truth**.
A finite-N exact energy would have caught it in one line. This is sub-project 1 of the "genuine
master field" follow-up; sub-project 2 (a direct-momentum variational upper bound) gets its own spec
and will be benchmarked against this. **This spec is sub-project 1 only.**

---

## Conventions (pinned — derived, not guessed)

- **Model:** HHK Eq 17 (above); `X,Y` Hermitian `N×N` (**U(N)** — the trace mode is included),
  `[X_{ij},P_{X,kl}]=i\,\delta_{il}\delta_{jk}`, `ℏ=1`, `λ=Ng²` ('t Hooft); report `E/N²`.
- **Mode decomposition.** Orthonormal Hermitian basis `{T_a}_{a=1}^{N²}`, `Tr(T_aT_b)=δ_{ab}`
  (N=2: `{I,σ_x,σ_y,σ_z}/√2`; N=3: `I/√3` + the 8 Gell-Mann matrices `/√2`). Expand
  `X=Σ_a x_a T_a`, `P_X=Σ_a p_{x,a}T_a` (and `Y`,`P_Y`). Orthonormal completeness
  `Σ_a (T_a)_{ij}(T_a)_{kl}=δ_{il}δ_{jk}` gives **canonical modes** `[x_a,p_{x,b}]=i\,δ_{ab}`, and
  `Tr X²=Σ_a x_a²`, `Tr P_X²=Σ_a p_{x,a}²`.
- **Free part = decoupled oscillators:** `Tr(P_X²+m²X²)+Tr(P_Y²+m²Y²)=Σ_{a}(p_{x,a}²+m²x_a²)+(y)` —
  `2N²` independent harmonic oscillators, each `p²+m²x²` with spectrum `(2n+1)m`.
- **Interaction (derived):** with structure constants `[T_a,T_b]=i\,Σ_c f_{abc}T_c`
  (`f_{abc}=-i\,Tr([T_a,T_b]T_c)`, real),
  $$-g^2\,\mathrm{Tr}[X,Y]^2 = +g^2\sum_c\Big(\sum_{a,b} f_{abc}\,x_a y_b\Big)^2 \ \ge 0,$$
  a positive quartic (sum of squares of the bilinears `L_c=Σ_{ab}f_{abc}x_a y_b`). Confining
  (`Tr[X,Y]²≤0`). **`f_{abc}` and the quartic are FD-verified against `Tr[X,Y]²` on random
  Hermitian `X,Y` (not guessed).**
- **Trace-mode reduction.** `T_0∝I` commutes with everything, so `x_0,y_0` are *free* decoupled
  oscillators absent from the interaction; the interacting diagonalization runs over the
  `2(N²-1)` traceless modes, and the two trace modes add `2m` to `E` analytically (N=2: 6
  interacting + 2 trace; N=3: 16 interacting + 2 trace).
- **Singlet sector.** The free ground state `|0⟩` is U(N)-invariant (the oscillator vacuum is
  invariant under any rotation of the modes) ⇒ a **singlet**; the gauge-invariant interaction keeps
  the ground state a singlet — matching HHK's singlet bootstrap. **Verified** by the SU(N) Casimir
  of the computed ground state (`≈0`); if the absolute ground state were non-singlet, project.
- **`g=0` anchor (exact, all N, all truncation):** every mode in `n=0` ⇒ `E=2N²m` ⇒ **`E/N²=2m`**
  (no finite-N correction for the free theory). The setup's hard check.

---

## Architecture — oscillator-mode Fock diagonalization

Real-space (`2N²`-dimensional) grids and DMRG are infeasible/overkill at N=2–3; the Fock approach
exploits the "decoupled oscillators + few-body quartic" structure.

1. **Modes → Fock.** Each mode `→` HO at frequency `m`: `x_a=(â_a+â†_a)/√{2m}`,
   `p_{x,a}=-i√{m/2}(â_a-â†_a)`, so `p_{x,a}²+m²x_a²=m(2â†_aâ_a+1)`. Separate ladder sets for the
   x-modes (`â`) and y-modes (`b̂`).
2. **Total-quanta truncation.** Keep Fock states with `Σ_i n_i ≤ K` over the `2(N²-1)` interacting
   modes (the ground state has few quanta — far smaller than per-mode truncation). Basis size
   `= C(K + 2(N²-1),\,2(N²-1))`; N=2 (`6` modes) is tiny even at `K=14`; N=3 (`16` modes) is
   `~10^5–10^6` at `K=8–10`.
3. **Build `H` sparse.** Free part diagonal in the Fock basis; the quartic `g²Σ_c L_c²` is a
   few-body operator in `â,â†,b̂,b̂†` (each `L_c` is bilinear) → sparse. Assemble as a
   `scipy.sparse` matrix over the truncated basis.
4. **Ground state via Lanczos.** `scipy.sparse.linalg.eigsh(H, k=1, which='SA')` → lowest
   eigenvalue `E`; `E/N²=(E_{\rm interacting}+2m)/N²`. Converge in `K`.

---

## Modules

- **`matrix_master_field/exact_diag.py` (new):**
  - `hermitian_basis(N)` → orthonormal `{T_a}` (incl. trace); `structure_constants(N)` → `f_{abc}`
    (FD-verified helper).
  - `build_two_matrix_qm_hamiltonian(N, m, g, K)` → sparse `H` over the truncated interacting Fock
    basis (+ the analytic trace contribution bookkeeping).
  - `ground_energy(N, m, g, K)` → `dict(E_over_N2, E, K, basis_dim, converged?)` via `eigsh`.
  - `casimir_expectation(N, ground_state, ...)` → the SU(N) Casimir on the ground state (singlet check).
- **`matrix_master_field/tests/test_exact_diag.py`:** the validation obligations below.

---

## Validation obligations (proof / computation for each)

| # | Claim | Check |
|---|---|---|
| V1 | structure constants + quartic correct | `f_{abc}` and `g²Σ_c L_c²` reproduce `−g²Tr[X,Y]²` to FD tolerance on random Hermitian `X,Y` |
| V2 | `g=0` anchor | `ground_energy(N, m, 0, K)` gives `E/N²=2m` to machine precision, any `N,K` |
| V3 | `K`-convergence | `E/N²(K)` converges as `K` grows (report the tail; the reported value is converged to a stated tolerance) |
| V4 | bracket containment | for `g>0`, `2.0 ≤ E_exact(N)/N² ≤` finite-N Gaussian `⟨H⟩(N)`; and the extrapolated `N→∞` value lies in `[2.0, 2.365]` |
| V5 | singlet ground state | Casimir of the ground state `≈0` (else project to the singlet sector) |
| V6 | the deliverable | `E_exact(N=2)/N²`, `E_exact(N=3)/N²` at `λ∈{0,0.5,1}`, + a 2-point `N→∞` estimate that pins the truth in the bracket and **adjudicates** the M5c bounds/master-field |

Verification medium: Python + `pytest` (numpy/scipy), as M1–M5.

---

## Risks / open questions
- **R1 — N=3 `K`-convergence (main risk).** 16 interacting modes; the ground state may need
  `K≈10` (basis `~5×10^6`) for tight convergence at `λ=1`. *Mitigation:* exploit sparsity + total-
  quanta truncation; report the `K`-tail honestly; if N=3 won't fully converge at feasible `K`, the
  deliverable degrades to "N=2 exact + an N=3 bound/trend" (still adjudicates the artifact question).
- **R2 — finite-N vs large-N.** The bounds are large-N (planar); exact diag is finite-N. The clean
  comparison is `E_exact(N)/N²` for `N=2,3` + extrapolation, plus the finite-N Gaussian `⟨H⟩(N)` as a
  same-`N` upper sanity. The `g=0` value `2m` is exact at all N (no extrapolation needed there).
- **R3 — singlet assumption.** If the absolute ground state is not a singlet (V5 fails), restrict to
  the singlet sector (projector built from the SU(N) generators) — adds work but is well-defined.

## Out of scope (the next sub-project, own spec)
- **The direct-momentum variational master field** (explicit `P̂`, `[X,P]=i`, `⟨P²⟩` direct → a
  genuine non-exploitable upper bound) — to be designed and built next, **benchmarked against the
  ground truth from this sub-project**.
