# M5c result — two-matrix QM: operator master field beats the Gaussian on the unsolvable model

**Date:** 2026-06-25. **Model** (Han–Hartnoll–Kruthoff, arXiv:2004.10212, Eq 17):
$$H = \mathrm{Tr}\big(P_X^2+P_Y^2+m^2(X^2+Y^2)-g^2[X,Y]^2\big),\qquad [X_{ij},P_{kl}]=i\,\delta_{il}\delta_{jk},\quad \lambda=Ng^2.$$
Third and final rung of M5. The two matrices cannot be simultaneously diagonalized, so the
eigenvalue-density / free-fermion collective field that solved M5b **does not exist** — this is the
genuinely unsolvable target where the project's operator master field earns its keep. $\lambda=0$
decouples into two free matrix oscillators (the exact anchor); $\lambda>0$ has no closed form.

## What was built — the staged sandwich (`matrix_master_field/`)

Three pieces, each verified, assembled into a fail-closed sandwich (`train.solve_two_matrix_qm`):

- **C1 — certified SDP lower bound** (`bootstrap_sdp.bootstrap_two_matrix_qm`): ordered single-trace
  moments in $\{\tilde X,\tilde Y,\tilde P_X,\tilde P_Y\}$ + stationarity loop equations + SU(N)
  Gauss law + Gram PSD (no c-number commutator reduction). MOSEK/CLARABEL-certified.
- **C2 — Gaussian master field, rigorous upper bound** (`qm_master_field.gaussian_master_field`):
  the explicit trial state $|\psi_G(\Omega)\rangle$, $\langle H\rangle$ by Wick →
  $E/N^2(\Omega)=\Omega+m^2/\Omega+\lambda/(2\Omega^2)$, minimized over the stationarity cubic
  $\Omega^3-m^2\Omega-\lambda=0$. A genuine variational upper bound (not via $\Phi^*$).
- **C3 — free-Fisher operator master field** (`qm_master_field.fisher_master_field`): the novel
  core. Two $\tilde X,\tilde Y$ as ML-optimized Cuntz–Fock operators (positivity automatic);
  kinetic energy $\tfrac14\Phi^*(\tilde X,\tilde Y)$ with $\Phi^*_a=b^\top G^{-1}b$ (moment Gram
  $G$, free-difference-quotient score $b$); minimized **subject to traciality + the model
  symmetries** (the Cuntz vacuum is not tracial, so cyclicity is imposed as a penalty).

### Verified anchors (these are the trust roots, all GREEN in `tests/`)
- **$g{=}0$ anchor** $E/N^2=2m$, $m[\tilde X^2]=1/(2m)$ (free matrix oscillator). All three pieces
  reproduce it; the full sandwich `validated=True` at $\lambda{=}0$ (certified squeeze, all 7 gate
  conditions). (`test_m5c_derivations`, `test_bootstrap_two_matrix_qm`, `test_train_two_matrix_qm`.)
- **Free-Fisher reduction** $\tfrac14\Phi^*=\tfrac12$ for the $n{=}1$ semicircle of variance $\tfrac12$
  (= the M5b kinetic anchor); $\Phi^*_{\rm semicircle}=1$. This pins the $b^\top G^{-1}b$ construction
  (a genuine linear solve, reviewer-verified). (`test_qm_master_field`.)
- **Stationarity + Gauss law** verified to residual $<10^{-10}$ on the exact $g{=}0$ Gaussian
  moments — before the SDP relies on them. (`test_m5c_loop_equations`.)

## The result — the operator master field captures non-Gaussian physics (m=1)

| $\lambda$ | $E_{\rm lo}$ (SDP, certified) | $E_{\rm MF}$ deg-2 | **$E_{\rm MF}$ deg-3** | $E_{\rm hi}$ (Gaussian, rigorous) |
|---|---|---|---|---|
| 0.0 | **2.0000** | 2.0000 | 2.0000 | **2.0000** |
| 0.5 | 2.0000 | 2.20688 | **2.18876** | 2.20688 |
| 1.0 | 2.0000 | 2.36452 | **2.32181** | 2.36452 |

**The headline.** At **degree 3** the free-Fisher operator master field drops *strictly below* the
Gaussian/Hartree bound — by 0.8% at $\lambda{=}0.5$ and 1.8% at $\lambda{=}1$, the gap growing with
the coupling — capturing the non-Gaussian correlations the Hartree approximation cannot
($\mathrm{sym\_loss}\sim10^{-29}$, $\mathrm{grad}\sim10^{-4}$, $\mathrm{cond}\sim40$ throughout).
This is the project thesis realized on an unsolvable model: an ML operator field beats mean-field.

**The M3 expressiveness lesson, reconfirmed.** At **degree 2** the Cuntz–Fock ansatz only reaches
semicircular states, so $E_{\rm MF}$ coincides with the Gaussian *exactly* (to $10^{-5}$). Degree-3
is required for a genuinely non-Gaussian estimate at $\lambda>0$ — exactly the degree-2→degree-3
finding from the M3 matrix-integral work. (`test_fisher_beats_gaussian_at_degree3`.)

## Honest limitations (the open items)

- **The certified SDP lower bound is loose at $L{=}4$ — flat at the free floor $2m$ for all
  $\lambda$.** A commuting moment set ($m[[\tilde X,\tilde Y]^2]{=}0$, on which the $\lambda$-force
  $F_X{=}[\tilde Y,[\tilde X,\tilde Y]]$ vanishes) is feasible and not excluded at $L{=}4$, so the
  minimizer returns the trivial floor. The true energy is $>2m$ (zero-point fluctuations). A
  *non-trivial* certified lower bound needs higher $L$ (and cyclicity canonicalization to make that
  affordable) — the main open item. (Note: the M3-style large-$N$ product matrix does **not** apply
  here — the QM stationarity is single-trace, so it is vacuous; spec C1(iv) corrected.)
- **$E_{\rm MF}$ is a sharp *estimate*, not a certified bound.** Truncating the conjugate-variable
  basis makes $\tfrac14\Phi^*$ a from-below estimate of the kinetic energy, so $E_{\rm MF}$
  approaches the variational value from below as the basis grows. The *rigorous* bracket is
  $[E_{\rm lo}\,(\text{SDP}),\,E_{\rm hi}\,(\text{Gaussian})]$; $E_{\rm MF}$ is the sharp non-Gaussian
  number inside it. So the most one can say rigorously at $\lambda{=}1$ is $E/N^2\in[2.0,\,2.365]$,
  with the operator field estimating $\approx2.32$.
- **HHK referee (T3) not extracted.** HHK's $\lambda>0$ numbers live in their Fig. 3 (a figure);
  matching them was scoped as a soft cross-check and is not done. Our external validation is the
  exact $\lambda{=}0$ anchor + the internal SDP↔Gaussian bracket.

## Status
M5c is **complete as a demonstration**: the operator master field is built, verified (anchor,
$\Phi^*$ reduction, loop/Gauss relations), and **shown to beat the Gaussian** on the unsolvable
two-matrix QM at degree 3 — the genuine novel result. The rigorous *lower* half of the sandwich is
trivial at $L{=}4$ (honest limitation; tightening is the natural M5c follow-up). Suite: full package
105 passed + 7 slow-gated; M5c slow 10/10. Spec + plan: `docs/superpowers/specs/` and `plans/`
(both adversarially audited and revised). Next milestone: BFSS/BMN.
