# M5c result — two-matrix QM: the SDP↔Gaussian sandwich, and a truncation-sensitivity finding for the operator master field

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

## The result — at the optimization basis (m=1)

| $\lambda$ | $E_{\rm lo}$ (SDP, certified) | $E_{\rm MF}$ deg-2 (= Gaussian) | $E_{\rm MF}$ deg-3, mwl-3 | $E_{\rm hi}$ (Gaussian, rigorous) |
|---|---|---|---|---|
| 0.0 | **2.0000** | 2.0000 | 2.0000 | **2.0000** |
| 0.5 | 2.0000 | 2.20688 | 2.18876 | 2.20688 |
| 1.0 | 2.0000 | 2.36452 | 2.32181 | 2.36452 |

At the optimization basis (degree-3, `max_word_len=3`) the free-Fisher field's energy sits *below*
the Gaussian (e.g. 2.32 vs 2.365 at $\lambda{=}1$), and degree-2 coincides with the Gaussian exactly
(the M3 expressiveness lesson). **But this "beats the Gaussian" does NOT survive higher-basis
scrutiny** — see the next section. Only the $\lambda{=}0$ anchor (`E/N²=2m`, exact) is solid on the
operator-field side.

### ⚠️ The λ>0 free-Fisher estimate is truncation-sensitive (post-hoc finding, 2026-06-26)

A careful convergence study overturned the headline above:
- The energy is `¼Φ*_trunc + V` with `Φ*=bᵀG⁻¹b` a **from-below** (under-)estimate of the kinetic
  energy. Minimizing it **rewards states that exploit the truncation**, and the
  traciality/conditioning checks *at the optimization basis* are fooled.
- The committed single-init `2.32` looks perfect at mwl-3 (`sym~1e-32`, `cond~42`) but its
  traciality **degrades to `sym=7.7e-5`** when re-checked at mwl-4 (properly provisioned, above the
  `1e-6` gate); a fresh mwl-4 optimization behaves identically (`2.322`, `sym=6.7e-5`).
- Aggressive multi-restart + traciality annealing finds "lower" states (2.05–2.22) that are
  **unambiguous artifacts**: at a higher basis their conditioning blows up (`cond → 1e4–1e10`),
  traciality collapses (`sym → 1e-3–1`), and their true energies are *above* the Gaussian (3.9–4.7).
  The optimizer exploits the from-below `Φ*` + a fooled symmetry check + a near-singular Gram at once.

**Conclusion:** the operator master field's "beats the Gaussian" claim is **not robust** at the
couplings/bases we can afford; `E_MF≈2.32` is a **truncation-sensitive estimate**, not a bound. The
reliable bracket at $\lambda{=}1$ is `E/N² ∈ [2.0 (SDP, certified), 2.365 (Gaussian, rigorous)]`. (At
$\lambda{=}0$ the method is exact — the conjugate variable is linear and captured at any basis.)

**Lesson:** truncated free-Fisher functionals are prone to truncation artifacts under optimization;
validation MUST re-check traciality, conditioning, and Fisher-stability at a basis *larger* than the
one optimized on. The committed gate (`cond_max=1e8`, traciality at the optimization basis) is
insufficient. The real fix is to stop using the from-below `Φ*` for the energy — see Status.

## Honest limitations (the open items)

- **The certified SDP lower bound is loose at $L{=}4$ — flat at the free floor $2m$ for all
  $\lambda$.** A commuting moment set ($m[[\tilde X,\tilde Y]^2]{=}0$, on which the $\lambda$-force
  $F_X{=}[\tilde Y,[\tilde X,\tilde Y]]$ vanishes) is feasible and not excluded at $L{=}4$, so the
  minimizer returns the trivial floor. The true energy is $>2m$ (zero-point fluctuations). A
  *non-trivial* certified lower bound needs higher $L$ (and cyclicity canonicalization to make that
  affordable) — the main open item. (Note: the M3-style large-$N$ product matrix does **not** apply
  here — the QM stationarity is single-trace, so it is vacuous; spec C1(iv) corrected.)
- **$E_{\rm MF}$ at $\lambda>0$ is truncation-sensitive, not a bound** — see the ⚠️ section above.
  The rigorous bracket is $[E_{\rm lo}\,(\text{SDP}),\,E_{\rm hi}\,(\text{Gaussian})]=[2.0,\,2.365]$
  at $\lambda{=}1$; the free-Fisher value ($\approx2.32$) does not survive higher-basis validation,
  so it is reported as an unreliable estimate, not as the rigorous interior number it was first
  claimed to be.
- **HHK referee (T3) not extracted.** HHK's $\lambda>0$ numbers live in their Fig. 3 (a figure);
  matching them was scoped as a soft cross-check and is not done. Our external validation is the
  exact $\lambda{=}0$ anchor + the internal SDP↔Gaussian bracket.

## Status
M5c built the full sandwich and verified its trust roots (the $\lambda{=}0$ anchor, the $\Phi^*$
reduction, the loop/Gauss relations to $<10^{-10}$). **Correction (2026-06-26):** the headline
"operator master field beats the Gaussian at $\lambda>0$" did **not** survive higher-basis scrutiny
— it is a truncation-sensitive estimate (the ⚠️ section), not a robust result. The honest
deliverables are: (i) the certified SDP lower bound $E_{\rm lo}=2m$ (trivial at $L{=}4$, but the
machinery is correct and verified); (ii) the rigorous Gaussian upper bound $E_{\rm hi}$; (iii) the
methodological finding that the from-below $\Phi^*$ functional is exploitable and needs higher-basis
validation.

**Path to a genuine operator master field** (the real follow-up, in progress): replace the
from-below $\Phi^*$ with an explicit momentum operator $\hat P$ satisfying $[X,P]=i$, so
$\langle P^2\rangle$ is computed *directly* and $\langle H\rangle$ is a true variational **upper**
bound that cannot be exploited downward (the spec's deferred "Approach 2"). That is the formulation
that could legitimately beat the Gaussian. Suite: full package 105 passed + 8 slow-gated (the M5c
slow tests assert the reproducible mwl-3 behavior — but carry the truncation caveat). Then: BFSS/BMN.
