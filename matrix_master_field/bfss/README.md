# Bosonic matrix QM (BFSS class) — master field vs Lin–Zheng bootstrap

Validating the master-field + continuation vehicle on Lin–Zheng bosonic matrix quantum mechanics
(arXiv:2507.21007, the D=9 case = bosonic BFSS), before QCD₃. Model (App-E form, complex D=2
letters Z,z̄,P,q̄; P = Π = −iP_phys so correlators are real):

    H = ½ Σ_I ( Tr P_I² + M² Tr X_I² ) − (g²/4) Σ_{I,J} Tr[X_I,X_J]²

Full writeup + findings: `docs/superpowers/results/2026-07-01-bosonic-matrixQM-setup.md`.
Run each with `uv run --no-project --with numpy --with scipy [--with cvxpy] python <file>`.

## Files
- `lz_level5.py` — reproduces their closed-form level-5 bracket E39 (understanding check).
- `lz_gaussian_mf.py` — **leading master field = free/semicircle state = the D=∞ exact saddle**;
  verified vs random matrices. 𝓔: +0.87% (D=2,M²=1), +0.75% (D=9 BFSS). The cheap reliable number.
- `lz_port.py` — **faithful Python port of their loop-equation engine** (from their notebook
  `Canonical111/O2massiveBootstrap`). cyclicity+CCR double-trace rule (`cycZP`), gauge, T-reversal
  (`mirror`), O(2) (`reflect`), EOM (`commH`). Moment counts match their Table I exactly
  (single-trace vars 14/94/614 at level 4/6/8).
- `lz_gauss_moments.py` — exact free-Gaussian moments via planar Wick; the true continuation
  anchor. Satisfies the g=0 loop equations to 1e-15 (end-to-end engine validation).
- `lz_point2.py`, `lz_point6.py` — exact-factorization + continuation from the Gaussian anchor.
  Finding: reaches machine-exact but *physically wrong* solutions (factorized loop eqs are
  underdetermined for a QM ground state) — unlike the KZ tracial integral.
- `lz_pos2.py` — adds ground-state positivity (`inner2`→M⪰0, `innerground2`→N⪰0), verified PSD at
  the true Gaussian. Reproduces the analytic lower bound 𝓔≥D·M·√3/4=0.866; with the Gaussian upper
  bound gives 𝓔∈[0.89,1.18]∋1.172. Tight island needs level 10–14 (their O(D) irrep machinery).

**Conclusion:** for a QM ground state, positivity is essential and shared with the bootstrap, so
the master-field "point vs bracket" advantage is weak for BFSS (unlike KZ / QCD loop equations).
The genuine value-add is large observables (long single-trace words) — the next step.
