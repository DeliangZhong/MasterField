# Conventions — matrix_master_field

- Hermitian N×N matrices; action S = N·Tr[…]; couplings are 't Hooft (fixed as N→∞).
- Moments m_w = lim_{N→∞}(1/N)⟨tr M_{w₁}…M_{w_k}⟩, m_∅ = 1 (hard constraint, never optimized).
- One-matrix: V=½M² (V′=M); V=½M²+(g/4)M⁴ (V′=M+gM³).
- Two-matrix (commutator+mass): S = N·tr[½(M₁²+M₂²) − (λ/4)[M₁,M₂]²], λ>0 (confining).
- **Kazakov–Zheng "unsolvable" two-matrix model** (arXiv:2108.04830, eq. 6), transcribed:
  S = N·tr[ ½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]² ], couplings g (quartic) ≥ 0, h (commutator) ≥ 0.
  Symmetries: A↔B exchange; Z₂×Z₂ (A→−A and B→−B independently — every term is even in each
  matrix). Normalized trace τ=(1/N)tr, τ(1)=1.
  Force (derived; ∂_A tr[A,B]² = 2[B,[A,B]], with [B,[A,B]] = 2BAB − B²A − AB²):
    V′_A = A + g·A³ + h·(A B² + B² A − 2 B A B),  V′_B by A↔B.
  Planar loop equation (N=∞): τ(V′_A · w) = Σ_{k: w_k=A} τ(w_{<k}) τ(w_{>k}).
  Validation anchors: (i) h=0 → two DECOUPLED quartic one-matrix models V′=M+g M³ (exact);
  (ii) quartic g=0 → the commutator+mass model below, matching FORCE coefficients (KZ h ↔ that
  model's λ/2, since its V′ commutator coeff is λ/2); (iii) g>0,h>0 → our own certified KZ SDP
  island. Force verified by finite difference vs ∂_A tr V on random Hermitian A,B.
- Cuntz–Fock: a_i a†_j = δ_ij; vacuum |Ω⟩; tracial state τ=⟨Ω|·|Ω⟩ with cyclicity imposed (the Cuntz vacuum is not tracial in general). Positivity of τ is automatic.
- Float64 in all JAX code (`jax.config.update("jax_enable_x64", True)` at file top).
- **Quantum mechanics (Milestone 5)** — single particle (M5a, HHK arXiv:2004.10212 Eq 1):
  H = p² + x² + g x⁴, ℏ=1, **[x,p]=+i** (HHK write [p,x]=−i), g≥0. Energy eigenstate |E⟩,
  moments m_k=⟨x^k⟩, m₀=1 (hard), odd moments 0 (parity). Oscillator rep x̂=(â+â†)/√2,
  p̂=−i(â−â†)/√2 with **[â,â†]=1** (BOSONIC Fock — NOT the free Cuntz–Fock ââ†=1) ⟹ [x̂,p̂]=i.
  Stationarity recursion (= HHK Eq 6, re-derived in `derivations/m5a-anharmonic-qm.md`):
  4tE·m_{t-1} + t(t-1)(t-2)·m_{t-3} − 4(t+1)·m_{t+1} − 4g(t+2)·m_{t+3} = 0.
  Anchors: g=0 → E₀=1, m₂=½ (exact); g=1 → E₀=1.392352 (HHK). Method = sandwich: certified
  SDP lower bound (margin SDP `max t: Hankel⪰t·I` + anchored downward bisection from the
  exact-diag E₀) ≤ E₀ ≤ variational upper bound (λ_min of the truncated Ĥ), fail-closed in
  `train.solve_qm_anharmonic`. M5b (single matrix, HHK Eq 8) and M5c (two-matrix, Eq 17) follow.
- **Single-matrix QM (Milestone 5b)** — HHK Eq 8: H = Tr P² + Tr X² + (g/N) Tr X⁴, g fixed at
  large N. [X_ij,P_kl]=iδ_il δ_jk; matrix commutator **[X,P]=iN·𝟙**; 't Hooft scaling X=√N X̃;
  normalized moments m[w]=(1/N)⟨Tr w⟩; energy density **E/N² = m[P̃²]+m[X̃²]+g·m[X̃⁴]**. SU(N)
  Gauss law (derived): ⟨Tr XP⟩=iN²/2. Master field = rescaled eigenvalue density σ(y) (collective
  field), E/N²[σ]=∫[π²σ³/3+(y²+g y⁴)σ]dy, ∫σ=1; analytic minimizer σ=(1/π)√(μ−y²−g y⁴). Anchors
  (exact, free fermions = collective): g=0 → **E/N²=1, ⟨X²⟩/N²=½**; g=1 → E/N²=1.30190. Code:
  `qm_collective.py` (collective + variational + finite-N free-fermion referee). **OPEN ITEM:** the
  certified SDP lower bound is NOT done — a single-trace bootstrap collapses to the single particle
  (it is a state on the one-pair Heisenberg algebra); the matrix case needs large-N multi-trace
  factorization. See `docs/superpowers/results/2026-06-25-m5b-single-matrix-qm.md`.

## Test runner (Milestone 1)

The project `.venv` is unpopulated and `uv run` (project env) tries to sync the full
lock (torch, ~slow). Use the lightweight cached ephemeral env instead:

```
uv run --no-project --with numpy --with scipy --with pytest [--with cvxpy] python -m pytest matrix_master_field/tests/ -v
```
