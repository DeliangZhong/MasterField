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

## Test runner (Milestone 1)

The project `.venv` is unpopulated and `uv run` (project env) tries to sync the full
lock (torch, ~slow). Use the lightweight cached ephemeral env instead:

```
uv run --no-project --with numpy --with scipy --with pytest [--with cvxpy] python -m pytest matrix_master_field/tests/ -v
```
