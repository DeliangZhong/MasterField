# Conventions — matrix_master_field

- Hermitian N×N matrices; action S = N·Tr[…]; couplings are 't Hooft (fixed as N→∞).
- Moments m_w = lim_{N→∞}(1/N)⟨tr M_{w₁}…M_{w_k}⟩, m_∅ = 1 (hard constraint, never optimized).
- One-matrix: V=½M² (V′=M); V=½M²+(g/4)M⁴ (V′=M+gM³).
- Two-matrix (commutator+mass): S = N·tr[½(M₁²+M₂²) − (λ/4)[M₁,M₂]²], λ>0 (confining).
- Kazakov–Zheng action: TRANSCRIBE verbatim from arXiv:2108.04830 §2 before Milestone 4 — do not guess the quartic/commutator coefficients.
- Cuntz–Fock: a_i a†_j = δ_ij; vacuum |Ω⟩; tracial state τ=⟨Ω|·|Ω⟩ with cyclicity imposed (the Cuntz vacuum is not tracial in general). Positivity of τ is automatic.
- Float64 in all JAX code (`jax.config.update("jax_enable_x64", True)` at file top).

## Test runner (Milestone 1)

The project `.venv` is unpopulated and `uv run` (project env) tries to sync the full
lock (torch, ~slow). Use the lightweight cached ephemeral env instead:

```
uv run --no-project --with numpy --with scipy --with pytest [--with cvxpy] python -m pytest matrix_master_field/tests/ -v
```
