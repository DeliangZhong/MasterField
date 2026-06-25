# matrix_master_field/qm_master_field.py
"""M5c — master fields for two-matrix QM (HHK Eq 17).

C2 gaussian_master_field: explicit Gaussian trial state |ψ_G(Ω)⟩, ⟨H⟩ by Wick →
rigorous upper bound (NOT via Φ*). C3 free-Fisher operator field is added in a later
task. See docs/superpowers/specs/2026-06-25-m5c-two-matrix-qm-design.md (C2/C3).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np
from scipy import optimize


def gaussian_comm_moment(omega, N):
    """Wick ⟨Tr[X,Y]²⟩ in the Gaussian ground state of Tr(P²+Ω²X²) (X,Y independent).

    ⟨X_ij X_kl⟩=a δ_il δ_jk, a=1/(2Ω): ⟨Tr XYXY⟩=a²N, ⟨Tr XYYX⟩=a²N³ ⇒
    ⟨Tr[X,Y]²⟩ = 2a²N − 2a²N³ = 2a²N(1−N²)  (leading large-N: −N³/(2Ω²)).
    """
    a = 1.0 / (2.0 * omega)
    return 2.0 * a**2 * N * (1.0 - N**2)


def gaussian_master_field(m, lam):
    """Variational E/N² over the Gaussian family: min_Ω [Ω + m²/Ω + λ/(2Ω²)].

    A rigorous upper bound (variational principle on the explicit state |ψ_G(Ω)⟩).
    Returns dict(energy, omega, m2=1/(2Ω)).

    The stationarity condition f'(Ω)=0 is Ω³ − m²Ω − λ = 0, solved via brentq on
    the derivative 1 − m²/Ω² − λ/Ω³. At λ=0 the root is Ω=m exactly.
    """
    def f(omega):
        return omega + m**2 / omega + lam / (2.0 * omega**2)

    if lam == 0.0:
        omega = float(m)
    else:
        # f'(Ω) = 1 - m²/Ω² - λ/Ω³; unique positive root for λ>0.
        def df(omega):
            return 1.0 - m**2 / omega**2 - lam / omega**3

        # Upper bracket: for large Ω, df→1>0; lower bracket: df→-∞ for Ω→0+.
        upper = 10.0 + 10.0 * (m + lam)
        omega = float(optimize.brentq(df, 1e-8, upper))

    return {"energy": float(f(omega)), "omega": omega, "m2": 1.0 / (2.0 * omega)}


# ─── C3: free-Fisher operator master field ─────────────────────────────────────

def _free_diff_quotient_score(moment, basis_words, a):
    """b[c] = τ⊗τ(∂_a w_c) = Σ_{i: w_c[i]=a} moment(w_c[:i])·moment(w_c[i+1:])."""
    b = []
    for w in basis_words:
        s = 0.0
        for i, c in enumerate(w):
            if c == a:
                s = s + moment(tuple(w[:i])) * moment(tuple(w[i + 1:]))
        b.append(s)
    return jnp.asarray(b)


def _phi_star(moment, basis_words, n_matrices):
    """Φ* = Σ_a bᵀG⁻¹b, G[c,c']=moment(reverse(w_c)+w_c'). Differentiable hot path (no cond).

    `moment(word)->scalar` is the backend (numpy callable, or a JAX SuffixSharedMoments
    closure — both differentiable).
    """
    nb = len(basis_words)
    G = jnp.asarray([[moment(tuple(reversed(u)) + tuple(v)) for v in basis_words]
                     for u in basis_words]).reshape(nb, nb)
    phi = 0.0
    for a in range(n_matrices):
        b = _free_diff_quotient_score(moment, basis_words, a)
        phi = phi + jnp.real(b @ jnp.linalg.solve(G, b))
    return phi


def free_fisher_information(moment, basis_words, n_matrices):
    """(Φ*, cond(G)) — Φ* plus the Gram condition number (V6 diagnostic; call once, NOT in
    the optimization loop — the SVD in `cond` is unnecessary there). Truncating `basis_words`
    LOWERS Φ* (sup characterization), so ¼Φ* is a one-sided/from-below estimate of KE.
    """
    nb = len(basis_words)
    G = jnp.asarray([[moment(tuple(reversed(u)) + tuple(v)) for v in basis_words]
                     for u in basis_words]).reshape(nb, nb)
    cond = float(jnp.linalg.cond(jnp.real(G)))
    return _phi_star(moment, basis_words, n_matrices), cond


def fisher_master_field(m, lam, *, cutoff=8, degree=2, max_word_len=3,
                        steps=1500, lr=5e-3, w_sym=10.0, seed=0):
    """Minimize ¼Φ*(X̃,Ỹ) + m²(m[X̃²]+m[Ỹ²]) − λ·m[[X̃,Ỹ]²] over Cuntz–Fock operators,
    SUBJECT TO traciality + the model symmetries. **The Cuntz vacuum is NOT tracial**, so Φ*
    (which assumes a tracial state) is only valid once cyclicity is imposed (cf. M3/M4). The
    training loss = energy + w_sym·(cyclicity + X↔Y exchange + Z₂×Z₂); the reported `energy`
    excludes the penalty, and `sym_loss` reports the residual symmetry violation.

    Positivity automatic (tracial vacuum). Returns
    dict(energy, m2, comm2, phi_cond, sym_loss, grad_norm, params).
    """
    import optax
    from matrix_master_field.loss import symmetry_losses_from_moment
    from matrix_master_field.sparse_fock import SparseMonomialField, SuffixSharedMoments

    field = SparseMonomialField(n_matrices=2, cutoff=cutoff, degree=degree)
    basis = [w for w in _tm_position_words(max_word_len)]
    # Words the loss reads: Gram (reverse(u)+v), score splits, energy, comm2. The symmetry
    # reads (rotations / X↔Y exchange / Z₂ of basis words) stay within `basis`, hence within
    # `needed` (every basis word v appears via the u=() Gram row reverse(())+v = v).
    needed = set()
    for u in basis:
        for v in basis:
            needed.add(tuple(reversed(u)) + tuple(v))
    for w in basis:
        for i in range(len(w)):
            needed.add(tuple(w[:i]))
            needed.add(tuple(w[i + 1:]))
    for w in [(0, 0), (1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)]:
        needed.add(w)
    shared = SuffixSharedMoments(field, sorted(needed, key=len))

    def energy_only(moment):
        phi = _phi_star(moment, basis, n_matrices=2)
        comm2 = (moment((0, 1, 0, 1)) - moment((0, 1, 1, 0))
                 - moment((1, 0, 0, 1)) + moment((1, 0, 1, 0)))
        return 0.25 * phi + m**2 * (moment((0, 0)) + moment((1, 1))) - lam * comm2

    def loss_fn(params):
        moment = shared.moment_fn(params)
        return energy_only(moment) + w_sym * symmetry_losses_from_moment(moment, basis)

    params = field.params_for_free_field()
    opt = optax.adam(lr)
    state = opt.init(params)
    val_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    g = params
    for _ in range(steps):
        _, g = val_and_grad(params)
        updates, state = opt.update(g, state)
        params = optax.apply_updates(params, updates)
    grad_norm = float(optax.tree.norm(g))  # optimizer stationarity (V6); global L2 norm

    moment = shared.moment_fn(params)
    phi, cond = free_fisher_information(moment, basis, n_matrices=2)
    comm2 = float((moment((0, 1, 0, 1)) - moment((0, 1, 1, 0))
                   - moment((1, 0, 0, 1)) + moment((1, 0, 1, 0))).real)
    return {
        "energy": float(energy_only(moment).real),
        "m2": float(moment((0, 0)).real),
        "comm2": comm2,
        "phi_cond": cond,
        "sym_loss": float(symmetry_losses_from_moment(moment, basis)),
        "grad_norm": grad_norm,
        "params": params,
    }


def _tm_position_words(L):
    out, cur = [()], [()]
    for _ in range(L):
        cur = [w + (c,) for w in cur for c in (0, 1)]  # only X̃,Ỹ in the Φ* basis
        out += cur
    return out
