"""Test suite for cuntz_bootstrap Phase 4 implementation.

Organisation:
- @pytest.mark.unit      — fast structural tests (Cuntz algebra, shapes, API)
- @pytest.mark.integration — physics validation (differentiability, optimizer step)
- @pytest.mark.slow      — full phase experiments
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from cuntz_bootstrap.fock import CuntzFockJAX


# -------------------------------------------------------------------------
# Task 2: JAX Cuntz-Fock space
# -------------------------------------------------------------------------


@pytest.mark.unit
def test_basis_dimension_formula():
    """dim = (n^{L+1} - 1)/(n - 1) for n > 1; dim = L + 1 for n = 1."""
    # n = 2D for D=2: n=4, L=3 -> (4^4 - 1)/3 = 85
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    assert f.dim == 85
    # n = 1, L = 6: dim = 7
    f1 = CuntzFockJAX(n_labels=1, L_trunc=6)
    assert f1.dim == 7


@pytest.mark.unit
def test_vacuum_basis_index_zero():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    assert f.basis[0] == ()
    assert f.basis_to_idx[()] == 0


@pytest.mark.unit
def test_cuntz_relation_interior():
    """â_i â†_j = δ_{ij} on states of length < L (interior)."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    interior_idx = np.array(
        [k for k, w in enumerate(f.basis) if len(w) < f.L_trunc]
    )
    for i in range(4):
        for j in range(4):
            prod = np.asarray(f.a[i] @ f.adag[j])
            expected = float(i == j) * np.eye(f.dim)
            got_sub = prod[np.ix_(interior_idx, interior_idx)]
            exp_sub = expected[np.ix_(interior_idx, interior_idx)]
            err = np.max(np.abs(got_sub - exp_sub))
            assert err < 1e-12, f"(i={i}, j={j}) err={err}"


@pytest.mark.unit
def test_completeness_interior():
    """Σ_i â†_i â_i = I - |Ω⟩⟨Ω| on non-vacuum interior words (length < L)."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    sum_num = sum(f.adag[i] @ f.a[i] for i in range(4))
    sum_num = np.asarray(sum_num)
    P_Om = np.asarray(f.vacuum_projector())
    expected = np.eye(f.dim) - P_Om
    interior_idx = np.array(
        [k for k, w in enumerate(f.basis) if 0 < len(w) < f.L_trunc]
    )
    err = np.max(
        np.abs(
            sum_num[np.ix_(interior_idx, interior_idx)]
            - expected[np.ix_(interior_idx, interior_idx)]
        )
    )
    assert err < 1e-12


@pytest.mark.unit
def test_vacuum_vector_shape_and_value():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    vac = f.vacuum_state()
    assert vac.shape == (f.dim,)
    assert float(jnp.real(vac[0])) == 1.0
    assert float(jnp.sum(jnp.abs(vac[1:]))) == 0.0


@pytest.mark.unit
def test_adag_creates_state():
    """â†_i |Ω⟩ = |i⟩."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    vac = f.vacuum_state()
    created = f.adag[0] @ vac
    # Expected: basis vector at index of word (0,)
    idx = f.basis_to_idx[(0,)]
    target = np.zeros(f.dim, dtype=np.complex128)
    target[idx] = 1.0
    err = float(jnp.max(jnp.abs(created - jnp.asarray(target))))
    assert err < 1e-12


@pytest.mark.unit
def test_a_annihilates_vacuum_to_zero():
    """â_i |Ω⟩ = 0."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    vac = f.vacuum_state()
    ann = f.a[0] @ vac
    err = float(jnp.max(jnp.abs(ann)))
    assert err < 1e-12


@pytest.mark.unit
def test_adag_truncation_boundary():
    """â†_i |w⟩ = 0 when len(w) = L_trunc (no room to add)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    # Pick a length-2 word at the truncation boundary
    idx = f.basis_to_idx[(0, 0)]
    e = np.zeros(f.dim, dtype=np.complex128)
    e[idx] = 1.0
    e = jnp.asarray(e)
    result = f.adag[0] @ e
    err = float(jnp.max(jnp.abs(result)))
    assert err < 1e-12, f"Expected 0, got max entry {err}"


# -------------------------------------------------------------------------
# Task 3: Polynomial master operator
# -------------------------------------------------------------------------

from cuntz_bootstrap.master_operator import (  # noqa: E402
    assemble_master_operator,
    direction_to_label,
    init_master_operator_params,
)


@pytest.mark.unit
def test_direction_to_label_roundtrip():
    """+1 -> 0, -1 -> 1, +2 -> 2, -2 -> 3 for D = 2."""
    assert direction_to_label(1, D=2) == 0
    assert direction_to_label(-1, D=2) == 1
    assert direction_to_label(2, D=2) == 2
    assert direction_to_label(-2, D=2) == 3


@pytest.mark.unit
def test_direction_to_label_rejects_zero():
    with pytest.raises(ValueError):
        direction_to_label(0, D=2)


@pytest.mark.unit
def test_direction_to_label_rejects_out_of_range():
    with pytest.raises(ValueError):
        direction_to_label(3, D=2)


@pytest.mark.unit
def test_init_master_operator_params_shape():
    """Param count = 2*d_L - 1 complex per matrix."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    params = init_master_operator_params(n_matrices=2, fock=f, seed=0)
    assert len(params) == 2
    expected_n = 2 * f.dim - 1
    for p in params:
        assert p.shape == (expected_n,)
        assert p.dtype == jnp.complex128


@pytest.mark.unit
def test_assemble_master_operator_zero_gives_zero():
    """c = 0 → Û = 0 matrix."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128)
    U = assemble_master_operator(c, fock=f)
    assert float(jnp.max(jnp.abs(U))) < 1e-15


@pytest.mark.unit
def test_assemble_master_operator_identity_coefficient():
    """c[0] = 1, others = 0 (empty-word creation = identity) → Û = I."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[0].set(1.0)
    U = assemble_master_operator(c, fock=f)
    err = float(jnp.max(jnp.abs(U - jnp.eye(f.dim, dtype=jnp.complex128))))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_single_creation():
    """c[1] = 1 → Û = â†_0 (basis[1] = (0,) in canonical enumeration)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    assert f.basis[1] == (0,), "Test assumes basis[1] = (0,)"
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[1].set(1.0)
    U = assemble_master_operator(c, fock=f)
    err = float(jnp.max(jnp.abs(U - f.adag[0])))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_single_annihilation():
    """c[d] = 1 → Û = â_0 (first annihilation coefficient)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    assert f.basis[1] == (0,), "Test assumes basis[1] = (0,)"
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[f.dim].set(1.0)
    U = assemble_master_operator(c, fock=f)
    err = float(jnp.max(jnp.abs(U - f.a[0])))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_two_creation_words():
    """Linear combination: c[1] = 2, c[2] = 3 → Û = 2 â†_0 + 3 â†_1."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    assert f.basis[1] == (0,) and f.basis[2] == (1,)
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[1].set(2.0).at[2].set(3.0)
    U = assemble_master_operator(c, fock=f)
    expected = 2.0 * f.adag[0] + 3.0 * f.adag[1]
    err = float(jnp.max(jnp.abs(U - expected)))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_length2_creation():
    """c for the word (0, 1) → Û = â†_0 @ â†_1 (applied to |Ω⟩ gives |0, 1⟩)."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    idx = f.basis_to_idx[(0, 1)]
    c = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[idx].set(1.0)
    U = assemble_master_operator(c, fock=f)
    # Apply to vacuum, should give basis vector |0, 1⟩
    v_out = U @ f.vacuum_state()
    target = np.zeros(f.dim, dtype=np.complex128)
    target[f.basis_to_idx[(0, 1)]] = 1.0
    err = float(jnp.max(jnp.abs(v_out - jnp.asarray(target))))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_master_operator_rejects_wrong_shape():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    with pytest.raises(ValueError):
        assemble_master_operator(jnp.zeros(10, dtype=jnp.complex128), fock=f)


# -------------------------------------------------------------------------
# Task 4: Unitarity loss
# -------------------------------------------------------------------------

from cuntz_bootstrap.unitarity import (  # noqa: E402
    unitarity_loss,
    unitarity_loss_from_params,
)


@pytest.mark.unit
def test_unitarity_loss_identity_zero():
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    loss = unitarity_loss([I, I])
    assert float(loss) < 1e-20


@pytest.mark.unit
def test_unitarity_loss_random_nonzero():
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    A = (
        jax.random.normal(k1, (f.dim, f.dim)) + 1j * jax.random.normal(k2, (f.dim, f.dim))
    ).astype(jnp.complex128)
    loss = unitarity_loss([A])
    assert float(loss) > 1.0


@pytest.mark.unit
def test_unitarity_loss_from_params_differentiable():
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    params = init_master_operator_params(n_matrices=2, fock=f, seed=0)
    loss, grads = jax.value_and_grad(unitarity_loss_from_params, argnums=0)(params, f)
    assert bool(jnp.isfinite(loss))
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))
        # gradient must be nonzero somewhere (loss not already at minimum)
        assert float(jnp.max(jnp.abs(g))) > 0.0


@pytest.mark.unit
def test_unitarity_loss_from_identity_params_zero():
    """c = (1, 0, ..., 0) for each matrix → Û = I → loss = 0."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    c0 = jnp.zeros(2 * f.dim - 1, dtype=jnp.complex128).at[0].set(1.0)
    params = [c0, c0]
    loss = unitarity_loss_from_params(params, f)
    assert float(loss) < 1e-20


# -------------------------------------------------------------------------
# Task 5: Wilson loop evaluator
# -------------------------------------------------------------------------

from cuntz_bootstrap.wilson_loops import wilson_loop  # noqa: E402


@pytest.mark.unit
def test_empty_loop_gives_one():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    W = wilson_loop([I, I], loop=(), fock=f, D=2)
    assert float(jnp.abs(W - 1.0)) < 1e-12


@pytest.mark.unit
def test_identity_loop_gives_one():
    """Any closed loop with all-identity Û evaluates to 1."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    W = wilson_loop([I, I], loop=(1, 2, -1, -2), fock=f, D=2)
    assert float(jnp.abs(W - 1.0)) < 1e-12


@pytest.mark.unit
def test_loop_with_mu_zero_raises():
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    with pytest.raises(ValueError):
        wilson_loop([I, I], loop=(1, 0, -1), fock=f, D=2)


# Note: v1 used master_operator + wilson_loops.build_forward_link_ops.
# v2 uses hermitian_operator.build_forward_link_ops (imported above as
# build_forward_link_ops_v2). The tests below use the v2 API.


@pytest.mark.unit
def test_wilson_loop_differentiable_through_hermitian_params():
    """Replace v1's master_operator-based gradient test with v2 exp-Hermitian."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    # Forward reference: `init_hermitian_params` and `build_forward_link_ops_v2`
    # are imported near the top of the v2 Task 2 section below.
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    params = _init_v2(n_matrices=2, fock=f, seed=0, scale=0.1)

    def loss_fn(ps):
        Us = _build_v2(ps, fock=f)
        W = wilson_loop(Us, loop=(1, 2, -1, -2), fock=f, D=2)
        return jnp.real(W)

    val, grads = jax.value_and_grad(loss_fn)(params)
    assert bool(jnp.isfinite(val))
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.unit
def test_wilson_loop_single_edge_from_identity_unitary():
    """W[(1,)] with Û = I gives 1."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    U_list = [I]
    W = wilson_loop(U_list, loop=(1,), fock=f, D=1)
    assert float(jnp.abs(W - 1.0)) < 1e-12


@pytest.mark.unit
def test_wilson_loop_adjoint_convention_v2():
    """W[(1, -1)] = ⟨Ω|Û_1 Û_1†|Ω⟩ = ⟨Ω|I|Ω⟩ = 1 for a unitary Û_1."""
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    params = _init_v2(n_matrices=1, fock=f, seed=7, scale=0.2)
    U_list = _build_v2(params, fock=f)
    W = wilson_loop(U_list, loop=(1, -1), fock=f, D=1)
    # For exactly unitary Û, Û Û† = I, so W[(1, -1)] = (I)[0,0] = 1.
    # Padé expm gives unitarity to ~1e-10, so |W - 1| ≤ 1e-10.
    assert float(jnp.abs(W - 1.0)) < 1e-10


# -------------------------------------------------------------------------
# v2 Task 4: MM residuals (per-equation; total loss composes in total_loss.py)
# -------------------------------------------------------------------------

from cuntz_bootstrap.mm_loss import (  # noqa: E402
    _load_loop_system,
    compute_all_wilson_loops,
    default_area_law_target,
    make_mm_residuals_fn,
)


@pytest.mark.integration
def test_mm_residuals_shape():
    """Residual array length = number of MM equations."""
    loop_sys = _load_loop_system(D=2, L_max=4)
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    params = _init_v2(n_matrices=2, fock=f, seed=0, scale=0.02)
    U_list = _build_v2(params, fock=f)
    residuals_fn = make_mm_residuals_fn(loop_sys=loop_sys, fock=f, D=2)
    res = residuals_fn(U_list, 5.0)
    assert res.shape == (len(loop_sys.mm_equations),)


@pytest.mark.integration
def test_mm_residuals_differentiable():
    """jax.grad of (params → sum of MM residuals²) is finite."""
    loop_sys = _load_loop_system(D=2, L_max=4)
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    residuals_fn = make_mm_residuals_fn(loop_sys=loop_sys, fock=f, D=2)

    def scalar(ps, lam):
        U_list = _build_v2(ps, fock=f)
        return jnp.sum(residuals_fn(U_list, lam) ** 2)

    params = _init_v2(n_matrices=2, fock=f, seed=0, scale=0.02)
    val, grads = jax.value_and_grad(scalar, argnums=0)(params, 5.0)
    assert bool(jnp.isfinite(val))
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))


@pytest.mark.integration
def test_mm_residuals_identity_nonzero():
    """At Û = I (all W[C] = 1) the MM residuals do NOT all vanish."""
    loop_sys = _load_loop_system(D=2, L_max=4)
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    U_list = [I, I]
    residuals_fn = make_mm_residuals_fn(loop_sys=loop_sys, fock=f, D=2)
    res = residuals_fn(U_list, 5.0)
    assert float(jnp.max(jnp.abs(res))) > 0.01


@pytest.mark.integration
def test_default_area_law_target_plaquette():
    """At λ=5, the area-law target for a plaquette is 1/(2λ) = 0.1."""
    loop_sys = _load_loop_system(D=2, L_max=4)
    target = default_area_law_target(loop_sys, 5.0)
    # Find a loop with area 1 (plaquette)
    plaq_idxs = [i for i in range(loop_sys.K) if loop_sys.areas.get(i, 0) == 1]
    assert plaq_idxs, "no area-1 loop in D=2 L_max=4 loop system"
    for i in plaq_idxs:
        assert abs(float(target[i]) - 0.1) < 1e-12


# -------------------------------------------------------------------------
# v2 Task 5: Cyclicity
# -------------------------------------------------------------------------

from cuntz_bootstrap.cyclicity import (  # noqa: E402
    build_cyclicity_test_loops,
    cyclicity_loss,
)


@pytest.mark.unit
def test_build_cyclicity_test_loops_min_length():
    loop_sys = _load_loop_system(D=2, L_max=4)
    loops = build_cyclicity_test_loops(loop_sys, min_length=3)
    for C in loops:
        assert len(C) >= 3
    # Should contain the length-4 plaquette and longer loops
    assert any(len(C) == 4 for C in loops)


@pytest.mark.unit
def test_cyclicity_loss_identity_zero():
    """Û = I → W[any loop] = 1 → all cyclic rotations equal → loss = 0."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    loop_sys = _load_loop_system(D=2, L_max=4)
    test_loops = build_cyclicity_test_loops(loop_sys, min_length=3)
    loss = cyclicity_loss([I, I], test_loops, f, D=2)
    assert float(loss) < 1e-20


@pytest.mark.integration
def test_cyclicity_loss_random_nonzero():
    """Generic random h gives nonzero cyclicity residuals."""
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    loop_sys = _load_loop_system(D=2, L_max=4)
    test_loops = build_cyclicity_test_loops(loop_sys, min_length=3)
    params = _init_v2(n_matrices=2, fock=f, seed=3, scale=0.3)
    U_list = _build_v2(params, fock=f)
    loss = cyclicity_loss(U_list, test_loops, f, D=2)
    assert float(loss) > 1e-5


@pytest.mark.integration
def test_cyclicity_differentiable():
    """jax.grad through cyclicity_loss is finite."""
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    loop_sys = _load_loop_system(D=2, L_max=4)
    test_loops = build_cyclicity_test_loops(loop_sys, min_length=3)

    def scalar(ps):
        U_list = _build_v2(ps, fock=f)
        return cyclicity_loss(U_list, test_loops, f, D=2)

    params = _init_v2(n_matrices=2, fock=f, seed=5, scale=0.1)
    val, grads = jax.value_and_grad(scalar)(params)
    assert bool(jnp.isfinite(val))
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))


# -------------------------------------------------------------------------
# v2 Task 6: Lattice symmetry
# -------------------------------------------------------------------------

from cuntz_bootstrap.lattice_symmetry import (  # noqa: E402
    b_d_generators,
    lattice_symmetry_loss,
)


@pytest.mark.unit
def test_b_d_generators_counts():
    """D generators: D sign flips + (D-1) adjacent swaps = 2D - 1."""
    for D in [1, 2, 3, 4]:
        gens = b_d_generators(D)
        assert len(gens) == 2 * D - 1


@pytest.mark.unit
def test_b_d_generator_flip_applies():
    """Axis-1 flip on (1, 2, -1, -2) gives (-1, 2, 1, -2)."""
    gens = b_d_generators(2)
    flip_1 = gens[0]  # first flip (axis 1)
    out = flip_1((1, 2, -1, -2))
    assert out == (-1, 2, 1, -2)


@pytest.mark.unit
def test_b_d_generator_swap_applies():
    """D=2 adjacent swap on (1, 2, -1, -2) gives (2, 1, -2, -1)."""
    gens = b_d_generators(2)
    swap = gens[2]  # after 2 flips comes the swap
    out = swap((1, 2, -1, -2))
    assert out == (2, 1, -2, -1)


@pytest.mark.unit
def test_lattice_symmetry_loss_identity_zero():
    """Û = I → W[anything] = 1 → loss = 0."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    test_loops = [(1, 2, -1, -2), (1, 1, 2, -1, -1, -2)]
    gens = b_d_generators(2)
    loss = lattice_symmetry_loss([I, I], test_loops, gens, f, D=2)
    assert float(loss) < 1e-20


@pytest.mark.integration
def test_lattice_symmetry_loss_random_nonzero():
    """Generic h → W[plaquette] != W[swap(plaquette)], loss > 0."""
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    params = _init_v2(n_matrices=2, fock=f, seed=9, scale=0.3)
    U_list = _build_v2(params, fock=f)
    gens = b_d_generators(2)
    loss = lattice_symmetry_loss(
        U_list, [(1, 2, -1, -2), (1, 1, 2, -1, -1, -2)], gens, f, D=2
    )
    assert float(loss) > 1e-6


@pytest.mark.integration
def test_lattice_symmetry_differentiable():
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    gens = b_d_generators(2)
    test_loops = [(1, 2, -1, -2)]

    def scalar(ps):
        U_list = _build_v2(ps, fock=f)
        return lattice_symmetry_loss(U_list, test_loops, gens, f, D=2)

    params = _init_v2(n_matrices=2, fock=f, seed=11, scale=0.1)
    val, grads = jax.value_and_grad(scalar)(params)
    assert bool(jnp.isfinite(val))
    for g in grads:
        assert bool(jnp.all(jnp.isfinite(g)))


# -------------------------------------------------------------------------
# Task 7: Optimizer
# -------------------------------------------------------------------------

from cuntz_bootstrap.optimize import OptResult, optimize_cuntz  # noqa: E402


@pytest.mark.integration
def test_optimize_cuntz_reduces_trace_loss():
    """Run tiny optimisation on a scalar loss and verify it decreases.

    Uses v2 (exp-Hermitian) ansatz. Loss = -Re(Tr(Û)) drives Û toward I,
    which minimises the trace loss.
    """
    f = CuntzFockJAX(n_labels=2, L_trunc=2)
    from cuntz_bootstrap.hermitian_operator import (
        assemble_unitary,
        init_hermitian_params,
    )

    def loss_fn(params, lam):
        U = assemble_unitary(params[0], f)
        return -jnp.real(jnp.trace(U))

    params0 = init_hermitian_params(n_matrices=1, fock=f, seed=3, scale=0.3)
    init_loss = float(loss_fn(params0, 1.0))

    res = optimize_cuntz(
        loss_fn=loss_fn, params0=params0, lam=1.0,
        n_steps=200, lr=1e-2, warmup=10, log_every=50, verbose=False,
    )
    assert isinstance(res, OptResult)
    assert res.final_loss < init_loss, (
        f"loss did not decrease: init={init_loss}, final={res.final_loss}"
    )
    for p in res.params:
        assert bool(jnp.all(jnp.isfinite(p)))


@pytest.mark.integration
def test_optimize_cuntz_mm_residuals_reduce():
    """Run on pure MM loss (v2) with small L=2 and verify decrease."""
    loop_sys = _load_loop_system(D=2, L_max=4)
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    from cuntz_bootstrap.hermitian_operator import (
        build_forward_link_ops as _build_v2,
        init_hermitian_params as _init_v2,
    )

    residuals_fn = make_mm_residuals_fn(loop_sys=loop_sys, fock=f, D=2)

    def loss_fn(params, lam):
        U_list = _build_v2(params, fock=f)
        return jnp.sum(residuals_fn(U_list, lam) ** 2)

    params0 = _init_v2(n_matrices=2, fock=f, seed=0, scale=0.02)
    init_loss = float(loss_fn(params0, 5.0))

    res = optimize_cuntz(
        loss_fn=loss_fn, params0=params0, lam=5.0,
        n_steps=100, lr=5e-3, warmup=10, log_every=50, verbose=False,
    )
    assert res.final_loss < init_loss


# -------------------------------------------------------------------------
# Task 8: Phase A — Gross-Witten gate
# -------------------------------------------------------------------------

from cuntz_bootstrap.gw_validation import gw_moments  # noqa: E402


@pytest.mark.unit
def test_gw_moments_strong_coupling():
    """Strong coupling (lam >= 1): w_1 = 1/(2 lam), w_{k>=2} = 0."""
    for lam in [1.0, 2.0, 5.0, 10.0]:
        w = gw_moments(lam=lam, K=4)
        assert float(w[0]) == 1.0
        assert abs(w[1] - 1.0 / (2.0 * lam)) < 1e-12
        for k in [2, 3, 4]:
            assert abs(w[k]) < 1e-12


@pytest.mark.unit
def test_gw_moments_weak_coupling_closed_forms():
    """Weak coupling (lam < 1): w_1 = 1 - lam/2, w_2 = (1 - lam)^2."""
    for lam in [0.3, 0.5, 0.8]:
        w = gw_moments(lam=lam, K=2)
        assert abs(w[1] - (1.0 - lam / 2.0)) < 1e-9
        assert abs(w[2] - (1.0 - lam) ** 2) < 1e-9


# -------------------------------------------------------------------------
# v2 Task 2: Hermitian operator + expm assembly
# -------------------------------------------------------------------------

from cuntz_bootstrap.hermitian_operator import (  # noqa: E402
    assemble_hermitian,
    assemble_unitary,
    init_hermitian_params,
)
from cuntz_bootstrap.hermitian_operator import (  # noqa: E402
    build_forward_link_ops as build_forward_link_ops_v2,
)


@pytest.mark.unit
def test_init_hermitian_params_shape():
    """Param vector per matrix is length d_L complex128."""
    f = CuntzFockJAX(n_labels=4, L_trunc=3)
    params = init_hermitian_params(n_matrices=2, fock=f, seed=0)
    assert len(params) == 2
    for p in params:
        assert p.shape == (f.dim,)
        assert p.dtype == jnp.complex128


@pytest.mark.unit
def test_assemble_hermitian_is_hermitian():
    """H = Σ (h_w C_w + h_w* A_w) is Hermitian for any h."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    params = init_hermitian_params(n_matrices=1, fock=f, seed=1, scale=0.3)
    H = assemble_hermitian(params[0], fock=f)
    err = float(jnp.max(jnp.abs(H - H.conj().T)))
    assert err < 1e-12, f"Hermiticity violation {err}"


@pytest.mark.unit
def test_assemble_hermitian_zero_gives_zero():
    """h = 0 → Ĥ = 0."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    h = jnp.zeros(f.dim, dtype=jnp.complex128)
    H = assemble_hermitian(h, fock=f)
    assert float(jnp.max(jnp.abs(H))) < 1e-15


@pytest.mark.unit
def test_assemble_unitary_at_zero_is_identity():
    """h = 0 → expm(i·0) = I."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    h = jnp.zeros(f.dim, dtype=jnp.complex128)
    U = assemble_unitary(h, fock=f)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    err = float(jnp.max(jnp.abs(U - I)))
    assert err < 1e-12


@pytest.mark.unit
def test_assemble_unitary_is_unitary():
    """Random h → Û is unitary to machine precision."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    params = init_hermitian_params(n_matrices=1, fock=f, seed=2, scale=0.5)
    U = assemble_unitary(params[0], fock=f)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    err = float(jnp.max(jnp.abs(U @ U.conj().T - I)))
    assert err < 1e-10, f"unitarity error {err}"


@pytest.mark.unit
def test_single_coefficient_harmonic_generator():
    """h_{(0,)} = alpha (real) → Ĥ = alpha (â_0 + â†_0), Û matches build_unitary_gaussian."""
    L = 4
    f = CuntzFockJAX(n_labels=1, L_trunc=L)
    alpha = 0.4
    # Index of word (0,) in fock.basis (should be index 1)
    idx = f.basis_to_idx[(0,)]
    h = jnp.zeros(f.dim, dtype=jnp.complex128).at[idx].set(alpha)
    H = assemble_hermitian(h, fock=f)
    # Expected: alpha (â†_0 + â_0)
    expected_H = alpha * (f.adag[0] + f.a[0])
    err_H = float(jnp.max(jnp.abs(H - expected_H)))
    assert err_H < 1e-12

    # Cross-check Û against master_field/cuntz_fock.build_unitary_gaussian
    from master_field.cuntz_fock import CuntzFockSpace as NumpyFock

    nf = NumpyFock(n_matrices=1, max_length=L)
    U_ref = nf.build_unitary_gaussian(alpha, matrix_idx=0)
    U = assemble_unitary(h, fock=f)
    err_U = float(jnp.max(jnp.abs(U - jnp.asarray(U_ref))))
    # Tolerance a bit looser because of different impls (scipy expm vs jax eigh)
    assert err_U < 1e-10, f"Phase 0 cross-check failed: err={err_U}"


@pytest.mark.unit
def test_unitary_differentiable_through_expm():
    """jax.value_and_grad on a scalar built from Û works."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)

    def scalar(h):
        U = assemble_unitary(h, fock=f)
        return jnp.real(jnp.trace(U))

    params = init_hermitian_params(n_matrices=1, fock=f, seed=3, scale=0.1)
    val, grad = jax.value_and_grad(scalar)(params[0])
    assert bool(jnp.isfinite(val))
    assert bool(jnp.all(jnp.isfinite(grad)))
    # Gradient should be nonzero for generic h
    assert float(jnp.max(jnp.abs(grad))) > 0.0


@pytest.mark.unit
def test_build_forward_link_ops_v2_shape():
    """build_forward_link_ops returns D unitary matrices of shape (d, d)."""
    f = CuntzFockJAX(n_labels=4, L_trunc=2)
    params = init_hermitian_params(n_matrices=2, fock=f, seed=0)
    Us = build_forward_link_ops_v2(params, fock=f)
    assert len(Us) == 2
    for U in Us:
        assert U.shape == (f.dim, f.dim)


# -------------------------------------------------------------------------
# v1 Task 8: Phase A — Gross-Witten gate (LEGACY; superseded by v2 Task 9)
# -------------------------------------------------------------------------

@pytest.mark.integration
def test_phase_a_gw_strong_coupling_converges():
    """D=1, L=6, lam=5: recover w_1 = 0.1, w_k = 0 for k>=2 via unitarity + supervised."""
    f = CuntzFockJAX(n_labels=1, L_trunc=6)
    lam = 5.0
    w_exact = gw_moments(lam=lam, K=6)

    def loss_fn(params, lam_):
        U = assemble_master_operator(params[0], f)
        I = jnp.eye(f.dim, dtype=jnp.complex128)
        L_unit = jnp.sum(jnp.abs(U @ U.conj().T - I) ** 2)
        v = f.vacuum_state()
        L_sup = jnp.zeros((), dtype=jnp.float64)
        for k in range(1, 7):
            v = U @ v
            wk = jnp.real(v[0])
            L_sup = L_sup + (wk - float(w_exact[k])) ** 2
        return L_unit + L_sup

    p0 = init_master_operator_params(n_matrices=1, fock=f, seed=0, scale=0.05)
    res = optimize_cuntz(
        loss_fn=loss_fn, params0=p0, lam=lam,
        n_steps=3000, lr=5e-3, warmup=100, log_every=500, verbose=False,
    )

    U = assemble_master_operator(res.params[0], f)
    I = jnp.eye(f.dim, dtype=jnp.complex128)
    unit_err = float(jnp.sqrt(jnp.sum(jnp.abs(U @ U.conj().T - I) ** 2)))
    assert unit_err < 1e-2, f"unitarity error {unit_err} too large"

    v = f.vacuum_state()
    for k in range(1, 3):
        v = U @ v
        wk = float(jnp.real(v[0]))
        err = abs(wk - float(w_exact[k]))
        assert err < 1e-2, (
            f"w_{k}: got {wk}, expected {w_exact[k]}, err={err}"
        )
