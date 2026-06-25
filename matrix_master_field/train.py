"""Optimization engine: solve the exact nonlinear loop equations over an
operator ansatz. Positivity is automatic (Hermitian operators, vacuum state).

`solve` (one-matrix) and `solve_two_matrix` run multi-restart / g-homotopy Adam
then an L-BFGS polish on the exact nonlinear residual. This is the genuinely new
method: no SDP relaxation, no dropped positivity — minimize the exact nonlinear
residual over genuine states. The two-matrix objective adds cyclicity/exchange/Z₂
losses, because the Cuntz vacuum is not tracial.

NOTE on truncation: at finite cutoff the loop equations constrain only interior
moments; validate interior moments, and prefer model-appropriate cutoffs.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import optax  # noqa: E402
import scipy.optimize as sopt  # noqa: E402

from matrix_master_field.bootstrap_sdp import (  # noqa: E402
    HAS_CVXPY,
    TRUSTED_SOLVERS,
    bootstrap_two_matrix,
    bootstrap_two_matrix_kz,
)
from matrix_master_field.fock_jax import power_moments, word_moment  # noqa: E402
from matrix_master_field.loss import (  # noqa: E402
    kz_sd_residual_from_moment,
    one_matrix_sd_residual,
    sd_residual_from_moment,
    symmetry_losses,
    symmetry_losses_from_moment,
    two_matrix_sd_residual,
    two_matrix_test_words,
)
from matrix_master_field.sparse_fock import SuffixSharedMoments  # noqa: E402


def _lbfgs_polish(loss_fn, params, maxiter=3000):
    """L-BFGS-B polish of `loss_fn` from `params` (arbitrary pytree array shape)."""
    shape = params.shape
    vg = jax.jit(jax.value_and_grad(loss_fn))

    def fun_and_grad(x):
        loss, grads = vg(jnp.asarray(x).reshape(shape))
        return float(loss), np.asarray(grads, dtype=np.float64).ravel()

    r = sopt.minimize(
        fun_and_grad, np.asarray(params).ravel(), jac=True, method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-16, "gtol": 1e-14},
    )
    return jnp.asarray(r.x).reshape(shape)


def _adam_run(loss_fn, params, steps, lr):
    sched = optax.warmup_cosine_decay_schedule(lr * 0.01, lr, min(200, steps // 5), steps)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(sched))

    def run(p):
        opt_state = opt.init(p)

        def body(carry, _):
            p, s = carry
            loss, grads = jax.value_and_grad(loss_fn)(p)
            updates, s = opt.update(grads, s, p)
            return (optax.apply_updates(p, updates), s), loss

        (p, _), _ = jax.lax.scan(body, (p, opt_state), None, length=steps)
        return p

    return jax.jit(run)(params)


def _validate_two_matrix(tr_target, sd_loss, sym_loss, *, island_fn, target_word,
                         sd_tol, sym_tol, island_tol):
    """Fail-closed validation shared by the two-matrix solvers (commutator and KZ).

    `island_fn(target_word, maximize) -> (value, solver, status)` returns the SDP
    island edge for the model in question (e.g. `bootstrap_two_matrix` or
    `bootstrap_two_matrix_kz`, partially applied). `validated` is True ONLY if:
      (a) the SD residual is below `sd_tol`;
      (b) the symmetry residual (cyclicity+exchange+Z2 — required for a TRACIAL,
          exchange/Z2-symmetric master field) is below `sym_tol`. A low SD residual
          with broken symmetry is not a master field, so this is gated regardless of
          the caller's `w_sym`;
      (c) the target moment lies inside a CERTIFIED SDP island — both edges from a
          trusted interior-point solver (CLARABEL/MOSEK) with status 'optimal'. An
          SCS fallback or 'optimal_inaccurate' edge is not a certificate and forces
          validated=False even if the moment lies in that (uncertified) bracket.
    """
    def _cert(solver, status):
        return status == "optimal" and (solver in TRUSTED_SOLVERS or solver == "exact")

    sd_ok = sd_loss < sd_tol
    sym_ok = sym_loss < sym_tol
    lb = ub = in_island = None
    (lb_solver, lb_status) = (ub_solver, ub_status) = (None, None)
    certified = False
    if HAS_CVXPY and island_fn is not None:
        lb, lb_solver, lb_status = island_fn(target_word, False)
        ub, ub_solver, ub_status = island_fn(target_word, True)
        if lb is not None and ub is not None:
            in_island = (lb - island_tol) <= tr_target <= (ub + island_tol)
            certified = _cert(lb_solver, lb_status) and _cert(ub_solver, ub_status)
    validation = {
        "target_word": target_word, "target_value": tr_target,
        "sd_ok": sd_ok, "sym_ok": sym_ok, "island": (lb, ub),
        "in_island": in_island, "island_certified": certified,
        "island_solver": (lb_solver, ub_solver),
        "island_status": (lb_status, ub_status),
    }
    validated = bool(sd_ok and sym_ok and in_island is True and certified)
    return validation, validated


def solve(ansatz, v_prime, fock_ops, K, *, n_restarts=4, steps=3000, lr=1e-2, seed=0, polish=True):
    """Minimize the 1-matrix SD residual over `ansatz`; return the best run."""
    def loss_fn(params):
        return one_matrix_sd_residual(power_moments(ansatz.build_operators(params)[0], K), v_prime)

    key = jax.random.PRNGKey(seed)
    best = None
    for _ in range(n_restarts):
        key, sk = jax.random.split(key)
        p = _adam_run(loss_fn, ansatz.init_params(sk), steps, lr)
        if polish:
            p = _lbfgs_polish(loss_fn, p)
        fl = float(loss_fn(p))
        if best is None or fl < best["sd_loss"]:
            M = ansatz.build_operators(p)[0]
            best = {
                "moments": np.asarray(power_moments(M, K)),
                "sd_loss": fl, "params": p, "operator": np.asarray(M),
            }
    return best


def solve_two_matrix(
    ansatz, fock_ops, g_target, *, max_word_len=4, w_sym=10.0, g_schedule=None,
    steps=4000, lr=5e-3, polish=True, validate=True, target_word=(0, 0),
    sdp_word_len=8, sd_tol=1e-4, sym_tol=1e-6, island_tol=1e-3,
):
    """Solve the commutator+mass two-matrix master field at coupling g_target.

    g-homotopy from the exact g=0 free field upward; loss = SD residual +
    w_sym·(cyclicity+exchange+Z₂).

    FAILS CLOSED via `_validate_two_matrix`. The returned dict carries `validated`:
    True ONLY if (a) the SD residual is below `sd_tol`, (b) the symmetry residual
    (cyclicity+exchange+Z2) is below `sym_tol` — a non-tracial/asymmetric operator is
    not a master field regardless of `w_sym`, and (c) the target moment lies inside a
    CERTIFIED bootstrap_two_matrix island (both edges 'optimal' from a trusted
    interior-point solver). A low residual alone is NOT sufficient; callers MUST check
    `validated` (and may inspect `sym_ok`, `island_certified`, `island_status`).

    Ansatz expressiveness is decisive at g>0. A degree-2 ansatz floors the SD
    residual at ~1e-3 and parks the moment BELOW the SDP lower bound (validated
    False) at every g≥0.2. A degree-3 ansatz — with the Fock cutoff at the
    exactness bound enforced by the guard below — solves the truncated loop
    equations to machine zero and lands the moment INSIDE the rigorous island
    (validated True) across g∈[0.3,1.0] (e.g. g=1: tr M0²=0.69 ∈ [0.618,1.0]).
    Use degree ≥ 3 for g>0. The exact moment still depends on the loop-equation
    truncation `max_word_len` (it moves within the island as max_word_len grows).

    Validation order matters. The island is checked at `sdp_word_len` (default 8).
    L=6 is too loose to reject low-`max_word_len` truncation artifacts: at g=1 the
    L=6 island is [0.62,1.0], but L=8 tightens to ≈[0.63,0.73] and L=10 to
    ≈[0.69,0.71]. A `max_word_len`=2 solve lands tr M0²≈0.80 (inside L=6 but OUTSIDE
    L=8 — an artifact); only `max_word_len`≥3 lands inside the tight island
    (≈0.69, matching the bootstrap). Use max_word_len ≥ 3 AND sdp_word_len ≥ 8 for
    a meaningful g>0 guard. (SCS accuracy degrades at high L — see the M3 results
    doc; a high-accuracy conic solver is the rigorous follow-up.)
    """
    # Truncation guard (degree-aware). The SD residual evaluates words up to
    # length L = max_word_len + 3 (the commutator inserts 3 letters). Each
    # degree-d ansatz letter changes the Cuntz quanta number by up to ±d, so a
    # length-L vacuum→vacuum amplitude reaches at most ⌊L/2⌋·d quanta; the Fock
    # cutoff must be at least that for every evaluated moment to be EXACT.
    # Verified empirically: at this bound the degree-3 g>0 solve drives the SD
    # residual to machine zero. The old flat `max_word_len+3` was correct only at
    # degree 2 and would silently pass a contaminated higher-degree run.
    degree = getattr(ansatz, "degree", 1)
    need = ((max_word_len + 3) // 2) * degree
    if fock_ops.max_length < need:
        raise ValueError(
            f"Fock cutoff max_length={fock_ops.max_length} too small for "
            f"max_word_len={max_word_len}, ansatz degree={degree}: need >= {need} "
            f"(= ⌊(max_word_len+3)/2⌋·degree). Increase the Fock cutoff, lower the "
            f"ansatz degree, or lower max_word_len."
        )

    words = two_matrix_test_words(max_word_len)
    if g_schedule is None:
        g_schedule = [g_target * t for t in (0.2, 0.4, 0.6, 0.8, 1.0)]

    def make_loss(g):
        def loss_fn(params):
            ops = ansatz.build_operators(params)
            return two_matrix_sd_residual(ops, words, g) + w_sym * symmetry_losses(ops, words)
        return loss_fn

    params = ansatz.params_for_free_field()  # exact g=0 solution as warm start
    for g in g_schedule:
        loss_fn = make_loss(g)
        params = _adam_run(loss_fn, params, steps, lr)
        if polish:
            params = _lbfgs_polish(loss_fn, params)

    ops = ansatz.build_operators(params)
    sd_loss = float(two_matrix_sd_residual(ops, words, g_target))
    result = {
        "operators": [np.asarray(o) for o in ops],
        "params": params,
        "g": g_target,
        "sd_loss": sd_loss,
        "sym_loss": float(symmetry_losses(ops, words)),
    }

    if validate:
        tr_target = float(word_moment(ops, target_word))
        result["validation"], result["validated"] = _validate_two_matrix(
            tr_target, sd_loss, result["sym_loss"],
            island_fn=lambda tw, mx: bootstrap_two_matrix(
                g_target, max_word_len=sdp_word_len, target_word=tw, maximize=mx,
                with_status=True),
            target_word=target_word, sd_tol=sd_tol, sym_tol=sym_tol, island_tol=island_tol)
    return result


def solve_two_matrix_sparse(
    field, g_target, *, max_word_len=4, w_sym=10.0, g_schedule=None,
    steps=2000, lr=5e-3, polish=True, validate=True, target_word=(0, 0),
    sdp_word_len=8, sd_tol=1e-4, sym_tol=1e-6, island_tol=1e-3, init_params=None,
):
    """Sparse-Fock two-matrix solve: identical physics and fail-closed gate to
    `solve_two_matrix`, but evaluates moments with the scatter-add
    `SparseMonomialField` instead of dense D×D matrices — so it reaches large
    cutoffs (max_word_len ≥ 5, dim ≥ 8191) where the dense path runs out of memory.

    `field` is a `sparse_fock.SparseMonomialField`. Returns the same dict shape as
    `solve_two_matrix` (with `params`, `sd_loss`, `validated`, `validation`, `field`).

    `init_params` warm-starts the optimization (default: the exact g=0 free field).
    Because the ansatz coefficients have the same shape (n, n_monomials) at every
    cutoff — n_monomials depends only on `degree` — the converged params of a LOWER
    max_word_len solve are a valid warm start (a **max_word_len-homotopy**): you can
    chain truncation orders and skip a fresh g-homotopy. Measured benefit is modest
    (L-BFGS already converges from the free field in ~150 steps at dim 1023), and it
    does NOT reduce the dominant costs at large cutoffs — see the M3 doc §5 on why
    the high-max_word_len solve is run-bound (scatter compute) up to ~max_word_len=5
    and XLA-compile-bound beyond, and why scan/batching is the wrong fix.
    """
    need = ((max_word_len + 3) // 2) * field.degree
    if field.cutoff < need:
        raise ValueError(
            f"sparse Fock cutoff={field.cutoff} too small for max_word_len="
            f"{max_word_len}, degree={field.degree}: need >= {need} "
            f"(= ⌊(max_word_len+3)/2⌋·degree)."
        )

    words = two_matrix_test_words(max_word_len)
    if g_schedule is None:
        g_schedule = [g_target * t for t in (0.2, 0.4, 0.6, 0.8, 1.0)]

    # Suffix-shared moment evaluator (Codex finding 3): word_moment applies
    # right-to-left, so words sharing a suffix share their partial state. Record the
    # words the loss needs once — at a nonzero g if any stage is nonzero, so the
    # commutator words are included — and build the shared-suffix trie. This is ~5×
    # fewer scatter-adds per loss than re-running word_moment per word, and is faster
    # to BOTH compile and run than the naive per-word path (and than vmap/scan).
    rec_g = next((g for g in list(g_schedule) + [g_target] if g != 0), 0.0)
    _needed = []

    def _rec(w):  # record the words the loss requests (value is unused)
        _needed.append(tuple(w))
        return 0.0

    sd_residual_from_moment(_rec, words, rec_g)
    symmetry_losses_from_moment(_rec, words)
    shared = SuffixSharedMoments(field, _needed)

    def make_loss(g):
        def loss_fn(params):
            moment = shared.moment_fn(params)
            return (sd_residual_from_moment(moment, words, g)
                    + w_sym * symmetry_losses_from_moment(moment, words))
        return loss_fn

    params = field.params_for_free_field() if init_params is None else jnp.asarray(init_params)
    for g in g_schedule:
        loss_fn = make_loss(g)
        params = _adam_run(loss_fn, params, steps, lr)
        if polish:
            params = _lbfgs_polish(loss_fn, params)

    moment_final = shared.moment_fn(params)
    sd_loss = float(sd_residual_from_moment(moment_final, words, g_target))
    result = {
        "params": params, "g": g_target, "sd_loss": sd_loss,
        "sym_loss": float(symmetry_losses_from_moment(moment_final, words)),
        "field": field,
    }
    if validate:
        tr_target = float(field.word_moment(params, target_word))
        result["validation"], result["validated"] = _validate_two_matrix(
            tr_target, sd_loss, result["sym_loss"],
            island_fn=lambda tw, mx: bootstrap_two_matrix(
                g_target, max_word_len=sdp_word_len, target_word=tw, maximize=mx,
                with_status=True),
            target_word=target_word, sd_tol=sd_tol, sym_tol=sym_tol, island_tol=island_tol)
    return result


def solve_kz_sparse(
    field, g, h, *, max_word_len=4, w_sym=10.0, n_stages=5, steps=2000, lr=5e-3,
    polish=True, validate=True, target_word=(0, 0), sdp_word_len=8, sd_tol=1e-4,
    sym_tol=1e-6, island_tol=1e-3, init_params=None,
):
    """Kazakov–Zheng two-matrix master field (Milestone 4; arXiv:2108.04830 eq.6)
    S = N·tr[½(A²+B²) + (g/4)(A⁴+B⁴) − (h/2)[A,B]²], force V'_a = M_a + g·M_a³ +
    h·(M_a M_b²+M_b² M_a−2 M_b M_a M_b).

    Same engine as `solve_two_matrix_sparse` (sparse Cuntz–Fock + suffix-shared
    moments + Z2×Z2/exchange/cyclicity symmetry losses + the fail-closed gate), but
    with the KZ force and a coupling-homotopy from the exact g=h=0 free field: stage
    t∈{1/n_stages,…,1} ramps the couplings to (t·g, t·h). Validated against the KZ
    SDP island `bootstrap_two_matrix_kz`. Same result-dict shape as the other solvers.
    """
    degree = field.degree
    need = ((max_word_len + 3) // 2) * degree
    if field.cutoff < need:
        raise ValueError(
            f"sparse Fock cutoff={field.cutoff} too small for max_word_len="
            f"{max_word_len}, degree={degree}: need >= {need} "
            f"(= ⌊(max_word_len+3)/2⌋·degree)."
        )
    words = two_matrix_test_words(max_word_len)
    stages = [i / n_stages for i in range(1, n_stages + 1)]  # homotopy parameter t

    # Suffix-shared moments: record the word set once at the full couplings (g,h) —
    # a superset of every stage's, since the active force terms don't change with t>0.
    _needed = []

    def _rec(w):
        _needed.append(tuple(w))
        return 0.0

    kz_sd_residual_from_moment(_rec, words, g, h)
    symmetry_losses_from_moment(_rec, words)
    shared = SuffixSharedMoments(field, _needed)

    def make_loss(t):
        def loss_fn(params):
            moment = shared.moment_fn(params)
            return (kz_sd_residual_from_moment(moment, words, t * g, t * h)
                    + w_sym * symmetry_losses_from_moment(moment, words))
        return loss_fn

    params = field.params_for_free_field() if init_params is None else jnp.asarray(init_params)
    for t in stages:
        loss_fn = make_loss(t)
        params = _adam_run(loss_fn, params, steps, lr)
        if polish:
            params = _lbfgs_polish(loss_fn, params)

    moment_final = shared.moment_fn(params)
    sd_loss = float(kz_sd_residual_from_moment(moment_final, words, g, h))
    result = {
        "params": params, "g": g, "h": h, "sd_loss": sd_loss,
        "sym_loss": float(symmetry_losses_from_moment(moment_final, words)),
        "field": field,
    }
    if validate:
        tr_target = float(field.word_moment(params, target_word))
        result["validation"], result["validated"] = _validate_two_matrix(
            tr_target, sd_loss, result["sym_loss"],
            island_fn=lambda tw, mx: bootstrap_two_matrix_kz(
                g, h, max_word_len=sdp_word_len, target_word=tw, maximize=mx,
                with_status=True),
            target_word=target_word, sd_tol=sd_tol, sym_tol=sym_tol, island_tol=island_tol)
    return result
