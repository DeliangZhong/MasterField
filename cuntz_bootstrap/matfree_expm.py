"""Matrix-free e^{iH} v for the Cuntz-Fock bootstrap.

The dense path in `hermitian_operator.py::assemble_unitary` forms the full
d x d matrix `Uhat = expm(i * Hhat)` at O(d^3) cost. This module computes
the same action on a vector without ever forming `Uhat`, exploiting:

  1. Sparsity of creation strings: C_w |u> = |w.u> (or 0) has at most one
     nonzero per column in the basis indexing used by `CuntzFockJAX`.
  2. We only ever use e^{iH} v (vector), never e^{iH} (matrix), for Wilson
     loops and cyclicity evaluations.

Strategy: truncated Taylor series

    e^{i H} v = sum_{k=0}^{N} (i H)^k v / k!

The expansion is computed iteratively:

    term_0 = v
    term_k = (i H / k) * term_{k-1}
    result = sum_{k=0..N} term_k

H v is itself evaluated matrix-free via precomputed (src, tgt) index
pairs, ONE pair list per Cuntz basis word w. For coefficient vector
h (length d = fock.dim), H = sum_w h_w C_w + conj(h_w) C_w^dag.

Convergence: order N = 25 gives 1e-13 error for ||H|| ~ 1, 1e-8 for
||H|| ~ 5. If training pushes ||H|| past ~3, bump order or switch to
scaling-and-squaring (not implemented here; deferred until needed).

Interface mirrors hermitian_operator.py for easy A/B testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from .fock import CuntzFockJAX


# =====================================================================
# Sparse (src, tgt) index pairs for creation strings C_w
# =====================================================================


@dataclass(frozen=True)
class WordPairs:
    """Precomputed sparse index arrays for H = sum_w h_w C_w + h.c.

    For word w at index w_idx (w = fock.basis[w_idx]):
      C_w |basis[src]> = |w ++ basis[src]>   if |w| + |basis[src]| <= L_trunc
                       = 0                    otherwise.

    We flatten all nonzeros of sum_w C_w into three parallel int32 arrays:

      all_src[k], all_tgt[k], all_w[k]  for k = 0..N_nnz-1

    The K-th nonzero is C_{all_w[k]} |basis[all_src[k]]> = |basis[all_tgt[k]]>.

    Total nnz across all C_w (D=2, n_labels=4):
      L_trunc=3, dim=85   -> nnz = 313
      L_trunc=4, dim=341  -> nnz = 1593
      L_trunc=5, dim=1365 -> nnz = ~7k
    """

    all_src: jnp.ndarray                 # int32, shape (N_nnz,)
    all_tgt: jnp.ndarray                 # int32, shape (N_nnz,)
    all_w: jnp.ndarray                   # int32, shape (N_nnz,)
    n_words: int                         # = fock.dim
    n_basis: int                         # = fock.dim (same; words ARE basis)
    n_nnz: int


def build_word_pairs(fock: CuntzFockJAX) -> WordPairs:
    """Enumerate (src, tgt, w_idx) triples for C_w on the fock basis."""
    basis_to_idx = fock.basis_to_idx
    L = fock.L_trunc
    all_src: list[int] = []
    all_tgt: list[int] = []
    all_w: list[int] = []
    for w_idx, w in enumerate(fock.basis):
        lw = len(w)
        # For each preimage basis state u, check if w + u is in basis
        for src_idx, u in enumerate(fock.basis):
            if lw + len(u) > L:
                continue
            combined = w + u
            tgt_idx = basis_to_idx[combined]
            all_src.append(src_idx)
            all_tgt.append(tgt_idx)
            all_w.append(w_idx)
    all_src_arr = jnp.asarray(np.asarray(all_src, dtype=np.int32))
    all_tgt_arr = jnp.asarray(np.asarray(all_tgt, dtype=np.int32))
    all_w_arr = jnp.asarray(np.asarray(all_w, dtype=np.int32))
    return WordPairs(
        all_src=all_src_arr,
        all_tgt=all_tgt_arr,
        all_w=all_w_arr,
        n_words=fock.dim,
        n_basis=fock.dim,
        n_nnz=len(all_src),
    )


# =====================================================================
# H v matrix-free (no d x d matrix built)
# =====================================================================


def h_matvec(
    h: jnp.ndarray, v: jnp.ndarray, wp: WordPairs,
) -> jnp.ndarray:
    """Compute (H v)_i where H = sum_w h_w C_w + conj(h_w) C_w^dag.

    Forward pair (src, tgt, w): result[tgt] += h_w * v[src].
    Adjoint  pair (src, tgt, w): result[src] += conj(h_w) * v[tgt].
    Both are JAX scatter-adds (`.at[idx].add`) and are fully
    autodifferentiable.
    """
    coeff = h[wp.all_w]                                # (N_nnz,) complex
    contrib_fwd = coeff * v[wp.all_src]                # (N_nnz,) complex
    result = jnp.zeros_like(v).at[wp.all_tgt].add(contrib_fwd)
    contrib_adj = jnp.conj(coeff) * v[wp.all_tgt]
    result = result.at[wp.all_src].add(contrib_adj)
    return result


# =====================================================================
# e^{i H} v via Taylor truncation
# =====================================================================


def expm_iH_v(
    h: jnp.ndarray,
    v: jnp.ndarray,
    wp: WordPairs,
    order: int = 25,
    sign: float = +1.0,
) -> jnp.ndarray:
    """Compute e^{i * sign * H} v via truncated Taylor series.

    e^{A} v = sum_{k=0}^{order} A^k v / k!

    Iteratively: term_0 = v, term_k = (A / k) * term_{k-1}, result = sum_k term_k.

    For sign=+1: returns e^{iH} v (used for forward direction +mu).
    For sign=-1: returns e^{-iH} v = (e^{iH})^dag v (used for -mu).

    Uses `jax.lax.fori_loop` to keep the JIT-traced graph small (one
    iteration body) regardless of `order`. Python-unrolled version
    builds a massive graph that compiles very slowly for the full
    loss (many expm calls per step) — fori_loop avoids that.
    """
    i_sign = 1j * sign

    def body(k_minus_1, state):
        term, result = state
        k = k_minus_1 + 1  # fori_loop iterates k_minus_1 = 0..order-1
        new_term = (i_sign / k) * h_matvec(h, term, wp)
        return (new_term, result + new_term)

    _, result = jax.lax.fori_loop(0, order, body, (v, v))
    return result


def assemble_hermitian_matfree(
    h: jnp.ndarray, fock: CuntzFockJAX, wp: WordPairs,
) -> jnp.ndarray:
    """Build H as dense matrix via h_matvec on each basis column.

    Equivalent to `hermitian_operator.assemble_hermitian(h, fock)` but
    avoids the O(d^3) memory cache in `_build_word_operators` (which
    is infeasible at L_trunc >= 5). Uses `jax.vmap` to batch h_matvec
    over the d columns of the identity matrix.

    Cost: O(d * nnz) vs O(d^3) for the cached-product build.
    """
    d = fock.dim
    eye_cols = jnp.eye(d, dtype=jnp.complex128)
    # vmap over the column axis of eye. Each column ej is a basis vector
    # and H ej is the j-th column of H.
    H = jax.vmap(
        lambda ej: h_matvec(h, ej, wp), in_axes=1, out_axes=1
    )(eye_cols)
    return H


def assemble_unitary_matfree(
    h: jnp.ndarray, fock: CuntzFockJAX, wp: WordPairs,
) -> jnp.ndarray:
    """Drop-in replacement for `hermitian_operator.assemble_unitary` that
    avoids the word-operator cache.

    H = assemble_hermitian_matfree(h, fock, wp);   Uhat = expm(i H)

    At L_trunc=4 (d=341), the _build_word_operators cache needs ~630 MB
    and O(d^3) setup. This route builds H in O(d*nnz) memory and time.
    The expm itself is dense (O(d^3)) via `jax.scipy.linalg.expm` —
    same as the existing dense path, so grad complexity is identical
    to the dense `assemble_unitary`.
    """
    H = assemble_hermitian_matfree(h, fock, wp)
    H = 0.5 * (H + H.conj().T)                       # symmetrise roundoff
    return jax.scipy.linalg.expm(1j * H)


def build_forward_link_ops_matfree(
    params: list[jnp.ndarray], fock: CuntzFockJAX, wp: WordPairs,
) -> list[jnp.ndarray]:
    """Assemble D forward link operators Uhat_mu via the matfree H build."""
    return [assemble_unitary_matfree(h, fock, wp) for h in params]


def expm_iH_v_norm_check(
    h: jnp.ndarray,
    v: jnp.ndarray,
    wp: WordPairs,
    order: int = 25,
    sign: float = +1.0,
) -> tuple[jnp.ndarray, float]:
    """Same as expm_iH_v but also reports ||last_term|| / ||v|| as a
    convergence diagnostic. For info/diagnostic only; not used in tight
    training loops."""
    i_sign = 1j * sign
    term = v
    result = v
    # Python-for version retained for diagnostics (runs outside JIT).
    for k in range(1, order + 1):
        term = (i_sign / k) * h_matvec(h, term, wp)
        result = result + term
    last_rel = float(
        jnp.linalg.norm(term) / (jnp.linalg.norm(v) + 1e-30)
    )
    return result, last_rel
