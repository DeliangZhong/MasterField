"""Discover exact lattice Makeenko-Migdal equations via null-space search.

STRATEGY (Discussion-30): instead of deriving the MM equation from
Kazakov-Zheng sign conventions (which stalled on Fig 3 interpretation),
use `qcd2_wilson_loop` as an oracle.

For each (loop C, edge e_j):
  1. Enumerate candidate terms that could appear in the MM equation
     (staple-replaced loops, the loop itself, products, identity).
  2. Evaluate each at several lambda values using `qcd2_wilson_loop`.
  3. Build the N_lam x n_candidates evaluation matrix M.
  4. Null vectors of M are coefficients of exact polynomial identities.

The discovered equations are the exact MM equations — they reproduce
machine-precision zero on `qcd2_wilson_loop` inputs at any lambda.

This module is PURE PYTHON (numpy + the master_field/lattice utilities).
The discovered equations are JSON-serializable and can be loaded into
`exact_mm.py` for use in the unsupervised training pipeline.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_MASTER_FIELD_DIR = str(Path(__file__).resolve().parent.parent / "master_field")
if _MASTER_FIELD_DIR not in sys.path:
    sys.path.insert(0, _MASTER_FIELD_DIR)

from lattice import (  # noqa: E402
    cyclic_canonical,
    reduce_backtracks,
)

from .qcd2_exact import qcd2_wilson_loop


# ========================================================================
# Data structures
# ========================================================================


@dataclass(frozen=True)
class LinearTerm:
    """One term: `lambda_power * W[loop]`.

    Setting `lambda_power=-1` represents `(1/lambda) * W[loop]` (the
    MM equation's LHS staple coefficient). `lambda_power=0` is the
    raw Wilson loop. `lambda_power=-2` is `(1/lambda^2) * W[loop]`.
    """
    name: str
    loop: tuple[int, ...]       # canonical, backtrack-reduced
    lambda_power: int = 0        # `lam ** lambda_power` factor

    def evaluate(self, lam: float) -> float:
        if len(self.loop) == 0:
            W = 1.0
        else:
            W = qcd2_wilson_loop(self.loop, lam)
        return W * (lam ** self.lambda_power)


@dataclass(frozen=True)
class ProductTerm:
    """Product `lam^p * W[loop_a] * W[loop_b]`."""
    name: str
    idx_a: int                   # index in linear_terms (use power=0 entries)
    idx_b: int                   # index in linear_terms
    lambda_power: int = 0


@dataclass
class ScanResult:
    loop: tuple[int, ...]
    edge_idx: int
    D: int
    linear_terms: list[LinearTerm]
    product_terms: list[ProductTerm]
    lam_values: list[float]
    matrix: np.ndarray
    singular_values: np.ndarray
    null_vectors: list[np.ndarray] = field(default_factory=list)
    null_sigmas: list[float] = field(default_factory=list)

    def n_total(self) -> int:
        return len(self.linear_terms) + len(self.product_terms)


# ========================================================================
# Candidate enumeration
# ========================================================================


def canonical(loop: tuple[int, ...]) -> tuple[int, ...]:
    """Reduce backtracks then cyclically canonicalize."""
    reduced = reduce_backtracks(tuple(loop))
    if not reduced:
        return ()
    return cyclic_canonical(reduced)


def enumerate_staple_replacements(
    loop: tuple[int, ...], edge_idx: int, D: int,
) -> list[tuple[str, tuple[int, ...]]]:
    """All loops obtained by replacing edge_idx with a 3-edge staple.

    A staple has shape (nu, mu, -nu) where mu = loop[edge_idx] and
    nu runs over {+1, -1, ..., +D, -D} with |nu| != |mu|.
    2(D-1) staples per edge.

    Returns (label, canonical_loop) pairs.
    """
    if not loop:
        return []
    mu = loop[edge_idx]
    abs_mu = abs(mu)
    prefix = loop[:edge_idx]
    suffix = loop[edge_idx + 1:]
    out: list[tuple[str, tuple[int, ...]]] = []
    for abs_nu in range(1, D + 1):
        if abs_nu == abs_mu:
            continue
        for sign in (+1, -1):
            nu = sign * abs_nu
            staple = (nu, mu, -nu)
            replaced = prefix + staple + suffix
            canon = canonical(tuple(replaced))
            label = f"staple_nu={nu:+d}"
            out.append((label, canon))
    return out


def enumerate_candidates(
    loop: tuple[int, ...], edge_idx: int, D: int,
    include_products: bool = True,
    lambda_powers: tuple[int, ...] = (0, -1),
    product_lambda_powers: tuple[int, ...] = (0, -1),
) -> tuple[list[LinearTerm], list[ProductTerm]]:
    """Build the candidate basis for the MM equation at (loop, edge_idx).

    For each loop we generate one term per `lambda_powers` entry.
    Default: power 0 (raw W) and power -1 (W / lambda) — the MM staple
    coefficient is 1/lambda.

    Linear terms generated:
      - (empty, power=p) for p in lambda_powers
      - (the canonical loop, power=p)
      - Each 2(D-1) staple-replacement loop, at every power
      (de-duplicated by (loop, power) tuple)

    Product terms:
      - W[a] * W[b] for every unordered pair of distinct LOOPS (not
        lambda-powers); product terms default to power 0 (no extra
        lambda factor). We intentionally do NOT cross-multiply lambda
        powers for products — nonlinear splittings in MM have no
        lambda prefactor.
    """
    canon_loop = canonical(loop)
    linear: list[LinearTerm] = []
    seen: set[tuple[tuple[int, ...], int]] = set()

    def add(name: str, lp: tuple[int, ...], p: int) -> None:
        key = (lp, p)
        if key in seen:
            return
        seen.add(key)
        label = name if p == 0 else f"{name}[λ^{p}]"
        linear.append(LinearTerm(name=label, loop=lp, lambda_power=p))

    for p in lambda_powers:
        add("empty", (), p)
        add("loop", canon_loop, p)
        for label, lp in enumerate_staple_replacements(loop, edge_idx, D):
            add(label, lp, p)

    # Products: pairs of raw (power=0) linear terms, each included at
    # every requested product_lambda_power.
    products: list[ProductTerm] = []
    if include_products:
        raw_indices = [
            i for i, t in enumerate(linear) if t.lambda_power == 0
        ]
        for a_pos, i in enumerate(raw_indices):
            for j in raw_indices[a_pos:]:
                ti, tj = linear[i], linear[j]
                if ti.loop == () and tj.loop == ():
                    continue
                if ti.loop == ():
                    continue         # W[empty]*W[x] = W[x], redundant
                if tj.loop == ():
                    continue
                for p in product_lambda_powers:
                    lbl = f"{ti.name}·{tj.name}"
                    if p != 0:
                        lbl = f"{lbl}[λ^{p}]"
                    products.append(
                        ProductTerm(
                            name=lbl, idx_a=i, idx_b=j, lambda_power=p,
                        )
                    )

    return linear, products


# ========================================================================
# Matrix building and null-space scan
# ========================================================================


def build_matrix(
    linear: list[LinearTerm],
    products: list[ProductTerm],
    lam_values: list[float],
) -> np.ndarray:
    """M[i, j] = evaluation of candidate j at lam_values[i].

    Column order: linear terms first, then products.
    Products use the RAW W values (power-0 entries) of their two
    factors, then multiply by lam^(product.lambda_power).
    """
    n_lin = len(linear)
    n_prod = len(products)
    n_col = n_lin + n_prod
    n_row = len(lam_values)
    M = np.zeros((n_row, n_col), dtype=np.float64)

    for i, lam in enumerate(lam_values):
        lin_values = np.array([t.evaluate(lam) for t in linear])
        M[i, :n_lin] = lin_values
        # Raw values (power=0) needed for product factors
        raw_W = np.array([
            (1.0 if len(t.loop) == 0 else qcd2_wilson_loop(t.loop, lam))
            for t in linear
        ])
        for k, prod in enumerate(products):
            M[i, n_lin + k] = (
                raw_W[prod.idx_a] * raw_W[prod.idx_b]
                * (lam ** prod.lambda_power)
            )
    return M


def null_vectors(
    M: np.ndarray, tol: float = 1e-10,
) -> tuple[list[np.ndarray], list[float]]:
    """Return right singular vectors with singular values < tol.

    Also includes any extra null directions (cols > rows), returned as
    zero-singular vectors.
    """
    U, S, Vt = np.linalg.svd(M, full_matrices=True)
    nulls: list[np.ndarray] = []
    sigmas: list[float] = []
    # Right singular vectors are rows of Vt, length = n_col
    n_col = M.shape[1]
    for k in range(n_col):
        s = S[k] if k < len(S) else 0.0
        if s < tol:
            nulls.append(Vt[k].copy())
            sigmas.append(float(s))
    return nulls, sigmas


def scan_mm_equation(
    loop: tuple[int, ...], edge_idx: int, D: int,
    lam_values: list[float] = (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 100.0),
    sv_tol: float = 1e-10,
    include_products: bool = True,
) -> ScanResult:
    """Full scan. Returns ScanResult with candidates, matrix, null vectors."""
    linear, products = enumerate_candidates(
        loop, edge_idx, D, include_products=include_products
    )
    lams = list(lam_values)
    M = build_matrix(linear, products, lams)
    nulls, sigmas = null_vectors(M, tol=sv_tol)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return ScanResult(
        loop=canonical(loop),
        edge_idx=edge_idx,
        D=D,
        linear_terms=linear,
        product_terms=products,
        lam_values=lams,
        matrix=M,
        singular_values=S,
        null_vectors=nulls,
        null_sigmas=sigmas,
    )


# ========================================================================
# Pretty-printing + validation
# ========================================================================


def format_equation(
    coeffs: np.ndarray,
    linear: list[LinearTerm],
    products: list[ProductTerm],
    eps: float = 1e-10,
) -> str:
    """Human-readable string of the equation Σ c_j term_j = 0."""
    n_lin = len(linear)
    parts: list[str] = []
    # Normalize: divide by the largest coefficient so it's monic
    idx_max = int(np.argmax(np.abs(coeffs)))
    scale = coeffs[idx_max]
    cs = coeffs / scale
    for j, c in enumerate(cs):
        if abs(c) < eps:
            continue
        sign = "+" if c >= 0 else "-"
        mag = abs(c)
        if j < n_lin:
            term = f"W[{linear[j].name}={linear[j].loop}]"
        else:
            p = products[j - n_lin]
            la = linear[p.idx_a]
            lb = linear[p.idx_b]
            term = f"W[{la.name}]·W[{lb.name}]"
        parts.append(f"{sign} {mag:.6e} {term}")
    return " ".join(parts) + " = 0"


def validate_equation(
    coeffs: np.ndarray,
    linear: list[LinearTerm],
    products: list[ProductTerm],
    lam_test: float,
    W_func: Callable[[tuple[int, ...], float], float] | None = None,
) -> float:
    """Evaluate Σ coeffs_j * term_j at lam_test. Returns |residual|.

    If W_func is None, uses qcd2_wilson_loop (ground truth).

    For a linear term with lambda_power=p, the column value is
    `lam^p * W[loop]`. Product terms use the RAW W[loop] values (no
    extra lambda factor) since products in the candidate basis are
    built from power-0 entries only.
    """
    wf: Callable[[tuple[int, ...], float], float] = (
        W_func if W_func is not None else qcd2_wilson_loop
    )

    n_lin = len(linear)
    raw_W: list[float] = []
    lin_values: list[float] = []
    for t in linear:
        if len(t.loop) == 0:
            w = 1.0
        else:
            w = wf(t.loop, lam_test)
        raw_W.append(w)
        lin_values.append(w * (lam_test ** t.lambda_power))
    total = 0.0
    for j, c in enumerate(coeffs):
        if j < n_lin:
            total += c * lin_values[j]
        else:
            p = products[j - n_lin]
            total += (
                c * raw_W[p.idx_a] * raw_W[p.idx_b]
                * (lam_test ** p.lambda_power)
            )
    return abs(total)


def summarize_scan(result: ScanResult, eps: float = 1e-10) -> str:
    """Multi-line summary for printing."""
    lines: list[str] = []
    lines.append(
        f"=== MM scan: loop={result.loop}, edge_idx={result.edge_idx}, "
        f"D={result.D} ==="
    )
    lines.append(f"Linear candidates (n={len(result.linear_terms)}):")
    for t in result.linear_terms:
        lines.append(f"  {t.name}: {t.loop}")
    lines.append(
        f"Product candidates (n={len(result.product_terms)}): "
        + ", ".join(p.name for p in result.product_terms)
    )
    lines.append(
        f"Lambda values: {result.lam_values}  "
        f"Matrix shape: {result.matrix.shape}"
    )
    lines.append(
        "Singular values: "
        + ", ".join(f"{s:.3e}" for s in result.singular_values)
    )
    lines.append(
        f"Null vectors found (sigma < {eps:.1e}): {len(result.null_vectors)}"
    )
    for k, (v, s) in enumerate(
        zip(result.null_vectors, result.null_sigmas)
    ):
        lines.append(f"\n  Null vector {k} (sigma={s:.3e}):")
        eq = format_equation(
            v, result.linear_terms, result.product_terms, eps=eps
        )
        lines.append(f"    {eq}")
        # Validate at fresh lambda
        for lam_test in (7.0, 3.14159):
            res = validate_equation(
                v, result.linear_terms, result.product_terms, lam_test
            )
            lines.append(f"    |residual| at lam={lam_test}: {res:.3e}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Plaquette edge 0
    plaq = (1, 2, -1, -2)
    result = scan_mm_equation(plaq, edge_idx=0, D=2)
    print(summarize_scan(result))
