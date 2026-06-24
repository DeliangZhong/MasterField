"""Operator (Cuntz-Fock) realization of the master field.

Milestone 1: the one-matrix master field in closed form. The master field of a
one-matrix model is fixed by its free cumulants (Voiculescu): build the operator
M̂ = â + Σ_n M_n (â†)^n on the truncated Cuntz-Fock space and read its vacuum
moments. This validates that the operator representation reproduces the known
one-matrix answer (no optimization needed at one matrix). The optimization engine
for the genuinely-unsolvable multi-matrix case arrives in Milestone 2.
"""

import numpy as np

from matrix_master_field.cuntz_fock import CuntzFockSpace
from matrix_master_field.one_matrix import (
    r_transform_from_moments,
    voiculescu_coefficients,
)


def one_matrix_master_field_from_moments(target_moments, fock_length: int = 10):
    """Build the one-matrix master-field operator from target moments.

    Args:
        target_moments: array m_0, m_1, ..., m_K (m_0 must be 1).
        fock_length: truncation length L of the Cuntz-Fock space.

    Returns:
        (M_hat, model_moments): the assembled operator and its vacuum moments
        tr[M̂^p] = ⟨Ω|M̂^p|Ω⟩ for p = 0..K.
    """
    target_moments = np.asarray(target_moments, dtype=float)
    K = len(target_moments) - 1

    # Free cumulants κ_n, then Voiculescu coefficients M_n = κ_{n+1}.
    kappa = r_transform_from_moments(target_moments)
    v_coeffs = voiculescu_coefficients(kappa)

    fock = CuntzFockSpace(n_matrices=1, max_length=fock_length)
    n_coeffs = min(len(v_coeffs), fock_length)
    M_hat = fock.build_master_field_voiculescu(v_coeffs[:n_coeffs], matrix_idx=0)
    model_moments = fock.compute_moments(M_hat, max_power=K)
    return M_hat, model_moments
