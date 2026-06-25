# matrix_master_field/tm_qm_relations.py
"""M5c — two-matrix-QM stationarity loop equations + SU(N) Gauss law (T5, T2).

Words are tuples of ints: 0=X̃, 1=Ỹ, 2=P̃_X, 3=P̃_Y. The g=0 ground state is Gaussian,
so g0_moment(word) is an exact Wick value used to verify every relation BEFORE the SDP
(Task 4) depends on it. Heisenberg EOM (ℏ=1, [X,P]=i):
  [H,X̃]=−2i P̃_X,  [H,P̃_X]= i(2m² X̃ − 2λ·F_X),  F_X = [Ỹ,[X̃,Ỹ]] = 2ỸX̃Ỹ−Ỹ²X̃−X̃Ỹ²,
and X↔Y. ⟨[H,Tr w]⟩=0 ⇒ replace each letter by its [H,·] and sum (the QM loop equation).
"""
import numpy as np

POS = {0: 2, 1: 3}        # X̃→P̃_X, Ỹ→P̃_Y  (the conjugate momentum letter)
MOM = {2: 0, 3: 1}        # P̃_X→X̃, P̃_Y→Ỹ
OTHER = {0: 1, 1: 0}      # the other position letter


def g0_moment(word, m):
    """Exact g=0 ordered moment via Gaussian Wick on the free matrix oscillator.

    Two-point seeds (per matrix, leading large-N, normalized tr): X̃X̃=1/(2m), P̃P̃=m/2,
    X̃P̃=i/2, P̃X̃=−i/2; X and Y sectors independent. Wick-expand ordered words by summing
    over pairings with the planar (nearest-neighbour, non-crossing) contraction weights.
    """
    w = tuple(word)
    if len(w) % 2 == 1:
        return 0.0 + 0.0j
    if w == ():
        return 1.0 + 0.0j
    return _wick(w, m)


def _two_point(a, b, m):
    # sector check: X̃/P̃_X are 'X' (0,2); Ỹ/P̃_Y are 'Y' (1,3).
    sec = {0: "X", 2: "X", 1: "Y", 3: "Y"}
    if sec[a] != sec[b]:
        return 0.0 + 0.0j
    is_pos = {0: True, 1: True, 2: False, 3: False}
    pa, pb = is_pos[a], is_pos[b]
    if pa and pb:
        return 1.0 / (2.0 * m) + 0.0j      # ⟨X̃X̃⟩
    if (not pa) and (not pb):
        return m / 2.0 + 0.0j              # ⟨P̃P̃⟩
    if pa and not pb:
        return 0.5j                        # ⟨X̃P̃⟩ = i/2
    return -0.5j                           # ⟨P̃X̃⟩ = −i/2


def _wick(w, m):
    # Sum over non-crossing pairings with ordered two-point weights (planar large-N).
    n = len(w)
    if n == 0:
        return 1.0 + 0.0j
    total = 0.0 + 0.0j
    a = w[0]
    for k in range(1, n, 2):  # pair position 0 with k so [1..k-1] is closed (non-crossing)
        inner = _wick(w[1:k], m)
        outer = _wick(w[k + 1:], m)
        total += _two_point(a, w[k], m) * inner * outer
    return total


def stationarity_terms(word):
    """⟨[H,Tr word]⟩=0 as a list of (coeff(m,lam), substituted_word). At λ=0 the
    commutator force drops; the λ-terms carry F_X = 2ỸX̃Ỹ−Ỹ²X̃−X̃Ỹ² insertions."""
    w = tuple(word)
    terms = []
    for k, c in enumerate(w):
        pre, post = w[:k], w[k + 1:]
        if c in (0, 1):                       # position letter → [H,X̃]=−2i P̃_X
            terms.append((lambda m, lam, p=POS[c]: -2j, pre + (POS[c],) + post))
        else:                                  # momentum letter → [H,P̃]=i(2m² X̃ − 2λ F)
            x = MOM[c]                         # the position partner
            y = OTHER[x]
            terms.append((lambda m, lam, x=x: 2j * m**2, pre + (x,) + post))
            # −2iλ F_x with F_x = 2 y x y − y y x − x y y
            terms.append((lambda m, lam: -2j * lam * 2.0, pre + (y, x, y) + post))
            terms.append((lambda m, lam: +2j * lam, pre + (y, y, x) + post))
            terms.append((lambda m, lam: +2j * lam, pre + (x, y, y) + post))
    return terms


def gauss_terms(O):
    """SU(N) Gauss law per canonical pair: m[(0,2)+O]−m[(2,0)+O]=i·m[O] (X), and (1,3)/(3,1) (Y).
    Position-first MINUS momentum-first (matches the anchor m[X̃P̃]=+i/2, m[P̃X̃]=−i/2). One
    relation for the X pair (caller iterates pairs as needed)."""
    O = tuple(O)
    return [(1.0, (0, 2) + O), (-1.0, (2, 0) + O), (-1j, O)]
