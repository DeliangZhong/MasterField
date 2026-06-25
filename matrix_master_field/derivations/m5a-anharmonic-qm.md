# M5a derivation — stationarity recursion for `H = p² + x² + g x⁴`

Conventions (HHK arXiv:2004.10212, Eq 1): `ℏ=1`, `[x,p]=+i`, eigenstate `H|E⟩=E|E⟩`,
`m_k ≡ ⟨E|x^k|E⟩`, `m_0=1`, odd moments vanish by parity. This file re-derives the
recursion **from scratch** (it is not transcribed); it reproduces HHK Eq 6.

## D0 — canonical commutator (oscillator representation)
`x̂=(â+â†)/√2`, `p̂=−i(â−â†)/√2`, `[â,â†]=1`:
`[x̂,p̂] = (−i/2)[â+â†, â−â†] = (−i/2)(−2[â,â†]) = i`. ✓
Also `H₀ = p̂²+x̂² = 2â†â+1` (since `x̂²+p̂² = ½·2(ââ†+â†â) = 2â†â+1`), so at `g=0`,
`E_n=2n+1`, `E₀=1`, `m₂=⟨0|x̂²|0⟩=½`.

## D1 — energy relation (sandwich `p²`)
`p² = H − x² − g x⁴`, and `⟨x^{t-1}H⟩ = E⟨x^{t-1}⟩` in an eigenstate, so
> `⟨x^{t-1} p²⟩ = E m_{t-1} − m_{t+1} − g m_{t+3}`  (= HHK Eq 5).

## D2 — Heisenberg building blocks
From `[p,x]=−i` and the Leibniz rule:
- `[p, x^t] = −i t x^{t-1}` ⟹ `[H, x^t] = [p², x^t] = −i t (p x^{t-1} + x^{t-1} p)`.
- `[H, p] = [x²+g x⁴, p] = i(2x + 4g x³)` (= `i V'(x)`; Heisenberg `ṗ = −V'`).

## D3 — stationarity recursion (closes the system)
Stationarity `⟨[H,O]⟩=0` with `O = x^t p`:
`[H, x^t p] = [H,x^t]p + x^t[H,p] = −i t (p x^{t-1}+x^{t-1}p)p + i x^t(2x+4g x³)`.
Taking `⟨·⟩=0`:
> `t(⟨p x^{t-1} p⟩ + ⟨x^{t-1} p²⟩) = 2 m_{t+1} + 4g m_{t+3}`   (R2)

Reduce `⟨p x^{t-1} p⟩` using `[x,p]=i`:
`p x^{t-1} p = x^{t-1}p² − i(t-1) x^{t-2}p`. From `O=x^{t-1}` stationarity
`⟨x^{t-2}p + p x^{t-2}⟩=0` and `⟨[x^{t-2},p]⟩ = i(t-2) m_{t-3}`, so
`⟨x^{t-2}p⟩ = (i/2)(t-2) m_{t-3}`, hence
`⟨p x^{t-1} p⟩ = ⟨x^{t-1}p²⟩ + ½(t-1)(t-2) m_{t-3}`.
Substitute this and D1 into (R2) and multiply by 2:

> **`4 t E m_{t-1} + t(t-1)(t-2) m_{t-3} − 4(t+1) m_{t+1} − 4g(t+2) m_{t+3} = 0`**  (D3 = HHK Eq 6)

Consequences: `t=1` ⟹ `m₄=(E−2m₂)/(3g)`; the `g=0` limit forces `E=2m₂` (so `E₀=1`,
`m₂=½`). For fixed `E` the recursion is linear in the moments, expressing every even
`m_{2k}` affinely in the single free `m₂` (with `m_0=1`) — the SDP fixes `E` (LMI in `m₂`)
and bisects `E`.

## Verification (`tests/test_qm_recursion.py`)
- **D3 numeric:** `loss.qm_anharmonic_recursion_residual(m, E₀, g)` on exact-diag moments
  (`qm_fock.ground_state`/`moment`) is `< 1e-9` for `g∈{0,0.5,1,2}`.
- **D3 closed form:** `4E₀ − 8m₂ − 12g m₄ = 0`.
- **D1 operator form:** `⟨x^{t-1}p²⟩` computed directly from the operators equals
  `E₀ m_{t-1} − m_{t+1} − g m_{t+3}` for `t=1,3,5`.
