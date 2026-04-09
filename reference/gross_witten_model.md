# Gross-Witten-Wadia Model — Reference

## Definition

Partition function (arXiv:2308.06320):

Z(N,t) = ∫ dU/(vol U(N)) exp(N/(2t) (Tr U + Tr U†))

where U ∈ U(N) and t is the 't Hooft coupling. Phase transition at **t_c = 1** (3rd order).

In our convention (`--coupling g`): t = g² (so phase transition at g = 1).

## Exact Eigenvalue Density

Eigenvalues of U are e^{iθ_j}, density ρ(θ):

### Strong coupling (t > 1, ungapped)

ρ(θ) = (1/2π)(1 + (1/t)cos θ) on [-π, π]

Wilson loops: w_1 = 1/(2t), w_n = 0 for n ≥ 2.

### Weak coupling (t < 1, gapped)

ρ(θ) = (1/(πt)) cos(θ/2) √(t - sin²(θ/2)) on [-θ_c, θ_c]

where sin²(θ_c/2) = t.

Wilson loops (verified numerically):
- w_1 = 1 - t/2
- w_2 = (1-t)²

At t = 1: w_1 = 1/2 (continuous), w_2 = 0 (continuous).

## Why U = exp(iM) Does NOT Reduce to Hermitian Model

The mapping U = exp(iM) is algebraically correct, but the **measures differ**:

- Hermitian: ∫ dM · Π(θ_j - θ_k)² · exp(-N Tr V(M))
- Unitary: ∫ dθ · Π|2sin((θ_j-θ_k)/2)|² · exp(-(N/t)Σcos θ_j)

The Vandermonde determinant on the circle (sin² vs polynomial²) gives a **different saddle-point equation** and different SD/loop equations. Verified: Hermitian SD equations are NOT satisfied by exact GW moments.

## Saddle-Point Equation

From the Haar measure + action:

P.V. ∫ ρ(φ) cot((θ-φ)/2) dφ = (1/t) sin(θ)

Via the Hilbert transform on the circle, this becomes (for full-circle support):

2 Σ_{n≥1} w_n sin(nθ) = (1/t) sin(θ)

Giving w_1 = 1/(2t), w_n = 0 for n ≥ 2 (strong-coupling solution).

For the gapped phase, the Hilbert transform approach doesn't directly apply because the density has compact support on [-θ_c, θ_c] ⊂ [-π, π].

## Loop Equation (Open Problem for Implementation)

The Hermitian SD equation Σ_k v_k m_{n+k} = splitting does **NOT** apply to unitary moments.

Tested candidate equations:
- (1/(2t))(w_{n-1} - w_{n+1}) = Σ w_j w_{n-j} → fails for t < 1
- (1/(2t))(w_{n-1} - w_{n+1}) = n w_n + Σ w_j w_{n-j} → works at n=1, fails for n≥2 at t < 1
- All simple forms fail for the gapped phase

The correct unitary loop equation involves the resolvent on the unit circle:
- Outer resolvent R⁺(z) for |z| > 1
- Inner resolvent R⁻(z) for |z| < 1
- The spectral curve relates these

**Needed**: derive the correct moment recursion from the resolvent equation, or use the saddle-point equation directly as the loss function.

## Implementation Strategy (Proposed)

### Approach A: Saddle-point equation as loss

Parameterize ρ(θ) directly (e.g., Fourier coefficients or support endpoint + polynomial), minimize the saddle-point equation residual:

L = ∫ |P.V. ∫ ρ cot((θ-φ)/2) dφ - (1/t)sin θ|² dθ + constraints

Constraints: ∫ρ = 1, ρ ≥ 0.

### Approach B: Toeplitz moment matrix + resolvent

Parameterize Wilson loops w_n as optimization variables. Enforce:
1. Toeplitz moment matrix T_{ij} = w_{|i-j|} is PSD
2. The resolvent equation (once correctly derived) as the loss
3. Normalization w_0 = 1

## Key References

- Gross & Witten, "Possible third-order phase transition in large-N lattice gauge theory", Phys. Rev. D 21 (1980) 446
- Wadia, "N = ∞ phase transition in a class of exactly soluble model lattice gauge theories", Phys. Lett. B 93 (1980) 403
- Ahmed & Dunne, "Large N expansion of Wilson loops in the GWW matrix model", J. Phys. A 51 (2018) 055401 [arXiv:1707.09625]
- Dunne & Ünsal, "Complex eigenvalue instantons and the Fredholm determinant expansion in the GWW model", JHEP 01 (2024) 129 [arXiv:2308.06320]
- Mariño, "Chern-Simons Theory, Matrix Models, and Topological Strings", Oxford (2005)
