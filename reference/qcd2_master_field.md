# QCD₂ Master Field — Reference

## Exact Result (Migdal 1975, 't Hooft 1974)

For 2D lattice Yang-Mills in the N=∞ limit on an infinite square lattice, the Wilson loop for a non-self-intersecting closed loop C satisfies the **area law**:

**W[C] = exp(-λ · Area(C) / 2)**

where:
- λ = g²N is the 't Hooft coupling (held fixed as N→∞)
- Area(C) = number of lattice plaquettes enclosed by C

For the **elementary plaquette** (1×1): W[□] = exp(-λ/2).
For an **m×n rectangle**: W[m×n] = exp(-λmn/2).

Self-intersecting loops and twisted loops require the Kazakov-Wadia or Migdal factorization prescription; they reduce to polynomials in W for simpler sub-loops.

## Spacetime-Independent Master Field

From Gopakumar-Gross (hep-th/9411021) §1, eq. 1.6–1.7: at N=∞ the master gauge field Ā_μ can be taken spacetime-independent by a gauge transformation. On the lattice, this means the master link variables Ū_μ(x) = Ū_μ depend only on the direction index μ, not on the site x.

The gauge-invariant data is a set of **D unitary operators** Û_μ acting on a Cuntz-Fock space |Ω⟩, |i⟩, |ij⟩, .... Wilson loop expectation values become operator vacuum matrix elements:

**W[C] = ⟨Ω| Û_{μ_1} Û_{μ_2} ⋯ Û_{μ_k} |Ω⟩**

for a closed path C = (μ_1, μ_2, ..., μ_k) with the convention:
- μ_i > 0: step forward in direction |μ_i|, contributes Û_{|μ_i|}
- μ_i < 0: step backward in direction |μ_i|, contributes Û_{|μ_i|}†

## Gaussian Ansatz in Axial Gauge (Gopakumar-Gross §5)

In **axial gauge** (Ū_2 = I, choose temporal gauge suitably), 2D lattice QCD at large N has a Gaussian master field. Equivalently, on the lattice of arbitrary D = 2 setup we can parametrize:

**Û_μ = exp(i α · (â_μ + â_μ†))**

where:
- â_μ, â_μ† are Cuntz-Fock annihilation/creation operators (one per lattice direction)
- α = α(λ) is a single real parameter determined by matching the plaquette

The Hermitian generator M̂_μ = â_μ + â_μ† is precisely the Gaussian master field for a single Hermitian matrix (Wigner semicircle spectrum). Û_μ is unitary *exactly* on the infinite Cuntz-Fock space; *approximately* on any truncation.

### Fixing α

Solve the plaquette condition:

W[□]_ML = ⟨Ω| Û_1 Û_2 Û_1† Û_2† |Ω⟩ = exp(-λ/2)

Numerically find α ∈ (0, π) via root-finding (brentq). At truncation L, the answer depends on L, converging to the exact answer as L→∞.

## Lattice Makeenko-Migdal Equation (Chatterjee 2019, rigorous)

For any link e ∈ C:

**λ W[C] = Σ_{P ∋ e} W[P_e ∘ C] − Σ_{splits} W[C_1] W[C_2]**

where:
- P_e ranges over the 2(D−1) plaquettes containing link e. In 2D: 2 plaquettes per link.
- P_e ∘ C means inserting the plaquette loop at link e in C
- The splits are at self-intersections of C; for a simple loop without self-intersections, the sum is empty

At N=∞ this closes on single-trace expectation values (large-N factorization). The RHS is bilinear in W, the LHS is linear.

## Loop Encoding Conventions

A lattice loop is a sequence of signed step indices. In D dimensions, steps take values in {±1, ±2, ..., ±D}. A loop must close: starting from site x, after applying all steps we must return to x.

Example (D=2):
- Elementary plaquette: (1, 2, −1, −2). Starts at origin, go +x, +y, −x, −y.
- 2×1 rectangle: (1, 1, 2, −1, −1, −2).
- Backtrack-trivial loop (cancels): (1, −1). Reduces to empty.

Two loops are **equivalent** if they are related by:
1. **Cyclic rotation** (trace is cyclic): (1,2,−1,−2) ≡ (2,−1,−2,1).
2. **Backtrack reduction**: adjacent μ, −μ pairs cancel (from unitarity). (1,−1,2,−2) ≡ ().
3. **Lattice symmetry** (hyperoctahedral group B_D): permutations and sign flips of direction labels. In D=2: 8 elements.

The canonical form is backtrack-reduced, cyclically smallest rotation.

## Verification Criteria (Phase 0 — what we actually observe)

- **Plaquette**: |W[□]_ML − e^{−λ/2}| < 10^{−10} by construction (we fit α to match this).
- **α convergence with L**: α(λ, L) converges rapidly. At λ=1: α(L=6)=0.95176911, α(L=10)=0.95176915 — stable to 8 decimals.
- **Unitarity**: ||Û Û† − I||_F < 10^{−12} at all truncations (matrix exponential is unitary to machine precision).
- **2×1 rectangle** at λ=1: W[2×1]_ML = 0.37299, exact e^{−1} = 0.36788. Error **~5e-3 and L-independent** from L=6 onward. This is the key finding.
- **Larger loops**: errors grow with area. W[2×2] − e^{−2} ≈ 0.135 (error 37% at λ=1).

## Key Finding (Phase 0)

**The single-parameter Gaussian ansatz Û_μ = exp(iα(â_μ+â_μ†)) is NOT the full QCD₂ master field.**

In 2D lattice YM with the Wilson action, the exact factorization W[C₁∪C₂] = W[C₁]·W[C₂] for disjoint-area sub-loops means:
- W[2×1] = (W[□])² exactly
- W[m×n] = (W[□])^{mn} exactly

Our single-α ansatz matches the plaquette by construction but fails the factorization because the finite-dimensional Cuntz-Fock operators Û_1, Û_2 don't satisfy the full algebraic structure needed for the area law to emerge from the operator product.

This is **expected** and **physically interesting**:
1. Gopakumar-Gross §5's "Gaussian master field for QCD₂" is a continuum/scaling-limit statement. At fixed lattice truncation, the one-parameter ansatz is insufficient.
2. The correct master field has richer structure — more creation operators, or coefficients beyond a single α. A multi-coefficient Voiculescu-type expansion Û_μ = Σ_n c_n (â_μ†)^n + h.c. would have more degrees of freedom to match additional Wilson loops.
3. Phase 0 validates the **infrastructure** (Cuntz-Fock + lattice loops + unitary operators + Wilson loop VEVs), not the simplest ansatz.
4. The gap between "one-α fit" and "full area law" motivates Phase 1: **train a neural parametrization of the operator coefficients** (or of W[C] directly) against many Wilson loops simultaneously.

## Bibliography

- Migdal, "Recursion equations in gauge field theories", Zh. Eksp. Teor. Fiz. 69 (1975) 810. Original loop equations.
- 't Hooft, "A two-dimensional model for mesons", Nucl. Phys. B 75 (1974) 461. Exact N=∞ QCD₂.
- Kazakov-Wadia, "Wilson Loop Functional in Large N Gauge Theories", Phys. Lett. B 132 (1983). Factorization.
- Gopakumar-Gross, "Mastering the Master Field", hep-th/9411021. Cuntz-Fock construction, §5 QCD₂.
- Kazakov-Zheng, "Analytic and numerical bootstrap for one-matrix model and 'unsolvable' two-matrix model", arXiv:2108.04830 / 2203.11360. Lattice MM, SDP.
- Chatterjee, "Rigorous solution of strongly coupled SO(N) lattice gauge theory in the large N limit", Commun. Math. Phys. 366 (2019). Rigorous MM.
- Qiao-Zheng, "Bootstrapping Yang-Mills matrix model and its large-N solution", arXiv:2601.04316. Systematic loop enumeration.
