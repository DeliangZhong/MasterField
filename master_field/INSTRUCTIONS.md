# Master Field ML Computation — Claude Code Instructions

## Overview

This project numerically constructs the **master field** for large-$N$ matrix models
using modern ML optimisation. The physics is from Gopakumar–Gross (hep-th/9411021)
and the Jevicki–Sakita–Rodrigues collective field program.

## Setup

```bash
pip install jax jaxlib numpy scipy matplotlib optax flax cvxpy torch --break-system-packages
```

## Project Structure

```
master_field/
├── cuntz_fock.py          # Cuntz algebra, Fock space truncation, operator reps
├── one_matrix.py          # Exact one-matrix model: resolvent, eigenvalue density
├── schwinger_dyson.py     # Loop equations / Schwinger-Dyson for multi-matrix
├── collective_field.py    # Jevicki-Sakita V_eff, master variable parametrisation
├── neural_master_field.py # Neural-network ansatz for the master field (JAX/Flax)
├── bootstrap_sdp.py       # SDP validation via cvxpy
├── train.py               # Main training loop with curriculum learning
├── config.py              # Hyperparameters and model selection
└── visualize.py           # Eigenvalue densities, convergence plots
```

## Computational Pipeline

### Stage 1: Validation on exactly solvable models (MUST PASS before proceeding)

1. **Gaussian one-matrix model** ($V = \frac{1}{2}M^2$):
   - Exact: Wigner semicircle $\rho(x) = \frac{1}{2\pi}\sqrt{4-x^2}$
   - Exact moments: $\text{tr}[M^{2k}] = C_k$ (Catalan numbers)
   - Master field: $\hat{M} = \hat{a} + \hat{a}^\dagger$
   - **Test**: Verify Cuntz–Fock VEVs reproduce Catalan numbers to machine precision

2. **Quartic one-matrix model** ($V = \frac{1}{2}M^2 + \frac{g}{4}M^4$):
   - Exact resolvent from cubic equation
   - **Test**: Neural optimiser recovers exact eigenvalue density

3. **Two independent Gaussian matrices**:
   - Free random variables — moments factorise via freeness
   - **Test**: $\text{tr}[M_1 M_2 M_1 M_2] = \text{tr}[M_1^2]\text{tr}[M_2^2] + ...$

### Stage 2: Non-trivial coupled models

4. **Two-matrix model with quartic coupling** ($V = \frac{1}{2}\text{Tr}(M_1^2 + M_2^2) + g\,\text{Tr}(M_1 M_2 M_1 M_2)$):
   - No closed-form solution; SD equations must be solved numerically
   - Benchmark against Monte Carlo at finite $N$ (extrapolated)

5. **Yang–Mills matrix quantum mechanics** (2 matrices):
   - $H = \frac{1}{2}\text{Tr}(\Pi_1^2 + \Pi_2^2) - \frac{g^2}{4}\text{Tr}[M_1, M_2]^2$
   - Benchmark against Rodrigues et al. JHEP 2024

### Stage 3: Toward QCD

6. **One-plaquette model** (Gross–Witten):
   - Exact solution with 3rd-order phase transition
   - Two master fields (weak/strong coupling)

7. **2D lattice Yang–Mills** with small number of plaquettes

## Key Physics Encoded in the Code

### The Schwinger-Dyson equations (loop equations)

For a single matrix with potential $V(M)$:
$$\langle \frac{1}{N}\text{Tr}[V'(M) M^n]\rangle = \sum_{j=0}^{n-1} \langle\frac{1}{N}\text{Tr}[M^j]\rangle \langle\frac{1}{N}\text{Tr}[M^{n-j-1}]\rangle$$

These are EXACT at $N=\infty$ and form a closed set of recursion relations for the moments.

### The loss function

The total loss is:
$$\mathcal{L} = \mathcal{L}_{\text{SD}} + \lambda_{\text{EOM}} \mathcal{L}_{\text{EOM}} + \lambda_{\text{reg}} \mathcal{L}_{\text{reg}}$$

where:
- $\mathcal{L}_{\text{SD}} = \sum_{n} |(\text{SD equation}_n)\text{ residual}|^2$
- $\mathcal{L}_{\text{EOM}} = \sum_{\psi} |\langle\Omega|[V'(\hat{M}) - 2\hat{\Pi}]|\psi\rangle|^2$
- $\mathcal{L}_{\text{reg}}$: spectral regularisation to keep eigenvalues of $\Omega$ non-negative

### PSD constraint handling

The moment matrix $\Omega_{ij} = \text{tr}[w_i^\dagger w_j]$ (where $w_i$ are words in the matrices) must be
positive semidefinite. We enforce this by parametrising:
$$\Omega = L L^\dagger, \qquad L \text{ lower-triangular (Cholesky factor)}$$

The neural network outputs $L$; $\Omega$ is automatically PSD.

## How to Run

```bash
# Stage 1: Validate on Gaussian model
python train.py --model gaussian --validate

# Stage 1: Quartic one-matrix
python train.py --model quartic --coupling 0.5

# Stage 2: Two coupled matrices
python train.py --model two_matrix_coupled --coupling 1.0

# Stage 2: Yang-Mills QM
python train.py --model yang_mills_qm --coupling 1.0 --max_word_length 8
```

## Expected Outputs

- `results/moments_*.npy`: optimised loop moments
- `results/eigenvalue_density_*.png`: reconstructed eigenvalue distributions
- `results/convergence_*.png`: loss curves
- `results/sd_residuals_*.txt`: Schwinger-Dyson equation residuals (should → 0)
- `results/master_field_coefficients_*.npy`: the R-transform / master field coefficients
