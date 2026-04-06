"""
visualize.py — Plotting utilities for master field results.
"""
import numpy as np
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_convergence(losses, output_dir, tag):
    """Plot training loss convergence."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (SD residuals)")
    ax.set_title(f"Master Field Optimisation — {tag}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"convergence_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved convergence_{tag}.png")


def plot_moments(moments_ml, moments_exact, output_dir, tag):
    """Compare ML moments with exact (if available)."""
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = np.arange(0, len(moments_ml), 2)
    ax.plot(ks, moments_ml[::2], 'bo-', label='ML', markersize=6)
    if moments_exact is not None:
        ks_e = np.arange(0, min(len(moments_exact), len(moments_ml)), 2)
        ax.plot(ks_e, moments_exact[::2][:len(ks_e)], 'r^--', label='Exact', markersize=6)
    ax.set_xlabel("Moment order k")
    ax.set_ylabel(r"$\mathrm{tr}[M^k]$")
    ax.set_title(f"Moments — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"moments_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved moments_{tag}.png")


def plot_eigenvalue_density(moments, output_dir, tag, n_points=500):
    """Reconstruct and plot the eigenvalue density from moments.
    
    Uses the maximum entropy method or direct Stieltjes inversion.
    """
    if not HAS_MPL:
        return
    
    # Simple approach: Stieltjes inversion of the resolvent
    # R(ζ) = Σ m_k / ζ^{k+1}
    # ρ(x) = -(1/π) Im R(x + iε)
    
    # Determine support range from moments
    m2 = moments[2] if len(moments) > 2 else 1.0
    a_est = 2 * np.sqrt(max(m2, 0.1))
    
    x = np.linspace(-a_est * 1.2, a_est * 1.2, n_points)
    eps = 0.05  # small imaginary part
    
    rho = np.zeros(n_points)
    for i, xi in enumerate(x):
        zeta = xi + 1j * eps
        R = 0.0
        for k in range(len(moments)):
            R += moments[k] / zeta**(k + 1)
        rho[i] = -np.imag(R) / np.pi
    
    rho = np.maximum(rho, 0)
    
    # Also plot exact for comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, rho, 'b-', linewidth=2, label='ML (Stieltjes)')
    
    # Exact Wigner for Gaussian
    if 'gaussian' in tag.lower():
        from one_matrix import gaussian_density
        rho_exact = gaussian_density(x)
        ax.plot(x, rho_exact, 'r--', linewidth=2, label='Wigner semicircle')
    elif 'quartic' in tag.lower():
        try:
            g = float(tag.split('g')[1])
            from one_matrix import quartic_eigenvalue_density
            x_e, rho_e = quartic_eigenvalue_density(g)
            ax.plot(x_e, rho_e, 'r--', linewidth=2, label='Exact')
        except:
            pass
    
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho(x)$")
    ax.set_title(f"Eigenvalue Density — {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"eigenvalue_density_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved eigenvalue_density_{tag}.png")


def plot_moment_matrix_spectrum(Omega, output_dir, tag):
    """Plot the eigenvalue spectrum of the moment matrix."""
    if not HAS_MPL:
        return
    eigvals = np.linalg.eigvalsh(Omega)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.maximum(eigvals, 1e-16), 'bo-', markersize=4)
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Moment Matrix Spectrum — {tag}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"moment_spectrum_{tag}.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved moment_spectrum_{tag}.png")
