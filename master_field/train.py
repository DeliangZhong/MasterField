#!/usr/bin/env python3
"""
train.py — Main entry point for master field ML computation.

Usage:
    python train.py --model gaussian --validate
    python train.py --model quartic --coupling 0.5
    python train.py --model quartic --coupling 1.0 --max_word_length 14 --n_epochs 10000
    python train.py --model two_matrix_coupled --coupling 1.0 --max_word_length 6
    
See INSTRUCTIONS.md for the full computational pipeline.
"""
import argparse
import os
import json
import time
import numpy as np

import jax
import jax.numpy as jnp
from jax import random


def main():
    parser = argparse.ArgumentParser(description="Master Field ML Optimisation")
    parser.add_argument("--model", type=str, default="gaussian",
                        choices=["gaussian", "quartic", "sextic", 
                                 "two_matrix_coupled", "yang_mills_qm",
                                 "gross_witten"],
                        help="Matrix model to solve")
    parser.add_argument("--coupling", type=float, default=0.5,
                        help="Coupling constant g")
    parser.add_argument("--max_word_length", type=int, default=12,
                        help="Loop truncation level L")
    parser.add_argument("--n_epochs", type=int, default=5000,
                        help="Number of optimisation epochs")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation against exact solutions")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Run SDP bootstrap for bounds")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for output files")
    parser.add_argument("--interaction", type=str, default="commutator_squared",
                        choices=["commutator_squared", "quartic_mixed"],
                        help="Interaction type for two-matrix model")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    key = random.PRNGKey(args.seed)
    
    print("=" * 70)
    print(f"  Master Field ML — {args.model} model, g = {args.coupling}")
    print(f"  Truncation L = {args.max_word_length}, epochs = {args.n_epochs}")
    print("=" * 70)
    
    t0 = time.time()
    
    # ──────────────────────────────────────────────────────────
    # Stage 1: Validate Cuntz–Fock space (always)
    # ──────────────────────────────────────────────────────────
    print("\n[1] Cuntz–Fock space validation...")
    from cuntz_fock import CuntzFockSpace
    
    n_mat = 1 if args.model not in ["two_matrix_coupled", "yang_mills_qm"] else 2
    fock_L = min(args.max_word_length, 8)  # keep Fock space manageable
    fock = CuntzFockSpace(n_matrices=n_mat, max_length=fock_L)
    fock.verify_cuntz_relations()
    
    if args.model == "gaussian" and args.validate:
        print("\n  Gaussian master field M̂ = â + â†:")
        M_gauss = fock.x(0)
        moments = fock.compute_moments(M_gauss, max_power=min(2*fock_L, 12))
        catalan = [1, 1, 2, 5, 14, 42, 132]
        for k in range(0, min(len(moments), 13), 2):
            ck = catalan[k//2] if k//2 < len(catalan) else "?"
            print(f"    tr[M^{k}] = {moments[k]:.6f}  (Catalan: {ck})")
    
    # ──────────────────────────────────────────────────────────
    # Stage 2: Run ML optimisation
    # ──────────────────────────────────────────────────────────
    if args.model in ["gaussian", "quartic", "sextic"]:
        print(f"\n[2] One-matrix ML optimisation ({args.model})...")
        from neural_master_field import MasterFieldTrainer
        
        trainer = MasterFieldTrainer(
            args.model, n_matrices=1, coupling=args.coupling,
            max_word_length=args.max_word_length,
            lr=args.lr, n_epochs=args.n_epochs)
        
        params, losses = trainer.train(key, verbose=True)
        sol = trainer.get_solution()
        
        # Save results
        np.save(os.path.join(args.output_dir, f"moments_{args.model}_g{args.coupling}.npy"),
                sol['moments'])
        np.save(os.path.join(args.output_dir, f"losses_{args.model}_g{args.coupling}.npy"),
                np.array(losses))
        np.save(os.path.join(args.output_dir, f"free_cumulants_{args.model}_g{args.coupling}.npy"),
                sol['free_cumulants'])
        
        print(f"\n  Solution saved to {args.output_dir}/")
        print(f"  Final loss: {sol['final_loss']:.2e}")
        print(f"  Moments: {sol['moments'][:9]}")
        print(f"  Free cumulants: {sol['free_cumulants'][:6]}")
        
        # ──────────────────────────────────────────────────────
        # Validation against exact solution
        # ──────────────────────────────────────────────────────
        if args.validate or args.model == "gaussian":
            print(f"\n[3] Validation...")
            from one_matrix import gaussian_moments, quartic_moments_from_sd
            
            if args.model == "gaussian":
                m_exact = gaussian_moments(args.max_word_length)
            elif args.model == "quartic":
                m_exact = quartic_moments_from_sd(args.coupling, args.max_word_length)
            else:
                m_exact = None
            
            if m_exact is not None:
                print(f"\n  {'k':>4s}  {'ML':>14s}  {'Exact':>14s}  {'Error':>10s}")
                print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*10}")
                max_err = 0
                for k in range(0, min(len(sol['moments']), len(m_exact)), 2):
                    err = abs(sol['moments'][k] - m_exact[k])
                    max_err = max(max_err, err)
                    print(f"  {k:4d}  {sol['moments'][k]:14.8f}  {m_exact[k]:14.8f}  {err:10.2e}")
                print(f"\n  Max error: {max_err:.2e}")
                
                # Also verify via Cuntz–Fock space if model is solvable
                if args.model in ["gaussian", "quartic"]:
                    print("\n  Cuntz–Fock cross-check:")
                    from one_matrix import r_transform_from_moments, voiculescu_coefficients
                    kappa = r_transform_from_moments(m_exact[:min(len(m_exact), 10)])
                    v_coeffs = voiculescu_coefficients(kappa)
                    
                    fock_test = CuntzFockSpace(n_matrices=1, max_length=min(8, args.max_word_length))
                    M_test = fock_test.build_master_field_voiculescu(v_coeffs[:min(len(v_coeffs), 7)])
                    m_fock = fock_test.compute_moments(M_test, max_power=min(8, args.max_word_length))
                    for k in range(0, min(9, len(m_fock)), 2):
                        print(f"    tr[M̂^{k}] (Fock) = {m_fock[k]:.6f}")
        
        # ──────────────────────────────────────────────────────
        # SDP Bootstrap bounds
        # ──────────────────────────────────────────────────────
        if args.bootstrap:
            print(f"\n[4] SDP Bootstrap bounds...")
            from bootstrap_sdp import bootstrap_moment_bounds
            
            if args.model == "gaussian":
                v_prime = [0.0, 1.0]
            elif args.model == "quartic":
                v_prime = [0.0, 1.0, 0.0, args.coupling]
            else:
                v_prime = [0.0, 1.0]
            
            bounds = bootstrap_moment_bounds(v_prime, max_moment=min(args.max_word_length, 10))
            
            print("\n  ML solution vs bootstrap bounds:")
            for k, (lb, ub) in bounds.items():
                if lb is not None and ub is not None:
                    m_val = sol['moments'][k] if k < len(sol['moments']) else "N/A"
                    inside = "✓" if lb <= m_val <= ub else "✗"
                    print(f"    m_{k}: {lb:.6f} ≤ {m_val:.6f} ≤ {ub:.6f}  {inside}")
    
    elif args.model in ["two_matrix_coupled", "yang_mills_qm"]:
        print(f"\n[2] Two-matrix ML optimisation...")
        from neural_master_field import MultiMatrixTrainer
        
        trainer = MultiMatrixTrainer(
            n_matrices=2, coupling=args.coupling,
            max_word_length=args.max_word_length,
            interaction=args.interaction,
            lr=args.lr, n_epochs=args.n_epochs)
        
        params, losses = trainer.train(key, verbose=True)
        
        # Extract moment matrix
        Omega = np.array(trainer.params_to_moment_matrix(params))
        np.save(os.path.join(args.output_dir, f"moment_matrix_{args.model}_g{args.coupling}.npy"),
                Omega)
        np.save(os.path.join(args.output_dir, f"losses_{args.model}_g{args.coupling}.npy"),
                np.array(losses))
        
        eigvals = np.linalg.eigvalsh(Omega)
        print(f"\n  Moment matrix eigenvalues (first 5): {eigvals[:5]}")
        print(f"  min eigenvalue: {eigvals[0]:.2e} (should be ≥ 0)")
        print(f"  Final loss: {losses[-1]:.2e}")
    
    # ──────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────
    print(f"\n[5] Generating plots...")
    try:
        from visualize import plot_convergence, plot_eigenvalue_density, plot_moments
        
        if args.model in ["gaussian", "quartic", "sextic"]:
            plot_convergence(losses, args.output_dir, f"{args.model}_g{args.coupling}")
            plot_moments(sol['moments'], m_exact if 'm_exact' in dir() else None,
                        args.output_dir, f"{args.model}_g{args.coupling}")
            
            if args.model in ["gaussian", "quartic"]:
                plot_eigenvalue_density(sol['moments'], args.output_dir, 
                                       f"{args.model}_g{args.coupling}")
    except ImportError:
        print("  (matplotlib not available, skipping plots)")
    except Exception as e:
        print(f"  Plot generation failed: {e}")
    
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  Completed in {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
