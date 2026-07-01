"""Leading master field for Lin-Zheng bosonic matrix QM = free/semicircle state.
Variational (Gaussian) energy in THEIR normalization; verified vs random matrices.

State: X_I free semicircular, tau(X_I X_J)=s*delta_IJ; conjugate P saturates the
oscillator uncertainty tau(P_I P_J)=(1/(4s))*delta_IJ  (=> saturates E41 [[s,1/2],[1/2,p]]).

Energy (their units, lambda=g^2_YM N=1):
  E(s) = D/(8s)  +  (1/2)M^2 D s  +  (lambda/2) D(D-1) s^2
         [kinetic]      [mass]           [commutator potential]
Commutator identity (free prob): sum_{I,J} tau(tr[X_I,X_J]^2) = -2 D(D-1) s^2.
  because for free centered semicirculars (I!=J): tau(X_I^2 X_J^2)=s^2,
  tau(X_I X_J X_I X_J)=0  =>  tau[X_I,X_J]^2 = 2*0 - 2 s^2 = -2 s^2.
Normalization: E/D >= per-flavor; <trX^2>=(1/D)<tr X_I X_I> = s.
"""
import numpy as np
from scipy.optimize import minimize_scalar

def E_gauss(s, D, M2, lam=1.0):
    return D/(8*s) + 0.5*M2*D*s + 0.5*lam*D*(D-1)*s**2

def solve(D, M2, lam=1.0):
    r = minimize_scalar(E_gauss, bracket=(0.05, 2.0), args=(D, M2, lam))
    s = r.x
    return dict(s=s, E=E_gauss(s, D, M2, lam), trXX_sum=D*s)

def rm_commutator(D, s, N=400, reps=40, seed=1):
    """Numerical check: sum_{I,J} (1/N)Tr[X_I,X_J]^2 for D indep GUE with tau(X^2)=s."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        Xs = []
        for _ in range(D):
            A = (rng.standard_normal((N,N)) + 1j*rng.standard_normal((N,N)))/np.sqrt(2)
            H = (A + A.conj().T)/np.sqrt(2)            # GUE
            H *= np.sqrt(s / ( np.trace(H@H).real/N )) # rescale to tau(X^2)=s
            Xs.append(H)
        tot = 0.0
        for I in range(D):
            for J in range(D):
                C = Xs[I]@Xs[J]-Xs[J]@Xs[I]
                tot += (np.trace(C@C).real)/N
        vals.append(tot)
    return np.mean(vals), np.std(vals)/np.sqrt(reps)

if __name__ == "__main__":
    print("="*72)
    print("LEADING MASTER FIELD (free/semicircle) — Lin-Zheng bosonic matrix QM")
    print("="*72)
    cases = [(2,1.0,"[1.172098376,1.172098408]",0.77800898),
             (2,0.0,"[0.707832,0.707868]      ",1.15420),
             (9,0.0,"[6.69946,6.69968]        ",2.29195)]
    print(f"\n{'model':>12} | {'E_Gauss':>9} {'island(true E)':>26} {'dE%':>7} | "
          f"{'<trXX>_sum':>11} (true)")
    for D,M2,isl,trXXtrue in cases:
        r = solve(D,M2)
        # true E = midpoint of island
        lo,hi = [float(x) for x in isl.replace('[','').replace(']','').split(',')]
        Etrue = 0.5*(lo+hi)
        dpct = 100*(r['E']-Etrue)/Etrue
        print(f"D={D},M^2={M2:g} | {r['E']:9.5f} {isl:>26} {dpct:+6.2f}% | "
              f"{r['trXX_sum']:11.5f} ({trXXtrue})")

    print("\n[free-probability commutator identity check vs random matrices]")
    for D in (2,9):
        s = 0.4
        exact = -2*D*(D-1)*s**2
        num, err = rm_commutator(D, s, N=300, reps=30)
        print(f"  D={D}, s={s}: sum tau[X_I,X_J]^2  exact={exact:+.4f}  "
              f"RM={num:+.4f} +/- {err:.4f}  ({'OK' if abs(num-exact)<4*err+0.02 else 'MISMATCH'})")

    print("\nInterpretation: E_Gauss is a rigorous variational UPPER bound (a genuine")
    print("master-field state). Tight at large D / massive (~0.8%); loose for D=2 massless")
    print("(6%, the flat-direction 'peninsula'). <trX^2>=s lands just BELOW the true island.")
