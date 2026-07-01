"""Lin-Zheng (2507.21007) App E level-5 UNDERSTANDING CHECK.
Reproduce the closed-form bracket E39 from the reduced positivity E41+E42, and
confirm the published islands sit inside. Also compute the g=0 Gaussian anchor.

Variables (O(D) singlets, normalized single trace, factor-N stripped as in App E):
  v = <tr X_I X_I>   (so their <tr X^2> = v/D)
  p = <tr P_I P_I>   (= -<tr Pi_I Pi_I>, Pi=-iP)
Canonical/Gauss input (E34): <tr X_I Pi_I> = D/2.
Virial (E40): E = (3/4) p + (1/4) M^2 v.

E41 (inner-product positivity on {X_I,-iP_I} and antisym):
   [[v, D/2],[D/2, p]] >= 0   ==>  v,p>0 and v*p >= D^2/4
   p >= M^2 v
E42 (ground-state positivity on {X_I,P_I}):
   [[D/2, p],[p, (D-1)v + (D/2)M^2]] >= 0  ==> (D/2)((D-1)v+(D/2)M^2) >= p^2
"""
import numpy as np

def E39_bracket(D, M2, w):
    """E39 lower/upper on E/D at fixed <trX^2>=w. Returns (lo, hi) for E (not E/D)."""
    M2 = float(M2)
    lo = D*max(3.0/(16*w) + M2*w/4.0, M2*w)
    hi = D*(1.0/8.0)*(2*M2*w + 3*np.sqrt(2*(D-1)*w + M2))
    return lo, hi

def best_lower_E(D, M2):
    """Best (largest) level-5 lower bound on E, minimizing the E39 lower expr over w>0."""
    ws = np.linspace(1e-4, 50, 2_000_00)
    lows = np.array([E39_bracket(D, M2, w)[0] for w in ws])
    # the feasible E must be >= lower(w) for the w it sits at; the *guaranteed* lower
    # bound on E is min over w of the upper envelope... no: E>=lower(w*) at the true w*.
    # The rigorous level-5 lower bound = min_w upper(w) intersect; but the simple
    # virial lower bound is the analytic branch-1 minimum. Report the branch mins.
    return lows

def gaussian_anchor(D, M2):
    """g=0: D decoupled matrix oscillators, freq w0=sqrt(M2). Ground state saturates
    p=M^2 v (equipartition) and v*p=D^2/4 (coherent/Gaussian). => v=D/(2M), p=DM/2."""
    M = np.sqrt(M2)
    v = D/(2*M); p = D*M/2.0
    E = 0.75*p + 0.25*M2*v
    return dict(v=v, p=p, trX2=v/D, E=E)

if __name__ == "__main__":
    print("="*70)
    print("Lin-Zheng App E level-5 reproduction (understanding check)")
    print("="*70)

    # 1) Analytic best lower bound on E, branch 1: min_w [3D/(16w)+D M^2 w/4] at w=sqrt(3)/(2M)
    print("\n[E39 lower-bound branch-1 analytic min]  E >= D*M*sqrt(3)/4   (M^2>0)")
    for (D, M2, true, isl) in [(2,1.0,1.172098,"[1.172098376,1.172098408]"),
                               (2,0.0,0.70783 ,"[0.707832,0.707868] (M^2=0)"),
                               (9,0.0,6.6996  ,"[6.69946,6.69968] (D=9 BFSS)")]:
        M = np.sqrt(M2)
        # numeric min of branch-1 lower over w (guard w>0)
        ws = np.linspace(1e-3, 200, 400000)
        b1 = D*(3.0/(16*ws) + M2*ws/4.0)      # branch 1 (inner-product/coherent)
        b2 = D*(M2*ws)                          # branch 2 (equipartition)
        lower_env = np.maximum(b1, b2)          # E39 lower at each w
        # E must exceed lower_env at its own w; a rigorous *number* needs w pinned.
        # Report: analytic branch-1 minimum (a valid lower bound since E>=b1 always).
        analytic = D*M*np.sqrt(3)/4 if M2>0 else 0.0
        print(f"  D={D} M^2={M2}: level-5 lower(analytic) = {analytic:.5f}"
              f"   true E ~ {true}   island {isl}")
        # sanity: is true inside [lower, upper] at some plausible w? find w where upper>=true
        hi = D*(1.0/8.0)*(2*M2*ws + 3*np.sqrt(2*(D-1)*ws + M2))
        ok = np.any((lower_env <= true) & (true <= hi))
        wtrue = ws[np.argmin(np.abs(hi-true))]
        print(f"      true value fits inside level-5 band [lower,upper]: {ok}  (upper=true near <trX2>={wtrue:.3f})")

    # 2) Gaussian anchor
    print("\n[g=0 Gaussian anchor: D decoupled matrix oscillators]")
    for (D, M2) in [(2,1.0),(9,1.0)]:
        a = gaussian_anchor(D, M2)
        print(f"  D={D} M^2={M2}:  v=<trX_IX_I>={a['v']:.4f}  p=<trP^2>={a['p']:.4f} "
              f" <trX^2>={a['trX2']:.4f}  E0(unnorm virial)={a['E']:.4f}")
    print("\n  NOTE: physical E = E0/N^2 (g^2 N)^{-1/3} is singular as g->0;")
    print("        continuation runs in lambda_eff = g^2 N / M^3 (0 -> target), not in E.")
