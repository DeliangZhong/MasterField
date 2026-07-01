"""v6 clean: correct auto-calibrated energy, levels 5-7, unbuffered. M^2=1.
Energy: <trP_IP_I> calibrated so E0=D*M/2 at g=0; E=(3/4)<trP_IP_I>+(1/4)M^2*<trX_IX_I>."""
import sys,numpy as np
from scipy.optimize import least_squares
import lz_port as E
from lz_point2 import precompute, make_resid
def P(*a): print(*a,flush=True)

def run(level,M2,nstep=31):
    compiled,vidx,nV=precompute(level,M2); resid=make_resid(compiled,nV)
    s0=1.0/(2*np.sqrt(M2))
    def gv(x,t):
        t=tuple(t); return x[vidx[t]] if t in vidx else (x[vidx[t[::-1]]] if t[::-1] in vidx else np.nan)
    x=np.zeros(nV)
    for t in [('Z','z'),('z','Z')]:
        if t in vidx: x[vidx[t]]=s0
    pin=[vidx[t] for t in [('Z','z'),('z','Z')] if t in vidx]
    x=least_squares(lambda y:np.array(list(resid(y,0.0))+[80.0*(y[i]-s0) for i in pin]),x,
                    method='lm',max_nfev=8000,xtol=1e-15,ftol=1e-15).x
    trPq0=gv(x,('P','q'))
    # calibrate: <trP_IP_I> := (trPq/trPq0)*P0 where P0 makes E0=D*M/2=sqrt(M2) (D=2)
    #   E0 = (3/4)P0 + (1/4)M2*2*s0 = sqrt(M2) => P0 = (4/3)(sqrt(M2)-0.5*M2*2*s0)
    P0=(4.0/3.0)*(np.sqrt(M2)-0.5*M2*2*s0)
    def trPP(x): return (gv(x,('P','q'))/trPq0)*P0
    def energy(x): return 0.75*trPP(x)+0.25*M2*2*gv(x,('Z','z'))
    P(f"  [g=0] <trZZb>={gv(x,('Z','z')):.5f} <trPq>={trPq0:.5f} E0={energy(x):.5f} (target {np.sqrt(M2):.3f})")
    for g in np.linspace(0,1,nstep)[1:]:
        xp=x.copy()
        for mu in (2e-3,0.0):
            f=(lambda y:np.concatenate([resid(y,g),np.sqrt(mu)*(y-xp)])) if mu>0 else (lambda y:resid(y,g))
            x=least_squares(f,x,method='lm',max_nfev=2500,xtol=1e-14,ftol=1e-14).x
    return gv(x,('Z','z')),energy(x),np.linalg.norm(resid(x,1.0)),nV

if __name__=="__main__":
    P("=== D=2 M^2=1  target: <trX^2>=0.38900, E in [1.172098376,1.172098408] ===")
    P("(Gaussian MF baseline: <trX^2>=0.3774, E=1.18226)")
    for level in (5,6,7):
        trX2,En,res,nV=run(level,1.0)
        P(f" => level {level} (nVar={nV}): <trX^2>={trX2:.5f} (Δ{100*(trX2-0.389)/0.389:+.1f}%)  "
          f"E={En:.5f} (Δ{100*(En-1.1721)/1.1721:+.2f}%)  |res|={res:.1e}\n")
