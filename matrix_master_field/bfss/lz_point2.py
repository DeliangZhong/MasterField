"""Vehicle A point solver, v2: precompute relation structure (coeff = c0 + g*c1),
fast residual. Continuation g:0->1. Built on validated lz_port engine."""
import numpy as np
from scipy.optimize import least_squares
import lz_port as E

def commH_split(w, mval):
    """Return commH relation as dict term-> (c0,c1): c0 const part (mass term), c1 * g (commutator)."""
    parts={'Z':[(1,0,('P',))],'z':[(1,0,('q',))],
        'P':[(mval,0,('Z',)),(0,-1,('Z','Z','z')),(0,-1,('z','Z','Z')),(0,2,('Z','z','Z'))],
        'q':[(mval,0,('z',)),(0,-1,('z','z','Z')),(0,-1,('Z','z','z')),(0,2,('z','Z','z'))]}
    e={}
    for i,c in enumerate(w):
        for c0,c1,frag in parts[c]:
            t=E.T(w[:i]+frag+w[i+1:])
            a,b=e.get(t,(0.0,0.0)); e[t]=(a+c0,b+c1)
    return e

def precompute(level, M2):
    W=E.neutral_words(level)
    Wl3=E.neutral_words(level-3) if level>=3 else []
    Wl1=E.neutral_words(level-1) if level>=1 else []
    # linear-in-g relations: g-independent ones have c1=0
    rels=[]  # each: dict term->(c0,c1)
    def wrap(e): return {t:(float(c),0.0) for t,c in e.items()}
    for w in W: rels.append(wrap(E.cycZP(w)))
    for w in Wl3: rels.append(wrap(E.gauge(w)))
    for w in W: rels.append(wrap(E.mirror(w)))
    for w in W: rels.append(wrap(E.reflect(w)))
    for w in Wl1: rels.append(commH_split(w,M2))
    rels=[r for r in rels if r]
    singles=sorted({t[0] for r in rels for t in r if len(t)==1}|{E.T(w)[0] for w in W if E.T(w)},
                   key=lambda x:(len(x),x))
    vidx={t:i for i,t in enumerate(singles)}
    # compile each relation into: const term (from tr[]), linear terms [(i,c0,c1)], quad [(i,j,c0,c1)]
    compiled=[]
    for rel in rels:
        cst0=cst1=0.0; lin=[]; quad=[]
        for term,(c0,c1) in rel.items():
            if len(term)==0: cst0+=c0; cst1+=c1
            elif len(term)==1: lin.append((vidx[term[0]],c0,c1))
            elif len(term)==2: quad.append((vidx[term[0]],vidx[term[1]],c0,c1))
            else: raise ValueError("triple trace unexpected")
        compiled.append((cst0,cst1,lin,quad))
    return compiled, vidx, len(singles)

def make_resid(compiled,nV):
    def resid(x,g):
        out=np.empty(len(compiled))
        for k,(c0,c1,lin,quad) in enumerate(compiled):
            v=c0+c1*g
            for i,a0,a1 in lin: v+=(a0+a1*g)*x[i]
            for i,j,a0,a1 in quad: v+=(a0+a1*g)*x[i]*x[j]
            out[k]=v
        return out
    return resid

def solve(level,M2,nstep=11,verbose=False):
    compiled,vidx,nV=precompute(level,M2)
    resid=make_resid(compiled,nV)
    x=np.zeros(nV)
    s0=1.0/(2*np.sqrt(M2)) if M2>0 else 0.5
    for t in [('Z','z'),('z','Z')]:
        if t in vidx: x[vidx[t]]=s0
    for g in np.linspace(0,1,nstep):
        r=least_squares(lambda y:resid(y,g),x,method='lm',max_nfev=2000,xtol=1e-13,ftol=1e-13)
        x=r.x
        if verbose: print(f"  g={g:.2f} |res|={np.linalg.norm(r.fun):.1e} <trZZb>={x[vidx[('Z','z')]]:.5f}")
    def gv(t):
        t=tuple(t)
        return x[vidx[t]] if t in vidx else (x[vidx[t[::-1]]] if t[::-1] in vidx else np.nan)
    trZZb=gv(('Z','z')); trPq=gv(('P','q'))
    trXX=2*trZZb; trPP=-2*trPq
    E_=0.75*trPP+0.25*M2*trXX
    return trZZb, E_, np.linalg.norm(resid(x,1.0)), nV

if __name__=="__main__":
    for (M2,isl,Etrue) in [(1.0,"[1.172098376,1.172098408]",1.1721),(0.0,"[0.707832,0.707868]",0.70783)]:
        print(f"\n=== D=2 M^2={M2}  island {isl} ===")
        for level in (5,6):
            trX2,Ev,res,nV=solve(level,M2,verbose=(level==6 and M2==1.0))
            dpct=100*(Ev-Etrue)/Etrue if Etrue else float('nan')
            print(f" level {level} (nVar={nV}): <trX^2>={trX2:.5f}  E={Ev:.5f}  dE={dpct:+.2f}%  |res|={res:.1e}")
