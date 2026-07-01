"""Faithful Python port of Lin-Zheng's O2MassCode.nb (D=2, complex Z,Zb,P,Pb).
Letters: 'Z','z'(=Zb),'P','q'(=Pb). level: Z,z=1; P,q=2. charge: Z,P=+1; z,q=-1.
conj: Z->q, z->P, P->z, q->Z.  ref(O2): Z<->z, P<->q.
CCR (real corr, P=Pi=-iP_phys):  moving first letter l to back:
   tr[l,*tail] - tr[*tail,l] = (-1)^{#P,q in l} * sum_{i in tail: tail[i]=conj(l) AND
        Accumulate(charge tail)[i]=charge(conj l)}  tr[tail[:i]] * tr[tail[i+1:]]
EOM commHitem: [H,Z]=P; [H,z]=q; [H,P]=m Z-(ZZz+zZZ-2ZZz-> ZZz+zZZ-2ZzZ); [H,q]=m z-(zzZ+Zzz-2zZz).
"""
import itertools, numpy as np
from fractions import Fraction as Fr

CH={'Z':1,'P':1,'z':-1,'q':-1}
CONJ={'Z':'q','z':'P','P':'z','q':'Z'}
REF={'Z':'z','z':'Z','P':'q','q':'P'}
LV={'Z':1,'z':1,'P':2,'q':2}
def lvl(w): return sum(LV[c] for c in w)
def npq(w): return sum(1 for c in w if c in 'Pq')

# term = tuple(sorted(traces)); trace=tuple of letters. expr = dict term->coeff(Fraction)
def T(*traces):  # make a monomial key from trace tuples (drop empty traces -> factor 1)
    ts=tuple(sorted(t for t in traces if len(t)>0))
    return ts
def add(e,term,c):
    e[term]=e.get(term,Fr(0))+Fr(c)
    if e[term]==0: del e[term]

def cycZP(w):
    """relation expr = tr[w] - tr[rotate] - tsum == 0, w a single trace tuple."""
    e={}
    if len(w)==0: return e
    l=w[0]; tail=w[1:]
    add(e, T(w), 1)
    add(e, T(tail+(l,)), -1)   # RotateLeft: l to back
    # double-trace splits
    acc=np.cumsum([CH[c] for c in tail]) if tail else []
    tgt=CH[CONJ[l]]; cj=CONJ[l]
    sgn=(-1)**(1 if l in 'Pq' else 0)
    for i,c in enumerate(tail):
        if c==cj and acc[i]==tgt:
            before=tail[:i]; after=tail[i+1:]
            add(e, T(before,after), -sgn)
    return e

def gauge(w):
    e={}
    for a,b,s in [('Z','q',1),('q','Z',-1),('z','P',1),('P','z',-1)]:
        add(e, T(w+(a,b)), s)
    add(e, T(w), -2)
    return e

def mirror(w):
    e={}; add(e,T(w[::-1]), (-1)**npq(w)); add(e,T(w),-1); return e

def reflect(w):
    e={}; add(e, T(tuple(REF[c] for c in w)),1); add(e,T(w),-1); return e

COMMIT={'Z':[(1,('P',))],
        'z':[(1,('q',))],
        'P':[('m',('Z',)),(-1,('Z','Z','z')),(-1,('z','Z','Z')),(2,('Z','z','Z'))],
        'q':[('m',('z',)),(-1,('z','z','Z')),(-1,('Z','z','z')),(2,('z','Z','z'))]}
def commH(w, mval):
    e={}
    for i,c in enumerate(w):
        for coef,frag in COMMIT[c]:
            cc = mval if coef=='m' else coef
            new=w[:i]+frag+w[i+1:]
            add(e, T(new), cc)
    return e

def neutral_words(level):
    out=[]
    for n in range(1,level+1):
        for w in itertools.product('ZzPq',repeat=n):
            if lvl(w)<=level and sum(CH[c] for c in w)==0:
                out.append(w)
    return out

def build(level, mval):
    W=neutral_words(level)
    Wl3=neutral_words(level-3) if level>=3 else []
    Wl1=neutral_words(level-1) if level>=1 else []
    rels=[]
    rels+=[cycZP(w) for w in W]
    rels+=[gauge(w) for w in Wl3]
    rels+=[mirror(w) for w in W]
    rels+=[reflect(w) for w in W]
    rels+=[commH(w,mval) for w in Wl1]
    rels=[r for r in rels if r]
    return W, rels

if __name__=="__main__":
    # ---- test E4 analog: tr[Z,q]-tr[q,Z] = 1 ----
    print("cycZP(tr[Z,q]) =", {k:str(v) for k,v in cycZP(('Z','q')).items()})
    print("  (expect tr[Z,q]-tr[q,Z]-tr[]*tr[]=0  i.e. tr[Zq]-tr[qZ]=1)\n")
    # ---- free-variable count vs Table I (D=2: L4->3, L6->8, L8->22) ----
    import numpy as np
    m=float(np.euler_gamma)   # generic mass^2 to avoid degeneracies (as they do)
    for level in (4,5,6,7,8):
        W,rels=build(level,m)
        terms=sorted({t for r in rels for t in r} | {T(w) for w in W}, key=lambda x:(len(x),x))
        # separate single-trace terms (the physical vars); double-traces are products
        idx={t:i for i,t in enumerate(terms)}
        A=np.zeros((len(rels),len(terms)))
        for r,rel in enumerate(rels):
            for t,c in rel.items(): A[r,idx[t]]=float(c)
        s=np.linalg.svd(A,compute_uv=False)
        tol=1e-9*max(A.shape)*s[0]
        rank=int((s>tol).sum())
        single=[t for t in terms if len(t)==1]
        free=len(terms)-rank
        print(f"level {level}: #terms={len(terms)} (single={len(single)}) #rels={len(rels)} "
              f"rank={rank} free={free}")
    print("\nTable I D=2: level4->3, level6->8, level8->22 free variables.")
