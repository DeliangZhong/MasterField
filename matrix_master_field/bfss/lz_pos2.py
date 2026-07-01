"""Positivity-augmented master field v2: N g-dependent, anchor at TRUE Gaussian,
SQP-linearized exact factorization + continuation. Minimize E s.t. loop eqs + M>=0 + N(g)>=0."""
import itertools, numpy as np, cvxpy as cp
import lz_port as E
from lz_gauss_moments import twopt_table, moment

REF2={'Z':('z',1),'z':('Z',1),'P':('q',-1),'q':('P',-1)}
def conj_op(w):
    s=1; out=[]
    for c in reversed(w):
        nc,sg=REF2[c]; s*=sg; out.append(nc)
    return tuple(out), s
def charged_words(maxlev):
    return [w for n in range(0,maxlev+1) for w in itertools.product('ZzPq',repeat=n) if E.lvl(w)<=maxlev]
def charge(w): return sum(E.CH[c] for c in w)
def commH_terms(w,mval,g):
    parts={'Z':[(1.0,('P',))],'z':[(1.0,('q',))],
        'P':[(mval,('Z',)),(-g,('Z','Z','z')),(-g,('z','Z','Z')),(2*g,('Z','z','Z'))],
        'q':[(mval,('z',)),(-g,('z','z','Z')),(-g,('Z','z','z')),(2*g,('z','Z','z'))]}
    e={}
    for i,c in enumerate(w):
        for cf,frag in parts[c]:
            t=w[:i]+frag+w[i+1:]; e[t]=e.get(t,0.0)+cf
    return e

def build(level,M2):
    Wneut=E.neutral_words(level); vidx={w:i for i,w in enumerate(Wneut)}; nV=len(Wneut)
    def vi(w): return vidx.get(w)
    def compile_rel(rel):
        cst=0.0; lin={}; quad=[]
        for term,c in rel.items():
            c=float(c)
            if len(term)==0: cst+=c
            elif len(term)==1:
                ii=vi(term[0])
                if ii is None: return None
                lin[ii]=lin.get(ii,0.0)+c
            elif len(term)==2:
                ii,jj=vi(term[0]),vi(term[1])
                if ii is None or jj is None: return None
                quad.append((ii,jj,c))
            else: return None
        return (cst,list(lin.items()),quad)
    relZ=[]
    for w in Wneut:
        for r in (E.cycZP(w),E.mirror(w),E.reflect(w)):
            cr=compile_rel(r)
            if cr: relZ.append(cr)
    for w in (E.neutral_words(level-3) if level>=3 else []):
        cr=compile_rel(E.gauge(w))
        if cr: relZ.append(cr)
    Wl1=E.neutral_words(level-1)
    def commH_rels(g):
        out=[]
        for w in Wl1:
            e={}
            for t,c in commH_terms(w,M2,g).items(): e[E.T(t)]=e.get(E.T(t),0.0)+c
            cr=compile_rel({k:v for k,v in e.items()})
            if cr: out.append(cr)
        return out
    # M blocks (g-independent): entries (idx,sign)
    opsM=charged_words(level//2)
    from collections import defaultdict
    byM=defaultdict(list)
    for o in opsM: byM[charge(o)].append(o)
    Mblocks=[]
    for c,ops in byM.items():
        n=len(ops); ent=[[None]*n for _ in range(n)]; ok=True
        for i,oi in enumerate(ops):
            ci,si=conj_op(oi)
            for j,oj in enumerate(ops):
                idx=vi(ci+oj)
                if idx is None: ok=False;break
                ent[i][j]=(idx,si)
            if not ok:break
        if ok and n>1: Mblocks.append(ent)   # skip 1x1 (trivial, and helps conditioning)
    # N block STRUCTURE (g-dependent): store (conj_i sign, ci, oj)
    opsN=charged_words((level-1)//2); byN=defaultdict(list)
    for o in opsN: byN[charge(o)].append(o)
    Npairs=[]
    for c,ops in byN.items():
        if len(ops)<=1: continue
        cinfo=[conj_op(o) for o in ops]
        Npairs.append((ops,cinfo))
    def Nblocks(g):
        blocks=[]
        for ops,cinfo in Npairs:
            n=len(ops); ent=[[None]*n for _ in range(n)]; ok=True
            for i,(ci,si) in enumerate(cinfo):
                for j,oj in enumerate(ops):
                    terms=[]
                    for w2,cf in commH_terms(oj,M2,g).items():
                        idx=vi(ci+tuple(w2))
                        if idx is None: ok=False;break
                        terms.append((idx,si*cf))
                    if not ok:break
                    ent[i][j]=terms
                if not ok:break
            if ok: blocks.append(ent)
        return blocks
    return dict(Wneut=Wneut,vidx=vidx,nV=nV,relZ=relZ,commH_rels=commH_rels,
               Mblocks=Mblocks,Nblocks=Nblocks)

def solve(level,M2,nstep=11,sqp=5,tr=0.25,verbose=True):
    B=build(level,M2); nV=B['nV']; vidx=B['vidx']
    def vio(t): t=tuple(t); return vidx.get(t,vidx.get(t[::-1]))
    iZz=vio(('Z','z')); iPq=vio(('P','q'))
    tp=twopt_table(M2)
    x0=np.array([moment(w,tp).real for w in B['Wneut']])   # TRUE Gaussian anchor
    def objE(xv): return -1.5*xv[iPq]+0.5*M2*xv[iZz]
    if verbose: print(f"  anchor(g=0): <trX^2>={x0[iZz]:.5f} E={objE(x0):.5f}")
    for g in np.linspace(0,1,nstep):
        crels=B['commH_rels'](g); Nb=B['Nblocks'](g)
        for it in range(sqp):
            x=cp.Variable(nV); cons=[]
            for (cst,lin,quad) in (B['relZ']+crels):
                ex=cst
                for i,c in lin: ex=ex+c*x[i]
                for i,j,c in quad: ex=ex+c*(x0[i]*x[j]+x[i]*x0[j]-x0[i]*x0[j])
                cons.append(ex==0)
            for ent in B['Mblocks']:
                n=len(ent); Mm=cp.bmat([[ent[i][j][1]*x[ent[i][j][0]] for j in range(n)] for i in range(n)])
                cons.append(0.5*(Mm+Mm.T)>>0)
            for ent in Nb:
                n=len(ent); Nm=cp.bmat([[sum(c*x[idx] for idx,c in ent[i][j]) if ent[i][j] else 0 for j in range(n)] for i in range(n)])
                cons.append(0.5*(Nm+Nm.T)>>0)
            cons.append(cp.norm(x-x0,'inf')<=tr)
            pr=cp.Problem(cp.Minimize(-1.5*x[iPq]+0.5*M2*x[iZz]),cons)
            try: pr.solve(solver=cp.SCS,verbose=False,eps=1e-7,max_iters=20000)
            except Exception as ex:
                if verbose:print(f"   g={g:.2f} fail {ex}")
                break
            if x.value is None:
                if verbose:print(f"   g={g:.2f} it{it} {pr.status}")
                break
            x0=np.array(x.value).flatten()
        if verbose and abs(g*5-round(g*5))<1e-9:
            print(f"  g={g:.2f} <trX^2>={x0[iZz]:.5f} E={objE(x0):.5f}")
    return x0[iZz],objE(x0)

if __name__=="__main__":
    print("=== D=2 M^2=1 + POSITIVITY (true <trX^2>=0.389, E in [1.172098376,1.172098408]) ===")
    for level in (5,6):
        trX2,En=solve(level,1.0,nstep=11,sqp=4)
        print(f" => level {level}: <trX^2>={trX2:.5f} (Δ{100*(trX2-0.389)/0.389:+.1f}%)  E={En:.5f} (Δ{100*(En-1.1721)/1.1721:+.2f}%)\n")
