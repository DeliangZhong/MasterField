"""Exact free-Gaussian (g=0) moments for the complex D=2 basis Z,z,P,q via planar Wick.
2-point matrix G in (X1,X2,Pi1,Pi2) basis; letters as complex vectors; planar (non-crossing)
Wick sum. Gives the TRUE Gaussian moment vector (correct continuation anchor + M/N PSD check)."""
import numpy as np, itertools
# code letters as vectors over (X1,X2,Pi1,Pi2):
def letters(M):
    r2=np.sqrt(2)
    return {'Z':np.array([1,1j,0,0])/r2,'z':np.array([1,-1j,0,0])/r2,
            'P':np.array([0,0,1,1j])/r2,'q':np.array([0,0,1,-1j])/r2}
def Gmat(M):
    return np.array([[1/(2*M),0,0.5,0],[0,1/(2*M),0,0.5],
                     [-0.5,0,-M/2,0],[0,-0.5,0,-M/2]],dtype=complex)
def twopt_table(M2):
    M=np.sqrt(M2) if M2>0 else 1e-9
    L=letters(M); G=Gmat(M); keys='ZzPq'
    return {(a,b): complex(L[a]@G@L[b]) for a in keys for b in keys}

def moment(word, tp):
    """planar (non-crossing) Wick sum of 2-points over the cyclic... NO: single trace, use
    non-crossing pairings of the LINEAR word (large-N planar = non-crossing on the circle)."""
    n=len(word)
    if n==0: return 1.0+0j
    if n%2==1: return 0.0+0j
    # non-crossing perfect matchings of positions 0..n-1 on a line/circle. For a single trace
    # (cyclic), planar = non-crossing on the circle. Use non-crossing on the circle: standard.
    # recursive: pair position 0 with position k (k odd distance so both sides even), non-crossing.
    def nc(seq):
        if not seq: return 1.0+0j
        total=0.0+0j; i0=seq[0]
        for m in range(1,len(seq)):
            # 0 pairs with m; inside seq[1..m-1], outside seq[m+1..]
            inside=seq[1:m]; outside=seq[m+1:]
            if len(inside)%2: continue
            total+=tp[(word[i0],word[seq[m]])]*nc(inside)*nc(outside)
        return total
    return nc(list(range(n)))

if __name__=="__main__":
    for M2 in (1.0,):
        tp=twopt_table(M2)
        print(f"M^2={M2} 2-points:")
        for k in ['Zz','zZ','Zq','qZ','ZP','PP','qq','Pq','qP']:
            print(f"  <tr {k[0]}{k[1]}>={tp[(k[0],k[1])].real:+.4f}{'' if abs(tp[(k[0],k[1])].imag)<1e-9 else ' +imag!'}")
        print("checks: <Zz>=0.5?", abs(tp[('Z','z')]-0.5)<1e-9, " <Zq>-<qZ>=1?", abs(tp[('Z','q')]-tp[('q','Z')]-1)<1e-9)
        # some 4-pt moments
        for w in [('Z','z','Z','z'),('Z','z','z','Z'),('Z','Z','z','z'),('P','q','Z','z')]:
            print(f"  <tr {''.join(w)}> = {moment(w,tp).real:+.5f}")
