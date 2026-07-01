"""Operator master field for KZ two-matrix via CONTINUATION from h=0.

Solve at h=0 (decoupled quartic, physical branch known: m[A^2]=0.516 at g=1),
then ramp h:0->h_target with warm starts, minimizing the KZ planar loop residual
at each step. Tracks the physical branch, avoiding spurious loop-eq solutions the
cold-start optimizer may land on.

A,B polynomials in free semicirculars x0,x1 (positivity+traciality automatic).
"""
import sys, time, itertools
import numpy as np
from pathlib import Path as _RepoP  # noqa: E402
sys.path.insert(0, str(_RepoP(__file__).resolve().parents[2]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import minimize
from matrix_master_field.cuntz_fock import CuntzFockSpace


def sa_monomials(x0, x1, degree):
    def sym(*mats):
        P = mats[0]
        for m in mats[1:]: P = P @ m
        Pr = mats[-1]
        for m in reversed(mats[:-1]): Pr = Pr @ m
        return 0.5 * (P + Pr)
    mons = [x0]
    if degree >= 3:
        mons += [x0@x0@x0, sym(x0, x1, x1), x1@x0@x1]
    if degree >= 5:
        mons += [x0@x0@x0@x0@x0, sym(x0,x0,x0,x1,x1), sym(x0,x1,x1,x1,x1),
                 x1@x1@x0@x1@x1, sym(x0,x0,x1,x0,x1), x1@x0@x0@x0@x1]
    return mons


def make(fock, g, W, degree):
    x0 = fock.x(0).astype(np.float64); x1 = fock.x(1).astype(np.float64)
    MA = jnp.asarray(np.stack(sa_monomials(x0, x1, degree)))
    MB = jnp.asarray(np.stack(sa_monomials(x1, x0, degree)))
    nP = MA.shape[0]; vac = jnp.asarray(fock.vacuum_state())
    words = [()]
    for L in range(1, W + 1):
        words += [tuple(c) for c in itertools.product((0, 1), repeat=L)]

    def build(c, h):
        A = jnp.tensordot(c, MA, axes=(0,0)); B = jnp.tensordot(c, MB, axes=(0,0))
        A2=A@A; B2=B@B
        VpA = A + g*(A2@A) + h*(A@B2+B2@A-2.0*(B@A@B))
        VpB = B + g*(B2@B) + h*(B@A2+A2@B-2.0*(A@B@A))
        return A,B,VpA,VpB
    def tau(ops):
        v = vac
        for op in reversed(ops): v = op @ v
        return v[0]
    def loss(c, h):
        A,B,VpA,VpB = build(c,h); opm=(A,B); s=0.0
        for w in words:
            wo=[opm[k] for k in w]
            for a,Vp in ((0,VpA),(1,VpB)):
                r = tau([Vp]+wo)
                for k in range(len(w)):
                    if w[k]==a:
                        r = r - tau([opm[j] for j in w[:k]])*tau([opm[j] for j in w[k+1:]])
                s = s + r*r
        return s
    def mA2(c):
        A,_,_,_ = build(c,0.0); return (A@(A@vac))[0]
    vg = jax.jit(jax.value_and_grad(loss)); L=jax.jit(loss); M=jax.jit(mA2)
    return vg, L, M, nP


def continuation(fock, g, h_target, W, degree, n_steps=10, maxiter=2000):
    vg, L, M, nP = make(fock, g, W, degree)
    c = np.zeros(nP); c[0]=1.0
    traj=[]
    hs = np.linspace(0.0, h_target, n_steps+1)
    for h in hs:
        def fg(x):
            v,gr = vg(jnp.asarray(x), h); return float(v), np.asarray(gr,float)
        r = minimize(fg, c, jac=True, method="L-BFGS-B",
                     options={"maxiter":maxiter,"ftol":1e-15,"gtol":1e-13})
        c = r.x
        lv=float(L(jnp.asarray(c),h)); m=float(M(jnp.asarray(c)))
        traj.append((float(h),lv,m))
        print(f"  h={h:.3f}  loss={lv:.3e}  m[A^2]={m:.5f}", flush=True)
    return traj, c


if __name__ == "__main__":
    g=1.0; h_target=1.0
    print(f"CONTINUATION operator master field, KZ g={g}, h:0->{h_target}")
    print("Anchors: h=0 -> m[A^2]=0.5161 (exact quartic);  h=1 -> ~0.421 (convex L=12)\n", flush=True)
    for W,deg,FL in [(2,3,8),(3,3,10)]:
        fock=CuntzFockSpace(2,FL)
        print(f"--- W={W} deg={deg} FockL={FL} dim={fock.dim} ---", flush=True)
        t=time.time()
        traj,c = continuation(fock,g,h_target,W,deg,n_steps=10)
        print(f"    final m[A^2]={traj[-1][2]:.5f}  (target ~0.421)  {time.time()-t:.0f}s\n", flush=True)
