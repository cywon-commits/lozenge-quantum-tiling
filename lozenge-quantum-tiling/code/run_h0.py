import numpy as np, time, pickle, sys
from dice_string import *
L=int(sys.argv[1]); nsteps=int(sys.argv[2])
mu_h, mu_r = float(sys.argv[3]), float(sys.argv[4])
h=float(sys.argv[5]) if len(sys.argv)>5 else 0.0
tag=sys.argv[6] if len(sys.argv)>6 else "run"
T=Tiling(L)
mu=np.where(T.color==0,mu_h,mu_r)
dirv=np.array([np.sqrt(3)/2,0.5]); perp=np.array([-0.5,np.sqrt(3)/2])
# reference: defect-free dice
sgn=np.where(T.color==0,-1.0,1.0)
if mu_h>2*mu_r: sgn=-sgn
s0=np.zeros((T.N,3)); s0[:,2]=sgn
B0=T.bonds(); E0=-np.sum(mu[B0[:,0]]*mu[B0[:,1]])-h*np.sum(mu*s0[:,2])
E0,s0,_=minimize_spins(B0,mu,h=h,x0=s0+0.05*np.random.default_rng(7).normal(size=s0.shape))
e0 = sorted(T.removed, key=lambda e: np.linalg.norm(T.pos[e[0]]-np.array([L/4,L/8])))[0]
ta,tb=T.create_pair(e0); mono,other=ta,tb
p0=T.tri_cent[mono].copy(); cur=p0.copy(); pother=T.tri_cent[other].copy()
x=s0.copy()
out=[]
for k in range(nsteps):
    mono,d=T.move(mono,other,dirv,offset=(cur-p0)@perp); cur=cur+d
    if k%2==1 or k<4:
        B=T.bonds()
        Eref_exch=-np.sum(mu[B[:,0]]*mu[B[:,1]])
        t=time.time()
        E,s,res=minimize_spins(B,mu,h=h,x0=x+0.01*np.random.default_rng(k).normal(size=x.shape))
        x=s
        dist=np.linalg.norm(cur-pother)
        out.append(dict(k=k,d=dist,E=E,dE=E-E0,dE_frust=E-Eref_exch,nit=res.nit,s=s.copy(),removed=set(T.removed),mono=mono,other=other,cur=cur.copy()))
        print(f"k={k} d={dist:.2f} dE={E-E0:.4f} frust={E-Eref_exch:.4f} nit={res.nit} {time.time()-t:.1f}s",flush=True)
pickle.dump(dict(L=L,mu=mu,h=h,E0=E0,out=out,p0=p0,pother=pother),open(f"{tag}.pkl","wb"))
