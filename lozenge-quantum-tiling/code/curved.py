import numpy as np, pickle, sys
from dice_string import *
L=96; T=Tiling(L); mu=np.ones(T.N)
e0 = sorted(T.removed, key=lambda e: np.linalg.norm(T.pos[e[0]]-np.array([L/2,L/4])))[0]
ta,tb=T.create_pair(e0); mono,other=ta,tb
path=[T.tri_cent[mono].copy()]; cur=path[0].copy()
legs=[(30,22),(90,14),(150,22)]   # U-shape
for ang,n in legs:
    dv=np.array([np.cos(np.radians(ang)),np.sin(np.radians(ang))]); pv=np.array([-dv[1],dv[0]]); st=cur.copy()
    for k in range(n):
        mono,d=T.move(mono,other,dv,offset=(cur-st)@pv); cur=cur+d; path.append(cur.copy())
pother=T.tri_cent[other]
s_len=np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1)); dd=np.linalg.norm(T.wrap(cur-pother))
print("arc length",s_len,"chord",dd)
B=T.bonds(); sgn=np.where(T.color==0,-1.0,1.0); x0=np.zeros((T.N,3)); x0[:,2]=sgn
res={}
for K in (0.0,3.0):
  x=x0.copy()
  for h in (0.0,0.005,0.01,0.02,0.05,0.1,0.2,0.4):
    if K==0 and h==0: xs=x0+0.05*np.random.default_rng(0).normal(size=x0.shape)
    else: xs=x+0.02*np.random.default_rng(1).normal(size=x0.shape)
    E,s,r=minimize_spins(B,mu,h=h,K=K,x0=xs)
    # also try starting from collinear ground state
    E2,s2,r2=minimize_spins(B,mu,h=h,K=K,x0=x0+0.02*np.random.default_rng(2).normal(size=x0.shape))
    if E2<E: E,s=E2,s2
    x=s; res[(K,h)]=s; print(K,h,E+len(B),flush=True)
pickle.dump(dict(res=res,removed=set(T.removed),mono=mono,other=other,path=np.array(path),cur=cur,pother=pother),open('curved.pkl','wb'))
