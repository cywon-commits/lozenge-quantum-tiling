import numpy as np, time
from ed_dice import *
import scipy.sparse as sp
# validate on random small graph with triangles
rng=np.random.default_rng(0); N=12
bb=np.array(sorted({tuple(sorted(rng.choice(N,2,replace=False))) for _ in range(20)}))
st=np.arange(2**N); bits=((st[:,None]>>np.arange(N))&1)
H=np.zeros((2**N,2**N))
for i,j in bb:
    H[st,st]+=0.25*(2*bits[:,i]-1)*(2*bits[:,j]-1)
    m=bits[:,i]!=bits[:,j]; H[st[m]^((1<<i)|(1<<j)),st[m]]+=0.5
nup=bits.sum(1)
for k in (4,6,9):
    w=np.linalg.eigvalsh(H[np.ix_(nup==k,nup==k)])[:3]
    w2,_,_=lowest(bb,N,k,3); print(k,np.round(w,6),np.round(w2,6))
T=TorusTiling((3,3),(-3,6)); print("N",T.N,"hubs",(T.color==0).sum(),"bonds",len(T.bonds()),"monomers",len(T.monomers()))
t=time.time(); w,v,s=lowest(T.bonds(),T.N,18,3); print("Sz=4.5 sector",w,time.time()-t)
