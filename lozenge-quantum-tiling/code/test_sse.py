import numpy as np, time
from sse import run
from dice_string import Tiling
import scipy.sparse as sp, scipy.sparse.linalg as sla
def ed(bonds,N,beta):
    dim=2**N; st=np.arange(dim)
    bits=((st[:,None]>>np.arange(N))&1)*2-1
    diag=np.zeros(dim); rows=[];cols=[];vals=[]
    for i,j in bonds:
        diag+=0.25*bits[:,i]*bits[:,j]
        m=bits[:,i]!=bits[:,j]; s=st[m]; rows+=list(s^((1<<i)|(1<<j))); cols+=list(s); vals+=[0.5]*m.sum()
    H=sp.coo_matrix((vals,(rows,cols)),shape=(dim,dim)).toarray()+np.diag(diag)
    w,v=np.linalg.eigh(H); mz2=(0.25*bits.sum(1)**2)
    p=np.exp(-beta*(w-w[0])); Z=p.sum()
    return (p*w).sum()/Z, (p*((v**2).T@mz2)).sum()/Z
if __name__=="__main__":
  pass
T=Tiling(6)
e0=sorted(T.removed)[0]; T.create_pair(e0)
B=T.bonds(); c=T.pos[e0[0]]
sel=np.where(np.array([np.linalg.norm(T.wrap(p-c)) for p in T.pos])<1.9)[0]
idx={s:k for k,s in enumerate(sel)}
bb=np.array([[idx[a],idx[b]] for a,b in B if a in idx and b in idx]); N=len(sel)
print("N",N,"bonds",len(bb))
for beta in (1.0,4.0):
    Ee,Me=ed(bb,N,beta)
    t=time.time(); sg,m2,nn=run(bb,N,beta,ntherm=5000,nmeas=200000,seed=3)
    Eq=(-(nn*sg).mean()/beta)/sg.mean()+len(bb)/4; Mq=(m2*sg).mean()/sg.mean()
    print(f"beta={beta} ED E={Ee:.4f} M2={Me:.4f} | QMC E={Eq:.4f} M2={Mq:.4f} <s>={sg.mean():.3f} ({time.time()-t:.1f}s)")
