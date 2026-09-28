import numpy as np, time
from sse import run
import scipy.sparse as sp
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
def cmp(bb,N,beta,nm=100000):
    Ee,Me=ed(bb,N,beta); t=time.time(); sg,m2,E=run(bb,N,beta,3000,nm,1)
    s=sg.mean()
    print(f"N={N} beta={beta}: ED E={Ee:.4f} M2={Me:.4f} | QMC E={(E*sg).mean()/s:.4f} M2={(m2*sg).mean()/s:.4f} <s>={s:.3f} ({time.time()-t:.1f}s)",flush=True)
if __name__=="__main__":
    for bb,N in ((np.array([[0,1],[1,2],[2,3],[3,0]]),4),(np.array([[0,1],[1,2],[2,0]]),3),(np.array([[0,1],[1,2],[2,3],[3,4],[4,0]]),5)):
        for beta in (1.0,4.0): cmp(bb,N,beta)
    from dice_string import Tiling
    T=Tiling(6); e0=sorted(T.removed)[0]; T.create_pair(e0); B=T.bonds(); c=T.pos[e0[0]]
    sel=np.where(np.array([np.linalg.norm(T.wrap(p-c)) for p in T.pos])<1.9)[0]; idx={s:k for k,s in enumerate(sel)}
    bb=np.array([[idx[a],idx[b]] for a,b in B if a in idx and b in idx])
    for beta in (1.0,4.0): cmp(bb,len(sel),beta,200000)
