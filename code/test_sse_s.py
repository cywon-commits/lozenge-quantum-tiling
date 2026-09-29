import numpy as np, itertools, time
from sse_s import run
def ed(bonds,N,S,beta,h):
    d=int(2*S+1); ms=np.arange(d)-S
    import scipy.sparse as sp
    # build via kron
    Sz=np.diag(ms); Sp=np.zeros((d,d))
    for k in range(d-1): Sp[k+1,k]=np.sqrt(S*(S+1)-ms[k]*(ms[k]+1))
    Sm=Sp.T; I=np.eye(d)
    def op(o,i):
        mats=[I]*N; mats[i]=o; r=mats[0]
        for mm in mats[1:]: r=np.kron(r,mm)
        return r
    H=0
    for i,j in bonds: H=H+op(Sz,i)@op(Sz,j)+0.5*(op(Sp,i)@op(Sm,j)+op(Sm,i)@op(Sp,j))
    Mz=sum(op(Sz,i) for i in range(N)); H=H-h*Mz
    w,v=np.linalg.eigh(H); p=np.exp(-beta*(w-w[0])); Z=p.sum()
    return (p*w).sum()/Z, (p*np.einsum('ik,ij,jk->k',v,Mz,v)).sum()/Z
bb=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0],[0,3],[1,4]]); N=6   # bipartite
for S in (0.5,1.0,1.5):
    for beta,h in ((1.0,0.0),(3.0,0.3)):
        if S==1.5 and N>6: continue
        Ee,Me=ed(bb,N,S,beta,h); t=time.time(); E,M=run(bb,N,beta,S=S,h=h,ntherm=3000,nmeas=100000,seed_=1)
        print(f"S={S} beta={beta} h={h}: ED E={Ee:.4f} M={Me:.4f} | QMC E={E.mean():.4f}±{E.std()/np.sqrt(len(E)/50):.4f} M={M.mean():.4f} ({time.time()-t:.0f}s)",flush=True)
