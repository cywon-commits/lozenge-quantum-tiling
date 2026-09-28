import numpy as np, scipy.sparse as sp
from sse_h import run
def ed(bonds,N,beta,h):
    dim=2**N; st=np.arange(dim); bits=((st[:,None]>>np.arange(N))&1)*2-1
    H=np.zeros((dim,dim))
    for i,j in bonds:
        H[st,st]+=0.25*bits[:,i]*bits[:,j]; m=bits[:,i]!=bits[:,j]; H[st[m]^((1<<i)|(1<<j)),st[m]]+=0.5
    H[st,st]-=h*0.5*bits.sum(1)
    w,v=np.linalg.eigh(H); mz=0.5*bits.sum(1); p=np.exp(-beta*(w-w[0])); Z=p.sum()
    return (p*w).sum()/Z,(p*((v**2).T@mz)).sum()/Z
bb=np.array([[0,1],[1,2],[2,3],[3,0],[0,4],[4,5],[5,6],[6,3],[1,7],[7,8],[8,2]]); N=9   # bipartite? check
for h in (0.0,0.5,1.2):
    Ee,Me=ed(bb,N,4.0,h); sg,m2,E=run(bb,N,4.0,2000,100000,1,h=h)
    print(h,"ED",round(Ee,4),round(Me,4),"QMC",round((E*sg).mean()/sg.mean(),4),"sign",sg.mean())
