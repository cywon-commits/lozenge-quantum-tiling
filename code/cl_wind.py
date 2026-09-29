import numpy as np
from wind import wound, bipartite
from dice_string import minimize_spins
for w,se in (((0,0),0),((1,0),0),((1,1),0),((2,0),1),((0,1),2)):
    T,f=wound(*w,start_edge_index=se); B=T.bonds(); mu=np.full(T.N,0.5)
    best=None
    for sd in range(60):
        E,s,_=minimize_spins(B,mu,seed=sd,tol=1e-14)
        if best is None or E<best[0]-1e-9: best=(E,s)
    E,s=best; M=np.linalg.norm((0.5*s).sum(0))
    # coplanarity / collinearity measure
    ev=np.linalg.eigvalsh(s.T@s/T.N)
    print(w,"E_cl=%.4f  (all-satisfied %.2f)  |M|_cl=%.3f  spin-tensor eig"%(E,-len(B)/4,M),np.round(ev,3))
