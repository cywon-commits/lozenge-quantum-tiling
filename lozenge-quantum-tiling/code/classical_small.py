import numpy as np, json
from run_ed import config
from dice_string import minimize_spins
res={}
for nh in (-1,0,1,3):
    T,info=config(nh); B=T.bonds(); N=T.N; mu=np.full(N,0.5)
    best=None
    for sd in range(40):
        E,s,_=minimize_spins(B,mu,seed=sd,tol=1e-14)
        if best is None or E<best[0]-1e-9: best=(E,s)
    E,s=best; M=(0.5*s).sum(0); Mn=np.linalg.norm(M)
    loc=(0.5*s)@(M/Mn)
    res[nh]=dict(d=None if info is None else info[2],E=E,M=Mn,loc=loc.tolist())
    print(nh, None if info is None else round(info[2],2),"E_cl=%.4f |M|_cl=%.3f"%(E,Mn))
json.dump(res,open('classical_small.json','w'))
