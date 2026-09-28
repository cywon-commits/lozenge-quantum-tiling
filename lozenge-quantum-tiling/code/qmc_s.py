import numpy as np, sys, json, time, pickle
from dice_string import Tiling
from tilt import strips
from periodic import Cell, replicate
from qmc_tiling import lieb
from sse_s import run
name=sys.argv[1]; L=int(sys.argv[2]); S=float(sys.argv[3]); beta=float(sys.argv[4]); h=float(sys.argv[5]); nm=int(sys.argv[6]); sd=int(sys.argv[7])
if name=='square': T,_=strips(L,[[0,L]])
elif name=='dice': T,_=strips(L,[])
else:
    R=json.load(open('periodic_all_lswt.json')); mt={'M3':1/18,'M2':1/12}[name]
    R=[r for r in R if abs(r['m_rep']-mt)<1e-4]; R.sort(key=lambda r:r['e_lswt'])
    for r in R:
        try: T=replicate(Cell(*r['cell']),[tuple(x) for x in r['tiling']],L); break
        except Exception: continue
Sl=lieb(T); t=time.time()
E,M=run(T.bonds(),T.N,beta,S=S,h=h,ntherm=max(1000,nm//5),nmeas=nm,seed_=sd)
nb=20; Eb=E.reshape(nb,-1).mean(1)/T.N; Mb=M.reshape(nb,-1).mean(1)/T.N
mL=Sl*2*S/T.N; m=Mb.mean(); e0=Eb.mean()+h*(m+mL)/2
print(json.dumps(dict(name=name,L=L,S=S,beta=beta,h=h,imb=Sl/T.N,mL=mL,e=Eb.mean(),e_err=Eb.std()/np.sqrt(nb),m=m,e0=e0,t=time.time()-t)),flush=True)
