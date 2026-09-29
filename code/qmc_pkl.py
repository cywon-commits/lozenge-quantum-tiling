import numpy as np, sys, json, time, pickle
from dice_string import Tiling
from qmc_tiling import lieb
from sse_h import run
pk=sys.argv[1]; L=int(sys.argv[2]); beta=float(sys.argv[3]); h=float(sys.argv[4]); nm=int(sys.argv[5]); sd=int(sys.argv[6])
T=Tiling(L); T.removed=pickle.load(open(pk,'rb')); assert not T.check_pairing(); S=lieb(T); assert S is not None
t=time.time(); sg,mz,E=run(T.bonds(),T.N,beta,max(1000,nm//5),nm,sd,h=h)
nb=20; Eb=E.reshape(nb,-1).mean(1)/T.N; Mb=mz.reshape(nb,-1).mean(1)/T.N
print(json.dumps(dict(L=L,kind='pkl',pkl=pk,beta=beta,h=h,S_lieb=S,e=Eb.mean(),e_err=Eb.std()/np.sqrt(nb),m=float(np.abs(Mb).mean()),t=time.time()-t)),flush=True)
