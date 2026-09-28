import numpy as np
from sse_s import run
for eps in (0.1,0.25,1.0,3.0):
    E,M=run(np.array([[0,1]]),2,2.0,S=1.0,h=0.0,ntherm=3000,nmeas=200000,seed_=7,eps=eps)
    nb=20; Eb=E.reshape(nb,-1).mean(1); print(eps, round(Eb.mean(),4), "+-", round(Eb.std()/np.sqrt(nb),4), "(ED -1.6875)")
