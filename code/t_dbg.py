import numpy as np
import sse_s
from sse_s import full_sweep, seed
S2=2; b=np.array([[0,1]],np.int64); nb=1; N=2
for beta in (0.05,2.0):
    hb=np.zeros((1,2)); Cb=np.array([1.0+0.25])
    seed(1); np.random.seed(1)
    spin=np.array([0,2],np.int64); M=40; ops=np.full(M,-1,np.int64); dlt=np.zeros(M,np.int64); n=0
    hist=np.zeros((3,3)); E=[]
    for t in range(200000):
        n=full_sweep(spin,ops,dlt,b,beta,nb,n,Cb,hb,S2,10)
        if t>1000: hist[spin[0],spin[1]]+=1; E.append(-n/beta+Cb.sum())
    print(beta, np.round(hist/hist.sum(),4).tolist(), np.mean(E))
