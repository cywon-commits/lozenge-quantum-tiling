import numpy as np, time
from ed_dice import *
rng=np.random.default_rng(0); N=12
bb=np.array(sorted({tuple(sorted(rng.choice(N,2,replace=False))) for _ in range(20)}))
print(lowest(bb,N,6,3)[0])
T=TorusTiling((3,3),(-3,6))
t=time.time(); w,v,s=lowest(T.bonds(),T.N,18,3); print("Sz=4.5",w,time.time()-t)
