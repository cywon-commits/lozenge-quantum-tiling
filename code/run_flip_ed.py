import numpy as np, time, json, sys
from ed_dice import TorusTiling, lowest
from flips import flip_B
res=[]
for k in [int(x) for x in sys.argv[1].split(',')]:
    T=TorusTiling((3,3),(-3,6)); Bs=np.where(T.color==1)[0]; flip_B(T,Bs[:k])
    S=abs(4.5-k) if k<=4 else abs(k-4.5); t=time.time()
    w,_,_=lowest(T.bonds(),27,int(round(13.5+S)),k=1)
    res.append(dict(k=k,S=S,E=float(w[0]))); print(json.dumps(res[-1]),f"{time.time()-t:.0f}s",flush=True)
