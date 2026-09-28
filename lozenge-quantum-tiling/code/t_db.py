import numpy as np, itertools
from sse_s import full_sweep, seed, wvert
S2=2; b=np.array([[0,1]],np.int64); hb=np.zeros((1,2)); Cb=np.array([1.25])
seed(3)
spin=np.array([1,1],np.int64); ops=np.array([0,0],np.int64); dlt=np.zeros(2,np.int64)
from collections import Counter
cnt=Counter()
for t in range(300000):
    full_sweep(spin,ops,dlt,b,1.0,1,2,Cb,hb,S2,3,True)
    cnt[(spin[0],spin[1],ops[0]%2,dlt[0],ops[1]%2,dlt[1])]+=1
# exact weights
tot=sum(cnt.values())
W={}
for n0 in range(3):
  for n1 in range(3):
    for d1 in (0,1,-1):
      for d2 in (0,1,-1):
        a=(n0,n1); m=(n0+d1,n1-d1); c=(m[0]+d2,m[1]-d2)
        if c!=a: continue
        w=wvert(a[0],a[1],m[0],m[1],S2,1.25,0.,0.)*wvert(m[0],m[1],c[0],c[1],S2,1.25,0.,0.)
        if w>0: W[(n0,n1,int(d1!=0),d1,int(d2!=0),d2)]=w
Z=sum(W.values())
print("config  exact  QMC")
for k in sorted(W): print(k, round(W[k]/Z,4), round(cnt[k]/tot,4))
print("extra keys in QMC:", [k for k in cnt if k not in W][:5])
