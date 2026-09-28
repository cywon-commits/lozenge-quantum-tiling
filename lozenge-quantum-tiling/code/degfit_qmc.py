import numpy as np, json, glob
from tilt import strips
from tiling_moves import degrees
from lswt import lswt_energy
W=json.load(open('walls.json'))
# add pure phases (QMC e0): square (L18 avg), dice
pts=[(18,[[0,18]],-0.6692),(18,[],-0.6379),(24,[],-0.6379)]+[(o['L'],[list(x) for x in o['iv']],o['e0']) for o in W]
X=[];Yq=[];Yl=[]
for L,iv,e in pts:
    T,ok=strips(L,iv); z=degrees(T); El,S=lswt_energy(T.N,T.bonds())
    n=[np.mean(z==k) for k in (3,4,5,6)]
    X.append([1,n[1],n[2],S/T.N]); Yq.append(e); Yl.append(El/T.N)
X=np.array(X); Yq=np.array(Yq); Yl=np.array(Yl)
for name,Y in (("QMC",Yq),("LSWT",Yl)):
    c=np.linalg.lstsq(X,Y,rcond=None)[0]; r=np.sqrt(np.mean((X@c-Y)**2))
    print(f"{name}: const(dice-like 3/6 sites)={c[0]:.4f}  eps'_4={c[1]:+.4f}  eps'_5={c[2]:+.4f}  c_m={c[3]:+.4f}  rms={r:.5f}  spread={Y.std():.4f}")
