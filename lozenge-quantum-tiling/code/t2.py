import numpy as np
from dice_string import *
L=30; T=Tiling(L)
e0 = sorted(T.removed, key=lambda e: np.linalg.norm(T.pos[e[0]]-np.array([L/2,L/4])))[0]
ta,tb=T.create_pair(e0); mono,other=ta,tb
p0=T.tri_cent[mono].copy(); cur=p0.copy(); out=[]
for k in range(14):
    mono,d=T.move(mono,other,np.array([np.sqrt(3)/2,0.5]),offset=(cur-p0)@np.array([-0.5,np.sqrt(3)/2])); cur=cur+d; out.append(cur-p0)
print(np.round(np.array(out),2))
