import numpy as np, time
from dice_string import *
L=30
T=Tiling(L)
print("monomers dice:",len(T.check_pairing()), "bonds", len(T.bonds()), "expected", 3*L*L-L*L)
# pick a removed edge near center
e0 = sorted(T.removed, key=lambda e: np.linalg.norm(T.pos[e[0]]-T.pos[L//2*L+L//2]*0 - np.array([L/2,L/4])))[0]
ta,tb = T.create_pair(e0)
print("monomers:",T.check_pairing())
mono = ta; other=tb
path=[T.tri_cent[mono].copy()]; cur=T.tri_cent[mono].copy()
for k in range(8):
    mono,d = T.move(mono, other, np.array([1.0,0]))
    cur=cur+d; path.append(cur.copy())
    assert sorted(T.check_pairing())==sorted([mono,other])
print(np.round(np.array(path)-path[0],3))
# ground state no defects
T2=Tiling(L); mu=np.ones(T2.N)
t=time.time(); E,s,res=minimize_spins(T2.bonds(),mu); print("E dice",E, -len(T2.bonds()), time.time()-t, res.nit)
