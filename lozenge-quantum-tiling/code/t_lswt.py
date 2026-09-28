import numpy as np, json
from lswt import lswt_energy
from tilt import strips
from dice_string import Tiling
# square lattice check
for L in (12,18,24):
    T,ok=strips(L,[[0,L]]); E,S=lswt_energy(T.N,T.bonds()); print("square L",L,E/T.N)
T,ok=strips(18,[]); E,S=lswt_energy(T.N,T.bonds()); print("dice L18",E/T.N,"S",S)
# compare with QMC e0 for all strip configs
Q=json.load(open('walls.json'))
print("config  L  e0_QMC  e_LSWT")
for o in Q:
    T,ok=strips(o['L'],[list(x) for x in o['iv']]); E,S=lswt_energy(T.N,T.bonds())
    print(o['L'],o['iv'],round(o['e0'],5),round(E/T.N,5))
