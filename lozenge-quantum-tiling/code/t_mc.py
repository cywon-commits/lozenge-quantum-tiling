import numpy as np, pickle, json
from dice_string import Tiling
from tiling_mc import run
T0=Tiling(24); T0.removed=pickle.load(open('per2_m12_L24.pkl','rb'))
Tr=Tiling(24); Tr.removed=set(T0.removed)
temps=[0.002,0.005,0.01,0.015,0.02,0.03,0.05,0.1]
for h in (0.19,):
    o=run(T0,Tr,h,temps,nsweep=2000,every=10)
    for r in o: print(h, r)
