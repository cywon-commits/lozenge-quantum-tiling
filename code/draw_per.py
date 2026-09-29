import pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dice_string import Tiling
from draw_hull import draw
from tilt import strips
fig,axs=plt.subplots(1,4,figsize=(20,5.6))
T,_=strips(24,[[0,24]]); draw(axs[0],T,'square (4,4,4):  M = 0',win=(-1,16,-0.5,13))
T=Tiling(18); T.removed=pickle.load(open('per_m18_L18.pkl','rb')); draw(axs[1],T,'periodic, cell 6×3 (18 sites):  M = M_dice/3\nz: 3:4:5 = 4:10:4',win=(-1,16,-0.5,13))
T=Tiling(24); T.removed=pickle.load(open('per_m12_L24.pkl','rb')); draw(axs[2],T,'periodic, cell 6×4 (24 sites):  M = M_dice/2\nz: 3:4:5 = 1:1:1 (no z=6 hubs)',win=(-1,16,-0.5,13))
T,_=strips(24,[]); draw(axs[3],T,'dice (6,3,3):  M = M_dice',win=(-1,16,-0.5,13))
plt.tight_layout(); plt.savefig('figures/periodic_ground_structures.png',dpi=120)
