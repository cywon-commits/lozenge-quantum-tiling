import json, pickle, sys, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dice_string import Tiling
from draw_hull import draw
L=int(sys.argv[1]); H=json.load(open(f'hull_fa_{L}.json'))
H=[p for p in H if p['pkl']]
n=len(H); cols=min(n,4); rows=(n+cols-1)//cols
fig,axs=plt.subplots(rows,cols,figsize=(4.6*cols,4.4*rows)); axs=axs.ravel() if n>1 else [axs]
for ax,p in zip(axs,H):
    T=Tiling(L); T.removed=pickle.load(open(p['pkl'],'rb')); draw(ax,T,f"L={L}  m = {p['m']:.4f}",win=(-1,L-0.5,-0.5,L*0.85))
for ax in axs[n:]: ax.axis('off')
plt.tight_layout(); plt.savefig(f'figures/lswt_hull_structures_L{L}.png',dpi=110)
