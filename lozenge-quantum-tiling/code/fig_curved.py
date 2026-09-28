import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dice_string import Tiling
D=pickle.load(open('curved.pkl','rb')); L=96; T=Tiling(L); hub=T.color==0; P=T.pos
cen=(D['path'].mean(0))
rel=np.array([T.wrap(p-cen) for p in P[hub]]); w=(np.abs(rel[:,0])<26)&(np.abs(rel[:,1])<22)
segs=[]
for e in D['removed']:
    if {T.color[e[0]],T.color[e[1]]}!={1,2}:
        a=T.wrap(P[e[0]]-cen); segs.append([a,a+T.wrap(P[e[1]]-P[e[0]])])
hs=[0.0,0.005,0.01,0.02,0.05,0.1,0.2,0.4]
fig,axs=plt.subplots(2,4,figsize=(15,7.6))
for ax,h in zip(axs.ravel(),hs):
    s=D['res'][(0.0,h)][hub]
    if h==0:
        far=s.mean(0); far/=np.linalg.norm(far); perp=s-np.outer(s@far,far)
        e2=np.linalg.svd(perp,full_matrices=False)[2][0]; c=np.arctan2(s@e2,s@far)
        sc=ax.scatter(rel[w,0],rel[w,1],c=c[w],cmap='twilight',vmin=-np.pi,vmax=np.pi,s=7)
        ax.set_title('h = 0 : hub-spin angle ψ')
    else:
        c=-s[:,2]; sc=ax.scatter(rel[w,0],rel[w,1],c=c[w],cmap='coolwarm',vmin=-1,vmax=1,s=7)
        ax.set_title(f'h = {h}:  ℓ_B ≈ {np.sqrt(3/h):.1f} a   (−S_z hub)')
    ax.add_collection(LineCollection(segs,colors='k',lw=1.0))
    for t in (D['mono'],D['other']):
        ax.plot(*T.wrap(T.tri_cent[t]-cen),'o',ms=8,mfc='none',mec='lime',mew=2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
fig.suptitle('U-shaped string (arc length 58 a, chord 23 a), equal moments, isotropic S²: red = M ∥ h, blue = M anti-∥ h',y=0.99)
plt.tight_layout(); plt.savefig('figures/curved_string_field.png',dpi=140)
