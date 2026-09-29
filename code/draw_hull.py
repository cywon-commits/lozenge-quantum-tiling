import numpy as np, json, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from dice_string import Tiling
from tilt import strips
from draw_s6d3 import coloring
A1=np.array([1.0,0]); A2=np.array([0.5,np.sqrt(3)/2])
def draw(ax,T,title,rot=-30,win=(-1,17.5,-0.5,15.5)):
    L=T.L; c=coloring(T)
    if (c==0).sum()<(c==1).sum(): c=1-c
    R=np.array([[np.cos(np.radians(rot)),-np.sin(np.radians(rot))],[np.sin(np.radians(rot)),np.cos(np.radians(rot))]])
    Tv=[L*A1,L*A2]; shifts=[a*Tv[0]+b*Tv[1] for a in range(-2,3) for b in range(-2,3)]
    cols={(1,0):'#e8c872',(0,1):'#8fbcd4',(1,-1):'#c9a3d6',(-1,1):'#c9a3d6',(-1,0):'#e8c872',(0,-1):'#8fbcd4'}
    x0,x1,y0,y1=win
    for e in T.removed:
        t1,t2=T.edge_tris[e]; vs=list(dict.fromkeys(T.tri_sites[t1]+T.tri_sites[t2]))
        base=T.pos[vs[0]]; P=np.array([base+T.wrap(T.pos[v]-base) for v in vs]); cen=P.mean(0)
        P=P[np.argsort(np.arctan2(*(P-cen).T[::-1]))]
        d=tuple(((T.ij[e[1]]-T.ij[e[0]]+L//2)%L-L//2).tolist())
        for s in shifts:
            Q=(P+s)@R.T
            if x0<=Q.mean(0)[0]<=x1 and y0<=Q.mean(0)[1]<=y1: ax.add_patch(Polygon(Q,fc=cols.get(d,'#ddd'),ec='#555',lw=0.5))
    for i in range(T.N):
        for s in shifts:
            p=(T.pos[i]+s)@R.T
            if x0<=p[0]<=x1 and y0<=p[1]<=y1: ax.plot(*p,'o',ms=3.5,mfc='#d62728' if c[i]==0 else '#1f5fbf',mec='k',mew=0.3,zorder=3)
    ax.set_xlim(x0,x1); ax.set_ylim(y0,y1); ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title,fontsize=9)
if __name__=="__main__":
    H=json.load(open('hull_lswt.json'))
    H=[p for i,p in enumerate(H) if not (p['m']>0.166 and p['pkl'])]
    fig,axs=plt.subplots(2,(len(H)+1)//2,figsize=(4.2*((len(H)+1)//2),8.4)); axs=axs.ravel()
    for ax,p in zip(axs,H):
        if p['pkl']: T=Tiling(18); T.removed=pickle.load(open(p['pkl'],'rb'))
        else:
            iv={'dice':[],'square':[[0,18]],'s6d3':[[0,6],[9,6]],'strip w6':[[0,6]],'strip w12':[[0,12]]}[p['label']]; T,_=strips(18,iv)
        draw(ax,T,f"m = {p['m']:.4f}  ({p['label'].replace('anneal ','annealed, ')})")
    for ax in axs[len(H):]: ax.axis('off')
    plt.tight_layout(); plt.savefig('figures/lswt_hull_structures.png',dpi=120)
