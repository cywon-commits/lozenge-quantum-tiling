import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from collections import deque
from tilt import strips
A1=np.array([1.0,0]); A2=np.array([0.5,np.sqrt(3)/2])
def coloring(T):
    B=T.bonds(); adj=[[] for _ in range(T.N)]
    for i,j in B: adj[i].append(j); adj[j].append(i)
    c=-np.ones(T.N,int); c[0]=0; q=deque([0])
    while q:
        u=q.popleft()
        for v in adj[u]:
            if c[v]<0: c[v]=1-c[u]; q.append(v)
    return c
def draw(ax,L,iv,title,rot=-30,win=(-1,17.5,-0.5,11.5),annot=False):
    T,ok=strips(L,iv); assert ok
    c=coloring(T)
    if (c==0).sum()<(c==1).sum(): c=1-c          # 0 = majority
    R=np.array([[np.cos(np.radians(rot)),-np.sin(np.radians(rot))],[np.sin(np.radians(rot)),np.cos(np.radians(rot))]])
    Tv=[L*A1,L*A2]
    cols={(1,0):'#e8c872',(0,1):'#8fbcd4',(1,-1):'#c9a3d6',(-1,1):'#c9a3d6',(-1,0):'#e8c872',(0,-1):'#8fbcd4'}
    shifts=[a*Tv[0]+b*Tv[1] for a in range(-2,3) for b in range(-2,3)]
    x0,x1,y0,y1=win
    dy=[]
    def inwin(p): return x0<=p[0]<=x1 and y0<=p[1]<=y1
    # lozenges
    for e in T.removed:
        t1,t2=T.edge_tris[e]
        vs=list(dict.fromkeys(T.tri_sites[t1]+T.tri_sites[t2]))
        base=T.pos[vs[0]]; P=np.array([base+T.wrap(T.pos[v]-base) for v in vs])
        # order polygon around centroid
        cen=P.mean(0); ang=np.arctan2(*(P-cen).T[::-1]); P=P[np.argsort(ang)]
        d=tuple(((T.ij[e[1]]-T.ij[e[0]]+L//2)%L-L//2).tolist())
        for s in shifts:
            Q=(P+s)@R.T
            if inwin(Q.mean(0)):
                ax.add_patch(Polygon(Q,closed=True,fc=cols.get(d,'#ddd'),ec='#555',lw=0.6))
                if annot and cols.get(d)!='#c9a3d6': dy.append(Q.mean(0)[1])
    # sites
    for i in range(T.N):
        for s in shifts:
            p=(T.pos[i]+s)@R.T
            if inwin(p):
                ax.plot(*p,'o',ms=5.5,mfc='#d62728' if c[i]==0 else '#1f5fbf',mec='k',mew=0.4,zorder=3)
    if annot and dy:
        ys=np.sort(np.array(dy)); groups=[[ys[0]]]
        for y in ys[1:]:
            (groups[-1].append(y) if y-groups[-1][-1]<0.8 else groups.append([y]))
        cs=[np.mean(g) for g in groups]
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((x1,y0-2),5,y1-y0+4,fc='white',ec='none',zorder=5))
        for yc in cs:
            ax.plot([x0,x1,x1,x0,x0],[yc-0.75,yc-0.75,yc+0.75,yc+0.75,yc-0.75],color='#b8860b',lw=1.6,ls='--',zorder=4)
            ax.text(x1+0.3,yc,'dice\n3 rows',va='center',fontsize=8,color='#8b6508',zorder=6)
        for a,b in zip(cs[:-1],cs[1:]):
            ax.text(x1+0.3,(a+b)/2,'square\n6 rows',va='center',fontsize=8,color='#6a3d7a',zorder=6)
    ax.set_xlim(x0,x1+(2.2 if annot else 0)); ax.set_ylim(y0,y1); ax.set_aspect('equal'); ax.axis('off')
    S=abs((c==0).sum()-(c==1).sum())/2
    ax.set_title(title+f"\nS = {S:.0f} of N = {T.N}   (m = {S/T.N:.4f})",fontsize=10)
fig,axs=plt.subplots(1,3,figsize=(18,5.4))
draw(axs[0],18,[[0,18]],'square (4,4,4) tiling:  M = 0\n(stable at h < 0.14 J)')
draw(axs[1],18,[[0,6],[9,6]],'s6d3 superlattice:  M = M_dice / 3\n(stable at 0.14 J < h < 0.22 J)',annot=True)
draw(axs[2],18,[],'dice (6,3,3) tiling:  M = M_dice\n(stable at h > 0.22 J)')
fig.text(0.5,0.03,'red = majority sublattice (spins ∥ h),  blue = minority (spins anti-∥ h).   Lozenge colour = orientation.  '
         'Middle: 6-row square layers (one lozenge orientation) alternate with 3-row dice lamellae (cube pattern).',ha='center',fontsize=9)
plt.tight_layout(rect=(0,0.05,1,1)); plt.savefig('figures/s6d3_superlattice.png',dpi=150)
