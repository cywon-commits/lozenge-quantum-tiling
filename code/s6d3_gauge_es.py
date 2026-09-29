import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from collections import deque
from s6d3_props import structs, efield, ent_spectrum
from draw_hull import draw
A1=np.array([1.0,0]); A2=np.array([0.5,np.sqrt(3)/2])
def height(T):
    L=T.L; h={}; start=(0,0); h[start]=0; q=deque([start])
    steps=[((1,0),1),((0,1),-1),((-1,1),1)]   # (+a1, +w), (+a2, -w), (+(a2-a1), +w)
    while q:
        i,j=q.popleft()
        for (di,dj),sg in steps:
            for dirn in (1,-1):
                ni,nj=i+dirn*di,j+dirn*dj
                if abs(ni)>2*L or abs(nj)>2*L: continue
                a,b=T.s(i,j),T.s(ni,nj); e=tuple(sorted((a,b)))
                w=-2 if e in T.removed else 1
                val=h[(i,j)]+dirn*sg*w
                if (ni,nj) not in h: h[(ni,nj)]=val; q.append((ni,nj))
    return h
L=18; St=structs(L)
fig=plt.figure(figsize=(16,10))
for k,(name,T) in enumerate(St.items()):
    ax=fig.add_subplot(2,3,k+1)
    draw(ax,T,f'{name}: tiling + emergent electric field E',win=(-0.5,12,-0.5,10))
    P,E=efield(T)
    R=np.array([[np.cos(np.radians(-30)),-np.sin(np.radians(-30))],[np.sin(np.radians(-30)),np.cos(np.radians(-30))]])
    sh=[a*L*A1+b*L*A2 for a in range(-2,3) for b in range(-2,3)]
    PP=[];EE=[]
    for s in sh:
        Q=(P+s)@R.T; m=(Q[:,0]>-0.5)&(Q[:,0]<12)&(Q[:,1]>-0.5)&(Q[:,1]<10); PP.append(Q[m]); EE.append((E@R.T)[m])
    PP=np.vstack(PP); EE=np.vstack(EE)
    ax.quiver(PP[:,0],PP[:,1],EE[:,0],EE[:,1],color='k',scale=12,width=0.004,zorder=6)
# height profile perpendicular to layers
ax=fig.add_subplot(2,3,4)
for name,T in St.items():
    h=height(T); g=[];hv=[]
    for (i,j),v in h.items():
        if abs(i)<=L and abs(j)<=L: g.append(i-j); hv.append(v)
    g=np.array(g); hv=np.array(hv); ug=np.unique(g); prof=[hv[g==u].mean() for u in ug]
    ax.plot(ug,np.array(prof)-np.interp(0,ug,prof),'.-',label=name,ms=3)
ax.set_xlabel('row index across layers (i − j)'); ax.set_ylabel('height h (layer-averaged)')
ax.set_title('height function = gauge potential: square layers tilted, dice layers flat',fontsize=9); ax.legend(fontsize=8)
# entanglement spectra
for k,(name,T) in enumerate([('square',St['square']),('s6d3',St['s6d3'])]):
    pass
L2=36; St2=structs(L2)
ax=fig.add_subplot(2,3,5); ax2=fig.add_subplot(2,3,6)
for name,T,c in (('square',St2['square'],'C0'),('dice',St2['dice'],'C2'),('s6d3',St2['s6d3'],'C3')):
    ks,eps=ent_spectrum(T)
    a=ax if name=='s6d3' else ax2
    a.plot(ks,eps,'o',ms=3,color=c,label=name,alpha=.8)
for a,t in ((ax,'s6d3'),(ax2,'square vs dice')):
    a.set_ylim(0,8); a.set_xlabel('momentum k along the cut'); a.set_ylabel('single-particle entanglement energy ε'); a.legend(fontsize=8)
    a.set_title(f'entanglement spectrum (LSWT, cut ∥ layers, L={L2}): {t}',fontsize=9)
plt.tight_layout(); plt.savefig('figures/s6d3_gauge_entanglement.png',dpi=120)
print('done')
