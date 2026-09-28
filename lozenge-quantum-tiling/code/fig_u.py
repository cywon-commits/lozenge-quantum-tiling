import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from u_analyze import analyze
from ushape import build
C=pickle.load(open('u_classical.pkl','rb'))
fig,axs=plt.subplots(2,3,figsize=(14,8))
for r,(key,legs,se) in enumerate((('A',[(30,2),(90,2),(150,2)],2),('B',[(0,2),(90,2),(180,2)],1))):
    T,m,o,p,arc,ch,nr=build(legs,se); B=T.bonds(); I=C[key]['I']
    ax=axs[r,0]; cen=T.tri_cent[o]
    P=np.array([T.wrap(x-cen) for x in T.pos])
    segs=[[P[i],P[i]+T.wrap(T.pos[j]-T.pos[i])] for i,j in B]
    ax.add_collection(LineCollection(segs,colors='0.75',lw=1))
    rew=[[P[e[0]],P[e[0]]+T.wrap(T.pos[e[1]]-T.pos[e[0]])] for e in T.removed if {T.color[e[0]],T.color[e[1]]}!={1,2}]
    ax.add_collection(LineCollection(rew,colors='k',lw=2.2,linestyles=':'))
    cc=np.array(['#4a78d6','#3fa06a','#d97a3a'])[T.color]
    ax.scatter(P[:,0],P[:,1],c=cc,s=40,zorder=3)
    ax.scatter(P[I,0],P[I,1],s=220,facecolors='none',edgecolors='m',lw=2,zorder=4)
    for t in (m,o):
        tri=np.array([P[s] for s in T.tri_sites[t]]); c0=tri.mean(0)
        ax.fill(*(T.wrap(T.tri_cent[t]-cen)+ (tri-c0)).T,color='red',alpha=.35,zorder=2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{key}: string arc {arc:.0f}a, chord {ch:.2f}a\n(dotted = removed rhombus diagonals of rewired lozenges,\nred = frustrated triangles, magenta = inside sites)',fontsize=9)
    rows,steps,hs,szq,aq=analyze(key)
    Cl=C[key]
    ax=axs[r,1]; ax.plot(hs,szq,'k-',label='quantum ED'); ax.plot(Cl['hs'],Cl['M'],'o',ms=3,mfc='none',color='C0',label='classical')
    ax.set_xlabel('h/J'); ax.set_ylabel('M'); ax.legend(fontsize=8); ax.set_title(f'{key}: magnetization')
    ax=axs[r,2]; ax.plot(hs,aq,'k-',label='quantum ED'); ax.plot(Cl['hs'],Cl['aI'],'o',ms=3,mfc='none',color='C0',label='classical')
    ax.axhline(0,color='gray',lw=.5); ax.set_xlabel('h/J'); ax.set_ylabel('inside-site order  σ_i⟨S^z_i⟩')
    ax.set_title(f'{key}: <0: inside reversed (chord state)  >0: aligned (string state)',fontsize=9); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/u_string_field_27.png',dpi=140)
