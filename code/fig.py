import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from dice_string import Tiling
from torus import G_torus
plt.rcParams.update({'font.size':10})
def load(t): return pickle.load(open(t+'.pkl','rb'))
fig=plt.figure(figsize=(13,9.5))
# ---- (a) zero-field map
def spinmap(ax,tag,dtarget,mode):
    D=load(tag); L=D['L']; T=Tiling(L)
    o=min(D['out'],key=lambda o:abs(o['d']-dtarget)); s=o['s']
    hub=T.color==0; P=T.pos
    if mode=='plane':
        S=s[hub]; far=S.mean(0); far/=np.linalg.norm(far)
        # plane: far field and dominant perpendicular component
        perp=S-np.outer(S@far,far); u,sv,vt=np.linalg.svd(perp,full_matrices=False); e2=vt[0]
        ang=np.arctan2(S@e2,S@far); c=ang; cmap='twilight'; vmin,vmax=-np.pi,np.pi; lab='hub-spin angle ψ in rotation plane'
    else:
        S=s[hub]; c=-S[:,2]*np.sign(D['mu'][hub][0]) ; c=-S[:,2]; cmap='coolwarm'; vmin,vmax=-1,1; lab='−S_z (hub)  [M ∥ +z ⇔ hub ↓]'
    # window around string
    cen=(o['cur']+D['pother'])/2
    rel=np.array([T.wrap(p-cen) for p in P[hub]])
    w=(np.abs(rel[:,0])<o['d']/2+14)&(np.abs(rel[:,1])<16)
    sc=ax.scatter(rel[w,0],rel[w,1],c=c[w],cmap=cmap,vmin=vmin,vmax=vmax,s=9)
    segs=[]
    for e in o['removed']:
        if {T.color[e[0]],T.color[e[1]]}!={1,2}:
            a=T.wrap(P[e[0]]-cen); b=a+T.wrap(P[e[1]]-P[e[0]]); segs.append([a,b])
    ax.add_collection(LineCollection(segs,colors='k',lw=1.2))
    for t,mk in ((o['mono'],'^'),(o['other'],'v')):
        p=T.wrap(T.tri_cent[t]-cen); ax.plot(*p,marker='o',ms=9,mfc='none',mec='lime',mew=2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(sc,ax=ax,fraction=0.03,label=lab)
ax=fig.add_subplot(2,2,1); spinmap(ax,'eq96',30,'plane'); ax.set_title('(a) h=0, equal moments: hub-spin angle\n(black = rewired lozenges = string, green = frustrated triangles)')
ax=fig.add_subplot(2,2,2); spinmap(ax,'eqh0.1',30,'z'); ax.set_title('(b) h=0.1: hub −S_z\n(π-twist squeezed into a strip along the string)')
# ---- (c) h=0 energies with torus theory
ax=fig.add_subplot(2,2,3)
for tag,rho_th,col,lab in (('eq96',2/np.sqrt(3),'C0','S1=S2=S3 (ferri), h=0'),('comp_h0',4/np.sqrt(3),'C3','S1=2S2=2S3 (compensated), h=0'),('comp_h0.2',4/np.sqrt(3),'C1','compensated, h=0.2')):
    D=load(tag); d=np.array([o['d'] for o in D['out']]); E=np.array([o['dE_frust'] for o in D['out']])
    G,area=G_torus([o['cur']-D['pother'] for o in D['out']],D['L']); F=np.pi**2*(-G)+np.pi**2/2*d**2/area
    sel=d>4; c=np.mean(E[sel]-rho_th*F[sel])
    E0=E-E[sel][0]
    ax.plot(d,E-c,'o',color=col,ms=4,label=lab)
    ax.plot(d[sel],rho_th*F[sel],'-',color=col,lw=1)
ax.set_xscale('log'); ax.set_xlabel('triangle separation d (a)'); ax.set_ylabel('E_spin(d) − const')
ax.set_title('(c) zero-/compensated: ½-vortex pair on torus\nlines = continuum theory, ρ_s from lattice (no fit)')
ax.legend(fontsize=8)
# ---- (d) tension
ax=fig.add_subplot(2,2,4)
rows=[("eqh0.02",1,1,.02),("eqh0.05",1,1,.05),("eqh0.1",1,1,.1),("eqh0.2",1,1,.2),("r15_h0.1",1.5,1,.1),("r3_h0.1",3,1,.1)]
for tag,mh,mr,h in rows:
    D=load(tag); d=np.array([o['d'] for o in D['out']]); E=np.array([o['dE_frust'] for o in D['out']])
    sel=d>25; s=np.polyfit(d[sel],E[sel],1)[0]
    rho=2/np.sqrt(3)*mh*mr; m=abs(2*mr-mh)/(3*np.sqrt(3)/2); x=np.sqrt(rho*h*m)
    ax.plot(x,s,'o' if mh==1 else 's',color='C0' if mh==1 else 'C2',ms=7)
    ax.annotate(f"S1/S2={mh:g}, h={h}",(x,s),textcoords='offset points',xytext=(5,-10),fontsize=7)
for tag,h in (("comp_h0.1",.1),("comp_h0.2",.2)):
    ax.plot(0,0,'D',color='C3',ms=7)
ax.annotate('compensated (h=0.1, 0.2): no tension',(0,0),textcoords='offset points',xytext=(8,8),fontsize=7,color='C3')
xx=np.linspace(0,0.4,10); ax.plot(xx,8*(1-np.cos(np.pi/4))*xx,'k--',lw=1,label='theory σ = 8(1−cos π/4)√(ρ_s h m)')
ax.set_xlabel('√(ρ_s h m)'); ax.set_ylabel('string tension σ (J/a)'); ax.legend(fontsize=8)
ax.set_title('(d) field-induced linear confinement (ferri only)')
plt.tight_layout(); plt.savefig('figures/string_feasibility.png',dpi=150)
