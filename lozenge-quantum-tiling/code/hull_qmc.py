import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hull_lswt import lower_hull
pts=[(0.0,-0.66920,'square'),(1/18,-0.66193,'s6d3 (strips)'),(1/18,-0.66016,'strip w12'),(1/9,-0.64977,'strip w6'),(1/6,-0.63794,'dice')]
for l in open('qhull.log'):
    if l.startswith('{'):
        r=json.loads(l); mL=r['S_lieb']/324; e0=r['e']+r['h']*(r['m']+mL)/2
        pts.append((mL,e0,'annealed '+r['pkl'].replace('ann_18_','').replace('.pkl','')))
m=np.array([p[0] for p in pts]); e=np.array([p[1] for p in pts])
H=lower_hull(m,e)
print("QMC lower hull, L=18 (e0 = e + h(m+mL)/2 from h=0.15 runs):")
for a,b in zip(H[:-1],H[1:]): print(f"  {pts[a][2]:40s} m={m[a]:.4f} e0={e[a]:.5f} -> step h={(e[b]-e[a])/(m[b]-m[a]):.3f}")
print(f"  {pts[H[-1]][2]:40s} m={m[H[-1]]:.4f} e0={e[H[-1]]:.5f}")
print("\nall points:"); [print(f"  {p[2]:40s} m={p[0]:.4f} e0={p[1]:.5f}") for p in sorted(pts)]
fig,ax=plt.subplots(1,2,figsize=(12,4.5))
for p in pts: ax[0].plot(p[0],p[1],'s' if 'annealed' not in p[2] else 'o',color='C3' if 'annealed' not in p[2] else 'C0',ms=7)
ax[0].plot(m[H],e[H],'k-',lw=1); ax[0].set_xlabel('m'); ax[0].set_ylabel('e0 per site (QMC, J)')
ax[0].plot([],[],'s',color='C3',label='strip structures'); ax[0].plot([],[],'o',color='C0',label='LSWT-annealed structures (QMC-evaluated)'); ax[0].legend(fontsize=8)
ax[0].set_title('S=½ QMC, L=18: lower convex hull')
hs=np.linspace(0,0.3,1201); G=e[None,:]-hs[:,None]*m[None,:]
chi=np.zeros_like(m); chi[m==0]=0.065; G=G-0.5*chi[None,:]*hs[:,None]**2
ax[1].plot(hs,m[G.argmin(1)],'k-'); ax[1].plot([0,0,0.3],[0,1/6,1/6],'r--',label='classical')
ax[1].set_xlabel('h / J'); ax[1].set_ylabel('selected m'); ax[1].set_title('T=0 selection (QMC energies)'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('figures/qmc_hull_L18.png',dpi=140)
