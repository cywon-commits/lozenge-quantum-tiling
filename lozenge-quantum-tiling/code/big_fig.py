import json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dice_string import Tiling
from draw_hull import draw
ESQ,ED=-0.66917,-0.63786; lin=lambda m: ESQ+(ED-ESQ)*6*m
def load(f):
    out=[]
    for l in open(f):
        if l.startswith('{'):
            r=json.loads(l); N=r['L']**2; mL=r['S_lieb']/N; out.append((mL,r['e']+r['h']*(r['m']+mL)/2,r['e_err']))
    return out
new={}
for f in ('qbig_m9.log','qbig_m12.log','qbig_m36.log','qbig_m18.log'):
    try:
        v=load(f); m=v[0][0]; new[round(m,5)]=(m,np.mean([x[1] for x in v]),max(3e-4/np.sqrt(len(v)),max(x[2] for x in v)))
    except Exception as ex: print(f,ex)
# previous periodic QMC (qc.log, L=18/24): best per m
old={}
for m,e,er in load('qc.log'):
    k=round(m,5)
    if k not in old or e<old[k][1]: old[k]=(m,e,er)
pts={0.0:(0.0,ESQ,3e-4),round(1/6,5):(1/6,ED,3e-4)}
pts.update(old); pts.update(new)   # L=36 values supersede
P=sorted(pts.values()); m=np.array([p[0] for p in P]); e=np.array([p[1] for p in P]); er=np.array([p[2] for p in P])
for p in P: print(f"m={p[0]:.4f} M/Md={6*p[0]:.3f} e0={p[1]:.5f} dev={p[1]-lin(p[0]):+.5f} ±{p[2]:.5f}")
hs=np.linspace(0,0.35,3501); chi=np.where(m==0,0.065,0.0)
G=e[None,:]-hs[:,None]*m[None,:]-0.5*chi[None,:]*hs[:,None]**2; sel=G.argmin(1)
steps=[(round(hs[i],3),round(6*m[sel[i-1]],3),round(6*m[sel[i]],3)) for i in range(1,len(hs)) if sel[i]!=sel[i-1]]
print("steps (h, M/Md from, to):",steps)
json.dump(dict(points=[list(p) for p in P],steps=steps),open('big_staircase.json','w'))
fig=plt.figure(figsize=(15,4.8))
ax=fig.add_subplot(1,3,1)
ax.errorbar(m*6,e-lin(m),er,fmt='o-',color='C3',ms=6)
for k in new: ax.plot(6*new[k][0],new[k][1]-lin(new[k][0]),'s',mfc='none',mec='k',ms=11)
ax.axhline(0,color='k',lw=.5); ax.set_xlabel('M / M_dice'); ax.set_ylabel('e0 − linear (J/site)'); ax.set_title('QMC (S=½) periodic crystals; □ = L=36 (cells ≤36)',fontsize=10)
ax=fig.add_subplot(1,3,2)
ax.plot(hs,6*m[sel],'k-',lw=2); ax.plot([0,0,0.35],[0,1,1],'r--',label='classical')
for y,t in ((1/3,'1/3'),(1/2,'1/2'),(2/3,'2/3')): ax.axhline(y,color='gray',ls=':',lw=.8); ax.text(0.005,y+0.02,t,fontsize=8)
ax.set_xlabel('h / J'); ax.set_ylabel('M / M_dice'); ax.set_title('T=0 magnetization staircase (QMC)',fontsize=10); ax.legend(fontsize=8)
T=Tiling(36); T.removed=pickle.load(open('big_m9_L36.pkl','rb')); ax=fig.add_subplot(1,3,3); draw(ax,T,'new 2M_dice/3 crystal (cell 9×4, s=7)',win=(-1,20,-0.5,16))
plt.tight_layout(); plt.savefig('figures/staircase_cells36.png',dpi=130)
