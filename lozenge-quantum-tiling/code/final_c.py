import json, pickle, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from hull_lswt import lower_hull
from dice_string import Tiling
from draw_hull import draw
ESQ,ED=-0.66917,-0.63786
lin=lambda m: ESQ+(ED-ESQ)*6*m
P=[]
for l in open('qc.log'):
    if l.startswith('{'):
        r=json.loads(l); N=r['L']**2; mL=r['S_lieb']/N; e0=r['e']+r['h']*(r['m']+mL)/2
        P.append(dict(m=mL,e0=e0,err=r['e_err'],lab=r['pkl'].replace('.pkl',''),L=r['L']))
A=[]
for f in ('qhull.log','qb24.log','qb30.log'):
    for l in open(f):
        if l.startswith('{'):
            r=json.loads(l)
            if r.get('kind')!='pkl': continue
            N=r['L']**2; mL=r['S_lieb']/N; A.append(dict(m=mL,e0=r['e']+r['h']*(r['m']+mL)/2,err=r['e_err'],L=r['L']))
# periodic hull (best per m)
best={}
for p in P:
    k=round(p['m'],5)
    if k not in best or p['e0']-lin(p['m'])<best[k]['e0']-lin(best[k]['m']): best[k]=p
pts=[dict(m=0.0,e0=ESQ,lab='square')]+list(best.values())+[dict(m=1/6,e0=ED,lab='dice')]
pts.sort(key=lambda p:p['m'])
m=np.array([p['m'] for p in pts]); e=np.array([p['e0'] for p in pts])
for p in pts: print(f"m={p['m']:.4f} e0={p['e0']:.5f} dev={p['e0']-lin(p['m']):+.5f} {p['lab']}")
hs=np.linspace(0,0.35,3501); chi=np.where(m==0,0.065,0.0)
G=e[None,:]-hs[:,None]*m[None,:]-0.5*chi[None,:]*hs[:,None]**2; sel=G.argmin(1)
steps=[(hs[i],m[sel[i-1]],m[sel[i]]) for i in range(1,len(hs)) if sel[i]!=sel[i-1]]
print("steps:",[(round(a,3),round(b,4),round(c,4)) for a,b,c in steps])
fig=plt.figure(figsize=(16,9))
ax=fig.add_subplot(2,3,1)
ax.errorbar([a['m'] for a in A],[a['e0']-lin(a['m']) for a in A],[a['err'] for a in A],fmt='o',color='gray',alpha=.5,ms=4,label='LSWT-annealed (irregular), QMC')
ax.errorbar([p['m'] for p in P],[p['e0']-lin(p['m']) for p in P],[p['err'] for p in P],fmt='s',color='C3',ms=7,label='periodic small-cell crystals, QMC')
ax.axhline(0,color='k',lw=.5); ax.set_xlabel('m'); ax.set_ylabel('e0 − linear(square→dice)  (J/site)'); ax.legend(fontsize=8)
ax.set_title('QMC (S=½): periodic crystals vs irregular networks',fontsize=10)
ax=fig.add_subplot(2,3,4)
ax.plot(hs,m[sel]*6,'k-',lw=2); ax.plot([0,0,0.35],[0,1,1],'r--',label='classical')
for y,t in ((1/3,'M_dice/3'),(1/2,'M_dice/2')): ax.axhline(y,color='gray',ls=':',lw=.8); ax.text(0.005,y+0.02,t,fontsize=8)
ax.set_xlabel('h / J'); ax.set_ylabel('M / M_dice'); ax.set_title('T=0 magnetization from periodic crystals (QMC)',fontsize=10); ax.legend(fontsize=8)
for k,(pk,L,t) in enumerate((('per2_m18_L24.pkl',24,'M_dice/3 crystal (cell 3×12)'),('per2_m12_L24.pkl',24,'M_dice/2 crystal (cell 6×4), no z=6 hubs'))):
    try:
        T=Tiling(L); T.removed=pickle.load(open(pk,'rb')); ax=fig.add_subplot(2,3,2+3*k if False else [2,3][k]); draw(ax,T,t,win=(-1,16,-0.5,13))
    except Exception as ex: print(ex)
from tilt import strips
ax=fig.add_subplot(2,3,5); T,_=strips(24,[[0,24]]); draw(ax,T,'square (4,4,4), M=0',win=(-1,16,-0.5,13))
ax=fig.add_subplot(2,3,6); T,_=strips(24,[]); draw(ax,T,'dice (6,3,3), M=M_dice',win=(-1,16,-0.5,13))
plt.tight_layout(); plt.savefig('figures/periodic_crystals_qmc.png',dpi=120)
