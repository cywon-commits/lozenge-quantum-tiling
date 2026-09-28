import numpy as np, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from run_ed import config
from wind import wound
from dice_string import minimize_spins
hi=pickle.load(open('hi_sectors.pkl','rb'))
def low(f): return {float(k):v['E'][0] for k,v in pickle.load(open(f,'rb'))['out'].items()}
E={'dice':low('ed_nh-1.pkl'),'pair_d0.58':low('ed_nh0.pkl'),'pair_d1.15':low('ed_nh1.pkl'),'pair_d2.31':low('ed_nh3.pkl'),
   'wind10':low('wind_10.pkl'),'wind11':low('wind_11.pkl'),'wind20':{**low('wind_20_hi.pkl'),**low('wind_20.pkl')}}
for k in E: E[k].update({float(s):v for s,v in hi[k].items()})
hs=np.linspace(0,4.2,841)
def curve(Es):
    sz=np.array(sorted(Es)); e=np.array([Es[s] for s in sz])
    G=e[None,:]-hs[:,None]*sz[None,:]; i=G.argmin(1); return sz[i],G.min(1)
Q={k:curve(v) for k,v in E.items()}
# classical
cfg={'dice':('p',-1),'pair_d0.58':('p',0),'pair_d1.15':('p',1),'pair_d2.31':('p',3),'wind10':('w',((1,0),0)),'wind11':('w',((1,1),0)),'wind20':('w',((2,0),1))}
hc=np.linspace(0.1,4.2,42); Cl={}
import os
if os.path.exists('field_curves.pkl'):
    _D=pickle.load(open('field_curves.pkl','rb')); Cl={k:(v[0][1:],v[1][1:]) for k,v in _D['Cl'].items()}; hc=_D['hc'][1:]; cfg={}
for k,(kind,arg) in cfg.items():
    T=config(arg)[0] if kind=='p' else wound(*arg[0],start_edge_index=arg[1])[0]; B=T.bonds(); mu=np.full(27,0.5)
    Ms=[];Es=[]
    for h in hc:
        best=None
        for sd in range(12):
            e,s,_=minimize_spins(B,mu,h=h,seed=sd,tol=1e-13)
            if best is None or e<best[0]: best=(e,s)
        Es.append(best[0]); Ms.append(0.5*best[1][:,2].sum())
    Cl[k]=(np.array(Ms),np.array(Es)); print(k,'done',flush=True)

fig,ax=plt.subplots(1,3,figsize=(15,4.3))
cols={'dice':'k','pair_d0.58':'C0','pair_d1.15':'C1','pair_d2.31':'C3','wind10':'C2','wind11':'C4','wind20':'C5'}
lab={'dice':'dice (no defect)','pair_d0.58':'pair d=0.58','pair_d1.15':'pair d=1.15','pair_d2.31':'pair d=2.31','wind10':'string (1,0)','wind11':'string (1,1)','wind20':'string (2,0)'}
for k in ['dice','pair_d0.58','pair_d1.15','pair_d2.31']:
    ax[0].plot(hs,Q[k][0],color=cols[k],label=lab[k]); ax[0].plot(hc,Cl[k][0],'o',ms=3,mfc='none',color=cols[k])
for k in ['dice','wind10','wind11','wind20']:
    ax[1].plot(hs,Q[k][0],color=cols[k],label=lab[k]); ax[1].plot(hc,Cl[k][0],'o',ms=3,mfc='none',color=cols[k])
for a in ax[:2]: a.set_xlabel('h / J'); a.set_ylabel('M = S^z_tot'); a.legend(fontsize=8); a.axhline(4.5,color='gray',lw=.4,ls=':')
ax[0].set_title('triangle pairs (lines: ED, circles: classical)')
ax[1].set_title('torus-winding strings')
for k in ['pair_d1.15','pair_d2.31','wind10','wind11','wind20']:
    ax[2].plot(hs,Q[k][1]-Q['dice'][1],color=cols[k],label=lab[k]+' (Q)')
    ax[2].plot(hc,Cl[k][1]-Cl['dice'][1],'--',color=cols[k],lw=1)
ax[2].axhline(0,color='k',lw=.6); ax[2].set_xlabel('h / J'); ax[2].set_ylabel('E_g(h) − E_g,dice(h)  (J)')
ax[2].set_title('E_g(h) − E_g,dice(h): solid quantum, dashed classical'); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig('figures/field_ed_27.png',dpi=140)
for k in Q:
    sz,g=Q[k]; ch=hs[1:][np.diff(sz)!=0]
    print(k,"M(0)=",sz[0]," first steps at h =",np.round(ch[:4],3), " sat h=",ch[-1] if len(ch) else None)
dq=Q['wind20'][1]-Q['dice'][1]; print("wind20 vs dice crossing h:",hs[np.where(np.diff(np.sign(dq)))[0]])
dq=Q['wind11'][1]-Q['dice'][1]; print("wind11 vs dice: min over h",dq.min(), "at h=0",dq[0])
