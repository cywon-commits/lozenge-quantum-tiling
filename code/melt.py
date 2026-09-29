import numpy as np, json, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dice_string import Tiling
from periodic import Cell, replicate
from tiling_mc import run
R=json.load(open('periodic_all_lswt.json')); R=[r for r in R if abs(r['m_rep']-1/12)<1e-4]; R.sort(key=lambda r:r['e_lswt'])
def crystal(L):
    for r in R:
        try: return replicate(Cell(*r['cell']),[tuple(x) for x in r['tiling']],L)
        except Exception: pass
temps=np.round(np.concatenate([np.linspace(0.005,0.05,19),[0.06,0.08,0.1]]),4)
res={}
for L in (24,48):
    for h in (0.15,0.19,0.23):
        T0=crystal(L); Tr=crystal(L)
        o=run(T0,Tr,h,list(temps),nsweep=4000 if L==24 else 3000,every=10,seed=int(h*100)+L)
        res[f'heat_L{L}_h{h}']=o
        Tm=[r['T'] for r in o]; q=[r['q'] for r in o]
        print(L,h,"q(T):",[round(x,2) for x in q],flush=True)
# cooling from disordered state at L=48 h=0.19
T0=crystal(48); Tr=crystal(48)
o=run(T0,Tr,0.19,list(temps[::-1]),nsweep=3000,every=10,seed=7)
res['cool_L48_h0.19']=o[::-1]; print("cool q:",[round(r['q'],2) for r in o[::-1]])
json.dump(res,open('melt.json','w'))
fig,ax=plt.subplots(1,3,figsize=(15,4.3))
for k,o in res.items():
    Tm=[r['T'] for r in o]
    ls='--' if 'cool' in k else '-'
    ax[0].plot(Tm,[r['q'] for r in o],ls,marker='.',label=k); ax[1].plot(Tm,[6*r['m'] for r in o],ls,marker='.',label=k); ax[2].plot(Tm,[r['cv'] for r in o],ls,marker='.',label=k)
ax[0].set_ylabel('overlap with M_dice/2 crystal'); ax[1].set_ylabel('M / M_dice'); ax[2].set_ylabel('specific heat per site (tiling dof)')
for a in ax: a.set_xlabel('T / J'); a.legend(fontsize=6)
fig.suptitle('Tiling Monte Carlo (local flips, crystal tilt sector), QMC-scaled LSWT coordination model',fontsize=10)
plt.tight_layout(); plt.savefig('figures/crystal_melting.png',dpi=130)
