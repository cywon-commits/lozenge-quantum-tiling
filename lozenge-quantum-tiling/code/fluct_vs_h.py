import numpy as np, json, pickle, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from dice_string import Tiling
from tilt import strips
from lswt import lswt_energy, signs
from lswt_fluct import fluct
res={}
for L in (18,24,30):
    H=json.load(open(f'hull_fa_{L}.json'))
    M=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
    rows=[]
    for p in H:
        if p['pkl']: T=Tiling(L); T.removed=pickle.load(open(p['pkl'],'rb'))
        else: T,_=strips(L,[] if p['label']=='dice' else [[0,L]])
        B=T.bonds(); f=np.array([np.linalg.solve(M,x) for x in T.pos]); reg=np.where((f[:,0]%1)<0.5)[0]
        o=fluct(T.N,B,region=reg); E,S=lswt_energy(T.N,B)
        c=signs(T.N,B); maj=c==(1 if c.sum()>=0 else -1)
        # ordered moment per site along its sublattice direction, and net moment reduction
        rows.append(dict(m=p['m'],e=p['e'],n=float(o['n'].mean()),s_site=float(o['s_site'].mean()),
                         S_half=o['S_region']/(2*L),zpe=E/T.N+0.5,
                         n_z={int(k):float(o['n'][o['z']==k].mean()) for k in np.unique(o['z'])},
                         frac_z={int(k):float(np.mean(o['z']==k)) for k in np.unique(o['z'])}))
    res[L]=rows
    print(f"L={L}"); [print("  m=%.4f  <dS>=%.4f  s_site=%.4f  S_half/2L=%.3f  ZPE=%.4f"%(r['m'],r['n'],r['s_site'],r['S_half'],r['zpe'])) for r in rows]
json.dump(res,open('fluct_vs_h.json','w'),indent=1)
# field mapping from each hull
fig,axs=plt.subplots(2,2,figsize=(11.5,8))
hs=np.linspace(0,0.3,1201)
for L,col in ((18,'C0'),(24,'C1'),(30,'C2')):
    r=res[L]; m=np.array([x['m'] for x in r]); e=np.array([x['e'] for x in r])
    sel=(e[None,:]-hs[:,None]*m[None,:]).argmin(1)
    for ax,key,lab in ((axs[0,0],'m','Lieb moment per site m'),(axs[0,1],'n','mean spin deviation ⟨ΔS⟩ = ⟨a†a⟩ per site'),
                       (axs[1,0],'s_site','single-site entanglement entropy (nats)'),(axs[1,1],'S_half','half-system entanglement entropy / boundary length')):
        y=np.array([x[key] for x in r])[sel]
        ax.plot(hs,y,color=col,label=f'L={L}')
        ax.set_xlabel('h / J  (LSWT units; QMC fields ≈ 1.25×)'); ax.set_ylabel(lab)
for ax in axs.ravel(): ax.legend(fontsize=8)
axs[0,1].axhline(0,color='k',lw=.5); axs[0,1].text(0.005,0.004,'classical: 0',fontsize=8)
fig.suptitle('S=½ LSWT ground state along the selected tiling sequence (square → dice ribbons → dice)',fontsize=11)
plt.tight_layout(); plt.savefig('figures/fluctuations_vs_field.png',dpi=140)
# per-coordination deviations
print("per-coordination <dS> (L=24):"); [print("  m=%.4f"%x['m'],{k:round(v,3) for k,v in x['n_z'].items()},{k:round(v,2) for k,v in x['frac_z'].items()}) for x in res[24]]
