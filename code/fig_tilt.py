import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
rows=[r for r in (json.loads(l) for f in ['q1.log','q2.log','q3.log','q4.log']+glob.glob('r_*.log') for l in open(f) if l.startswith('{')) if 'kind' in r]
def m_of(r): return r['S_lieb']/(r['L']**2)
fig,ax=plt.subplots(1,2,figsize=(12,4.6))
res={}
for L,mk in ((12,'s'),(18,'o')):
    tl=[r for r in rows if r['L']==L and r['h']==0 and (r['kind'] in ('tilt','square') or r['kind']=='dice') and r['beta'] in ((12,24) if L==12 else (18,))]
    d={}
    for r in tl: d.setdefault(m_of(r),[]).append(r['e'])
    ms=np.array(sorted(d)); es=np.array([np.mean(d[m]) for m in ms]); er=np.array([max(0.0005,np.std(d[m])/np.sqrt(len(d[m])) if len(d[m])>1 else 0.0006) for m in ms])
    res[L]=(ms,es,er)
    ax[0].errorbar(ms,es,er,marker=mk,ls='-',label=f'winding (tilt) family, L={L}')
fl=[r for r in rows if r['L']==12 and r['h']==0 and r['kind'] in ('random','stripe') and r['beta']==12]
ax[0].plot([m_of(r) for r in fl],[r['e'] for r in fl],'x',color='gray',label='local hexagon flips (random/stripe), L=12')

ax[0].set_ylim(-0.675,-0.63)
ax[0].set_xlabel('Lieb moment per site  m = S/N'); ax[0].set_ylabel('ground-state energy per site e (J)'); ax[0].legend(fontsize=8)
ax[0].set_title('S=½ QMC: quantum fluctuations favour the square (4,4,4) tiling')
# Legendre staircase for L=18
ms,es,er=res[18]; hs=np.linspace(0,0.35,701)
G=es[None,:]-hs[:,None]*ms[None,:]; msel=ms[G.argmin(1)]
ax[1].plot(hs,msel,'k-',label='quantum (QMC, T→0, no in-sector canting)')
ax[1].plot([0,0,0.35],[0,1/6,1/6],'r--',label='classical (degenerate → dice for any h>0)')
sl=np.diff(es)/np.diff(ms)
for k,s in enumerate(sl):
    e=np.sqrt(er[k]**2+er[k+1]**2)/np.diff(ms)[k]; ax[1].axvspan(s-e,s+e,color='C0',alpha=.15)
ax[1].set_xlabel('h / J'); ax[1].set_ylabel('m (tiling sector selected)'); ax[1].legend(fontsize=8)
ax[1].set_title('L=18: square → 1 strip-pair → … → dice, step fields ± 1σ')
plt.tight_layout(); plt.savefig('figures/tilt_staircase_qmc.png',dpi=140)
for L,(ms,es,er) in res.items():
    print(L, [f"m={m:.4f} e={e:.5f}±{r:.5f}" for m,e,r in zip(ms,es,er)])
    sl=np.diff(es)/np.diff(ms); se=[np.sqrt(er[k]**2+er[k+1]**2)/np.diff(ms)[k] for k in range(len(sl))]
    print("   step fields h_k =",[f"{s:.3f}±{e:.3f}" for s,e in zip(sl,se)])
