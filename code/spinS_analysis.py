import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=[json.loads(l) for l in open('qs_S.log') if l.startswith('{')]
D={}
for r in R: D[(r['S'],r['name'])]=r
Ss=sorted(set(r['S'] for r in R))
rows=[]
print(" S   e_sq/S^2   e_dice/S^2   (e_d-e_sq)/S   dev(M2)/S   dev(M3)/S    h(sq->..)  h(M2->dice)")
for S in Ss:
    if not all((S,k) in D for k in ('square','dice')): continue
    esq=D[(S,'square')]['e0']; ed=D[(S,'dice')]['e0']
    lin=lambda m: esq+(ed-esq)*m/(S/3)       # m in units of moment per site; dice m = S/3
    out=dict(S=S,esq=esq,ed=ed,gap=ed-esq)
    for k,mm in (('M2',S/6),('M3',S/9)):
        if (S,k) in D: out['dev_'+k]=D[(S,k)]['e0']-lin(mm)
    if 'dev_M2' in out:
        eM2=D[(S,'M2')]['e0']; out['h_top']=(ed-eM2)/(S/3-S/6)
    rows.append(out)
    print(f"{S:3.1f}  {esq/S**2:+.5f}   {ed/S**2:+.5f}    {(ed-esq)/S:.5f}     {out.get('dev_M2',np.nan)/S:+.5f}   {out.get('dev_M3',np.nan)/S:+.5f}    {out.get('h_top',np.nan):.3f}")
json.dump(rows,open('spinS.json','w'))
fig,ax=plt.subplots(1,2,figsize=(11,4.3))
x=[1/r['S'] for r in rows]
ax[0].plot(x,[r['gap']/r['S']**2 for r in rows],'o-',label='(e_dice − e_square)/S²  (classical: 0)')
ax[0].plot(x,[-r.get('dev_M2',np.nan)/r['S']**2 for r in rows],'s-',label='−dev(M_dice/2 crystal)/S²')
ax[0].set_xlabel('1/S'); ax[0].set_ylabel('energy / (J S²)'); ax[0].legend(fontsize=8); ax[0].set_xlim(0,2.1)
ax[0].set_title('selection energies vanish linearly in 1/S (order by disorder)',fontsize=10)
ax[1].plot([r['S'] for r in rows],[r.get('h_top',np.nan) for r in rows],'o-',label='M/2 → dice field (J)')
ax[1].plot([r['S'] for r in rows],[r.get('h_top',np.nan)/(2*r['S']*6) for r in rows],'s--',label='same / saturation-scale (J·z·S)')
ax[1].set_xlabel('S'); ax[1].legend(fontsize=8); ax[1].set_title('plateau field vs spin size',fontsize=10)
plt.tight_layout(); plt.savefig('figures/spinS_scaling.png',dpi=130)
