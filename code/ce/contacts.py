import json, numpy as np
from features import features, NAMES, SHELLS
R=json.load(open('hubfree_lswt.json')); X=np.load('X.npy'); y=np.array([r['e_inf'] for r in R]); m=X[:,1]; nd=X[:,0]
# polarised structures: every defect has the same moment sign  <=>  |m| = n_def/8
pol=np.abs(m-nd/8)<1e-9
print('polarised structures:',pol.sum(),'of',len(y))
# per defect, count neighbours at each shell that are same-kind (same z) / other-kind, for the best structure per m
esq=-0.65795; ED=-0.63315; lin=lambda mm: esq+(ED-esq)*6*mm
for mm in sorted(set(np.round(m[pol],6))):
    idx=np.where(pol&(np.round(m,6)==mm))[0]
    if mm==0: continue
    order=idx[np.argsort(y[idx])]
    for tag,i in (('best',order[0]),('worst',order[-1])):
        f=X[i]; rho=f[0]
        row=[]
        for s in (1,3,4,7):
            b=2+4*SHELLS.index(s)
            same=f[b+0]+f[b+3]*0   # (kk=+1,ee=+1) -> same z, same sublattice
            opp=f[b+3]             # (kk=-1,ee=-1) -> 3-5 on opposite sublattice
            row.append(f"r2={s}: like {2*f[b]/rho:4.2f} unlike {2*f[b+3]/rho:4.2f}")
        print(f"M/Md={6*mm:.3f} {tag:5s} n={len(idx):4d} dev={1e3*(y[i]-lin(mm)):+.3f}  "+' | '.join(row))
