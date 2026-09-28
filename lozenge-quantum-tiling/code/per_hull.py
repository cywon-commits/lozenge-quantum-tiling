import json, sys, numpy as np
from hull_lswt import lower_hull
from periodic import Cell, replicate
from lswt import lswt_energy
f=sys.argv[1]; L=int(sys.argv[2]); ntop=int(sys.argv[3]) if len(sys.argv)>3 else 3
R=json.load(open(f))
# best model energy per distinct m
best={}
for r in R:
    k=round(r['m'],6)
    best.setdefault(k,[]).append(r)
cands=[]
for k,v in best.items():
    v.sort(key=lambda r:r['e_model']); cands+=v[:ntop]
print("distinct m:",len(best),"candidates for LSWT:",len(cands))
out=[]
for r in cands:
    C=Cell(*r['cell']); T=replicate(C,[tuple(x) for x in r['tiling']],L)
    E,S=lswt_energy(T.N,T.bonds())
    out.append(dict(r,e_lswt=E/T.N,m_rep=S/T.N))
json.dump(out,open(f.replace('.json','_lswt.json'),'w'))
m=np.array([o['m_rep'] for o in out]); e=np.array([o['e_lswt'] for o in out])
esq=e[m==0].min(); ed=e[np.isclose(m,1/6)].min()
H=lower_hull(m,e)
print("LSWT hull of periodic structures (L=%d):"%L)
for a,b in zip(H[:-1],H[1:]):
    o=out[a]; print(f"  m={m[a]:.4f} e={e[a]:.5f} dev={e[a]-(esq+(ed-esq)*6*m[a]):+.5f} cell={o['cell']} hist={o['hist']} -> h={(e[b]-e[a])/(m[b]-m[a]):.3f}")
o=out[H[-1]]; print(f"  m={m[H[-1]]:.4f} e={e[H[-1]]:.5f} cell={o['cell']}")
