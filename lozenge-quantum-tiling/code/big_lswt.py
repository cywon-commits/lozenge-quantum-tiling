import json, glob, sys
from periodic import Cell, replicate, EPS, C_M
from lswt import lswt_energy
esq=EPS[4]; ed=(EPS[6]+2*EPS[3])/3+C_M/6; lin=lambda m: esq+(ed-esq)*6*m
R=[]
for f in glob.glob('big_*.json'): R+=json.load(open(f))
targets=[float(x) for x in sys.argv[1:]]
seen=set(); out=[]
for r in sorted(R,key=lambda r:r['e_model']):
    if not any(abs(r['m']-t)<1e-4 for t in targets): continue
    if r['e_model']-lin(r['m'])>-0.001: continue
    key=tuple(r["cell"]); cnt=sum(1 for x in out if tuple(x["cell"])==key)
    if cnt>=3: continue
    C=Cell(*r['cell']); T=replicate(C,[tuple(x) for x in r['tiling']],36)
    E,S=lswt_energy(T.N,T.bonds())
    if any(abs(E/T.N-x['e_lswt'])<1e-7 for x in out if tuple(x['cell'])==key): continue
    r2=dict(r,e_lswt=E/T.N,m_rep=S/T.N)
    out.append(r2); print(r['cell'],f"m={r['m']:.4f} mrep={S/T.N:.4f} model_dev={r['e_model']-lin(r['m']):+.5f} lswt={E/T.N:.5f} dev={E/T.N-lin(S/T.N):+.5f}",r['hist'],flush=True)
json.dump(out,open('big_lswt_'+'_'.join(sys.argv[1:])+'.json','w'))
