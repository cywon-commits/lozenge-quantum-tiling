import json, pickle, sys
from periodic import Cell, replicate
from lswt import lswt_energy
f=sys.argv[1]; L=int(sys.argv[2]); mt=float(sys.argv[3]); out=sys.argv[4]
R=json.load(open(f)); R=[r for r in R if abs(r['m_rep']-mt)<1e-4]; R.sort(key=lambda r:r['e_lswt'])
r=R[0]; C=Cell(*r['cell']); T=replicate(C,[tuple(x) for x in r['tiling']],L)
E,S=lswt_energy(T.N,T.bonds()); print("cell",r['cell'],"L",L,"e_lswt",E/T.N,"m",S/T.N,r['hist'])
pickle.dump(set(T.removed),open(out,'wb'))
