import json, pickle, numpy as np, sys
import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from dice_string import Tiling
from lswt import lswt_energy
from periodic import Cell, replicate, DIRS
from features import features, NAMES, SHELLS
L=36; C=Cell(L,L,0)
F=json.load(open('fit.json'))['+ pairs up to r2<=7']; cols=F['cols']; w=np.array(F['w'])
def to_cell(T):
    til=[]
    for x,y in T.removed:
        i,j=divmod(x,L)
        for a,b in ((x,y),(y,x)):
            i,j=divmod(a,L)
            for d,(di,dj) in enumerate(DIRS):
                if C.idx(i+di,j+dj)==b: til.append((a,d)); break
            else: continue
            break
    return til
def contacts(f):
    rho=f[0]; b=2
    return 2*f[b+3]/rho, 2*f[b]/rho      # 3-5 NN (opp. sublattice), like NN (same sublattice)
H={round(p['m'],6):p for p in json.load(open('ce_hull.json'))['points']}
# need the tilings of the LSWT-best: re-find in cand files
import glob
best={}
for fn in glob.glob('cand_*.json'):
    for r in json.load(open(fn)):
        k=round(r['m'],6)
        if k in H and r['cell']==H[k]['cell'] and r['hist']==H[k]['hist']:
            best.setdefault(k,[]).append(r)
for pk,mm in (('big_m36_L36.pkl',1/36),('big_m18_L36.pkl',1/18),('big_m12_L36.pkl',1/12)):
    T=Tiling(L); T.removed=pickle.load(open('../'+pk,'rb'))
    Eq=lswt_energy(T.N,T.bonds())[0]/T.N; fq=features((L,L,0),to_cell(T))
    k=round(mm,6); cands=[]
    for r in best.get(k,[]):
        T2=replicate(Cell(*r['cell']),[tuple(x) for x in r['tiling']],L); E2=lswt_energy(T2.N,T2.bonds())[0]/T2.N
        cands.append((E2,r))
    E2,r=min(cands,key=lambda t:t[0]); f2=features(r['cell'],r['tiling'])
    print(f"M/Md={6*mm:.3f}  QMC-crystal LSWT(L36)={Eq:.6f}  LSWT-best={E2:.6f}  diff={1e6*(Eq-E2):+.1f}e-6 | contacts(3-5, like) QMC {contacts(fq)[0]:.2f},{contacts(fq)[1]:.2f}  LSWT-best {contacts(f2)[0]:.2f},{contacts(f2)[1]:.2f}")
