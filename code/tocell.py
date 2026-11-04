"""Convert a periodic cell tiling (cell json) to a Tiling(L) removed-edge set."""
import sys, json, numpy as np
sys.path.insert(0,'/home/claude/w/lozenge-quantum-tiling/code')
from dice_string import Tiling
from periodic import Cell, DIRS
from tiling_moves import degrees
def to_tiling(cell, tiling, L):
    p,q,s=cell; C=Cell(p,q,s); rem=set((int(a),int(b)) for a,b in tiling)
    T=Tiling(L); R=set()
    for n in range(T.N):
        i,j=T.ij[n]; ns=C.idx(i,j)
        for d,(di,dj) in enumerate(DIRS):
            if (ns,d) in rem: R.add(tuple(sorted((T.s(i,j),T.s(i+di,j+dj)))))
    assert all(e in T.edge_tris for e in R)
    T.removed=R; assert not T.check_pairing(); return T
if __name__=='__main__':
    print(DIRS)
    D='/mnt/user-data/outputs/.qdec/'
    for f,L in (('m35_cell',60),('x0p2_cell',30),('x0p25_cell',24),('x0p1429_cell',42)):
        r=json.load(open(D+f+'.json')); T=to_tiling(r['cell'],r['tiling'],L); z=degrees(T)
        print(f, r['cell'], r['m'], r.get('hist'), np.bincount(z.astype(int))[3:]/T.N*r['cell'][0]*r['cell'][1], len(T.removed), T.N)
