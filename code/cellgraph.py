"""Periodic graph of a cell tiling repeated k x k times (torus spanned by k*v1, k*v2). Returns N, bonds (numpy), z."""
import sys, numpy as np
sys.path.insert(0,'/home/claude/w/lozenge-quantum-tiling/code')
from periodic import Cell, DIRS
def cell_graph(cell, tiling, k):
    p,q,s=cell; small=Cell(p,q,s); big=Cell(k*p,k*q,k*s)
    rem=set((int(a),int(b)) for a,b in tiling)          # (site index in small cell, direction)
    bonds=[]
    for n in range(big.N):
        i,j=big.site(n)
        ns=small.idx(i,j)
        for d,(di,dj) in enumerate(DIRS):
            if (ns,d) in rem: continue
            bonds.append((n, big.idx(i+di,j+dj)))
    b=np.array(bonds); z=np.bincount(b.ravel(),minlength=big.N)
    assert len(b)==2*big.N, (len(b),big.N)
    return big.N, b, z
if __name__=='__main__':
    import json
    r=json.load(open(sys.argv[1])); k=int(sys.argv[2])
    N,b,z=cell_graph(r['cell'],r['tiling'],k); print(N,len(b),np.bincount(z))
