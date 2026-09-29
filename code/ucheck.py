import numpy as np
from ushape import build
from wind import bipartite
def close_check(legs,se):
    T,m,o,p,arc,ch,nr=build(legs,se)
    # greedily walk mono to other and annihilate
    for it in range(20):
        sh=[e for e in T.tri_edges[m] if o in T.edge_tris[e]]
        if sh: T.removed.add(sh[0]); break
        best=None
        for e in T.tri_edges[m]:
            a,b=T.edge_tris[e]; tp=b if a==m else a
            tpp,_=T.partner(tp); dd=np.linalg.norm(T.wrap(T.tri_cent[tpp]-T.tri_cent[o]))
            if best is None or dd<best[0]: best=(dd,e)
        m=T.hop(m,best[1])
    ok,c=bipartite(T)
    return ok, (abs((c==0).sum()-(c==1).sum())/2 if ok else None), T.monomers()
for legs,se in (([(30,2),(90,2),(150,2)],2),([(0,2),(90,2),(180,2)],1),([(30,2),(90,2),(150,2)],0),([(30,3),(90,3),(150,3)],0),([(0,2),(90,2),(180,2)],0)):
    print(legs,se,close_check(legs,se))
