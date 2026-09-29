import numpy as np
from ed_dice import TorusTiling
from wind import bipartite
def flip_B(T, sites):
    """flip hexagon around each given B-rim site (color 1): swap its 3 removed edges (to C) with its 3 present edges (to hubs)"""
    for v in sites:
        assert T.color[v]==1
        nb=[e for e in T.edge_tris if v in e]
        rem=[e for e in nb if e in T.removed]; pres=[e for e in nb if e not in T.removed]
        assert len(rem)==3 and len(pres)==3
        for e in rem: T.removed.remove(e)
        for e in pres: T.removed.add(e)
    assert not T.monomers()
    return T
if __name__=="__main__":
    T=TorusTiling((3,3),(-3,6)); Bs=np.where(T.color==1)[0]; print("B sites",Bs)
    for k in range(0,10):
        T=TorusTiling((3,3),(-3,6)); flip_B(T,Bs[:k]); ok,c=bipartite(T)
        print(k,"bipartite",ok,"Lieb S",abs((c==0).sum()-(c==1).sum())/2)
