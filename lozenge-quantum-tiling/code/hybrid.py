import numpy as np
from dice_string import Tiling
from qmc_tiling import lieb
def hybrid(L, w, ndir, off=0.0, sqdir=((1,-1),)):
    T=Tiling(L); n=np.array([np.cos(np.radians(ndir)),np.sin(np.radians(ndir))])
    # periodic projection coordinate: use fractional coordinate along n with torus period
    sel={}
    for t in range(T.T):
        c=T.tri_cent[t]; u=(c@n-off)
        # choose region by fractional coordinate in torus basis for periodicity
        sel[t]=u
    # use fractional coord along a2 (rows) to stay periodic
    M=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
    rem=set()
    for t in range(T.T):
        f=np.linalg.solve(M,T.tri_cent[t]); g=(f@np.array(ndir_vec(ndir)))%1.0
        sq = ((g*L - off)%L) < w
        for e in T.tri_edges[t]:
            d=((T.ij[e[1]]-T.ij[e[0]])%L).tolist()
            if sq and d in ([L-1,1],[1,L-1]): rem.add((e,t))
            if (not sq) and {T.color[e[0]],T.color[e[1]]}=={1,2}: rem.add((e,t))
    es={}
    for e,t in rem: es.setdefault(e,[]).append(t)
    ok=all(len(v)==2 for v in es.values()) and len(es)==T.T//2
    T.removed=set(es)
    return T, ok and not T.check_pairing()
def ndir_vec(k): return {0:(1,0),1:(0,1),2:(1,1),3:(1,-1)}[k]
for k in range(4):
    for w in range(0,13):
        for off in (0,0.5,1,1.5,2):
            T,ok=hybrid(12,w,k,off)
            if ok:
                deg=np.bincount(T.bonds().ravel(),minlength=T.N); print("dir",k,"w",w,"off",off,"S",lieb(T),dict(zip(*np.unique(deg,return_counts=True)))); break
