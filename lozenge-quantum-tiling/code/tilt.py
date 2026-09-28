import numpy as np, sys, json, time
from dice_string import Tiling
from qmc_tiling import lieb
from sse_h import run
def strips(L, intervals):
    """square-type (remove a2-a1 edges) inside row-intervals of the (1,-1) fractional coordinate, dice elsewhere"""
    T=Tiling(L); M=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
    rem={}
    for t in range(T.T):
        f=np.linalg.solve(M,T.tri_cent[t]); g=((f@np.array([1,-1]))%1.0)*L
        sq=any(((g-a)%L)<w for a,w in intervals)
        for e in T.tri_edges[t]:
            d=((T.ij[e[1]]-T.ij[e[0]])%L).tolist()
            if (sq and d in ([L-1,1],[1,L-1])) or ((not sq) and {T.color[e[0]],T.color[e[1]]}=={1,2}):
                rem.setdefault(e,[]).append(t)
    ok=all(len(v)==2 for v in rem.values()) and len(rem)==T.T//2
    T.removed=set(rem)
    return T, ok and not T.check_pairing()
if __name__=="__main__":
    L=int(sys.argv[1]); iv=json.loads(sys.argv[2]); beta=float(sys.argv[3]); h=float(sys.argv[4]); nm=int(sys.argv[5])
    T,ok=strips(L,iv); assert ok; S=lieb(T); assert S is not None
    t=time.time(); sg,mz,E=run(T.bonds(),T.N,beta,max(1000,nm//5),nm,int(sys.argv[6]) if len(sys.argv)>6 else 5,h=h)
    nb=20; Eb=E.reshape(nb,-1).mean(1)/T.N; Mb=mz.reshape(nb,-1).mean(1)/T.N
    print(json.dumps(dict(L=L,kind='tilt',iv=iv,beta=beta,h=h,S_lieb=S,e=Eb.mean(),e_err=Eb.std()/np.sqrt(nb),m=float(np.abs(Mb).mean()),t=time.time()-t)),flush=True)
