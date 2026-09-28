import numpy as np, time, json, sys, pickle
from ed_dice import *
def config(nh):
    T=TorusTiling((3,3),(-3,6))
    if nh<0: return T,None
    e0=sorted(T.removed)[0]; ta,tb=T.create_pair(e0); mono,other=ta,tb
    for k in range(nh):
        best=None
        for e in T.tri_edges[mono]:
            a,b=T.edge_tris[e]; tp=b if a==mono else a
            if tp==other: continue
            tpp,_=T.partner(tp)
            dd=np.linalg.norm(T.wrap(T.tri_cent[tpp]-T.tri_cent[other]))
            if best is None or dd>best[0]: best=(dd,e)
        mono=T.hop(mono,best[1])
    d=np.linalg.norm(T.wrap(T.tri_cent[mono]-T.tri_cent[other]))
    return T,(mono,other,d)
if __name__=="__main__":
    nh=int(sys.argv[1]); sectors=[float(x) for x in sys.argv[2].split(',')]
    T,info=config(nh); B=T.bonds(); N=T.N
    print("config",nh,info,"bonds",len(B),"monomers",T.monomers(),flush=True)
    out={}
    for sz in sectors:
        nup=int(round(N/2+sz)); t=time.time()
        w,v,states=lowest(B,N,nup,k=2)
        mz=szmap(v[:,0],states,N)
        out[sz]=dict(E=w.tolist(),mz=mz.tolist())
        print(f"Sz={sz} dim={len(states)} E={w} ({time.time()-t:.0f}s)",flush=True)
        pickle.dump(dict(nh=nh,info=info,bonds=B,out=out,color=T.color,pos=T.pos),open(f"ed_nh{nh}.pkl","wb"))
