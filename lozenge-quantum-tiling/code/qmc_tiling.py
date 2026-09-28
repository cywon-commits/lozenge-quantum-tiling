import numpy as np, time, json, sys
from sse_h import run
from dice_string import Tiling
def flip_B(T, sites):
    for v in sites:
        nb=[e for e in T.edge_tris if v in e]
        rem=[e for e in nb if e in T.removed]; pres=[e for e in nb if e not in T.removed]
        assert len(rem)==3 and len(pres)==3
        for e in rem: T.removed.remove(e)
        for e in pres: T.removed.add(e)
    assert not T.check_pairing()
    return T
def pattern(T, kind, f=0.5, seed=0):
    Bs=np.where(T.color==1)[0]; ij=T.ij[Bs]
    if kind=='dice': sel=[]
    elif kind=='stripe': sel=Bs[(ij[:,1]%2)==0]          # every other row of B sites
    elif kind=='random':
        rng=np.random.default_rng(seed); sel=rng.choice(Bs,int(round(f*len(Bs))),replace=False)
    elif kind=='single': sel=Bs[:1]
    elif kind=='square':
        L=T.L; T.removed=set(e for e in T.edge_tris if ((T.ij[e[1]]-T.ij[e[0]])%L).tolist() in ([L-1,1],[1,L-1]))
        return T, float('nan')
    return flip_B(T,sel), len(sel)/len(Bs)
def lieb(T):
    from collections import deque
    B=T.bonds(); adj=[[] for _ in range(T.N)]
    for i,j in B: adj[i].append(j); adj[j].append(i)
    c=-np.ones(T.N,int); c[0]=0; q=deque([0])
    while q:
        u=q.popleft()
        for v in adj[u]:
            if c[v]<0: c[v]=1-c[u]; q.append(v)
            elif c[v]==c[u]: return None
    return abs((c==0).sum()-(c==1).sum())/2
if __name__=="__main__":
    L=int(sys.argv[1]); kind=sys.argv[2]; beta=float(sys.argv[3]); h=float(sys.argv[4]); nm=int(sys.argv[5]); f=float(sys.argv[6]) if len(sys.argv)>6 else 0.5; sd=int(sys.argv[7]) if len(sys.argv)>7 else 0
    T=Tiling(L); T,fr=pattern(T,kind,f,sd); B=T.bonds(); S=lieb(T)
    t=time.time(); sg,mz,E=run(B,T.N,beta,max(1000,nm//5),nm,3,h=h)
    nb=20; Eb=E.reshape(nb,-1).mean(1)/T.N; Mb=mz.reshape(nb,-1).mean(1)/T.N
    r=dict(L=L,kind=kind,f=fr,seed=sd,beta=beta,h=h,S_lieb=S,e=Eb.mean(),e_err=Eb.std()/np.sqrt(nb),m=float(np.abs(Mb).mean()),m_err=Mb.std()/np.sqrt(nb),t=time.time()-t)
    print(json.dumps(r),flush=True)
