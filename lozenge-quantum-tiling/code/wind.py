import numpy as np
from ed_dice import TorusTiling
from collections import deque
def wound(wa,wb,start_edge_index=0):
    T=TorusTiling((3,3),(-3,6))
    if (wa,wb)==(0,0): return T,np.zeros(2)
    Mt=np.array([[1,0],[0.5,np.sqrt(3)/2]]).T@T.T   # torus vectors in cartesian (columns)
    target=Mt@np.array([wa,wb],float); dv=target/np.linalg.norm(target); pv=np.array([-dv[1],dv[0]])
    e0=sorted(T.removed)[start_edge_index]; ta,tb=T.create_pair(e0); mono,other=ta,tb
    disp=np.zeros(2)
    for step in range(200):
        # annihilate if adjacent and travelled nearly the full winding
        if np.linalg.norm(disp-target)<1.5:
            shared=[e for e in T.tri_edges[mono] if other in T.edge_tris[e]]
            if shared:
                T.removed.add(shared[0]); break
        best=None
        for e in T.tri_edges[mono]:
            a,b=T.edge_tris[e]; tp=b if a==mono else a
            if tp==other: continue
            tpp,_=T.partner(tp); d=T.wrap(T.tri_cent[tpp]-T.tri_cent[mono])
            nd=disp+d; score=-np.linalg.norm(target-nd)  # greedy toward target
            score-=0.5*abs((nd)@pv)
            if best is None or score>best[0]: best=(score,e,d)
        mono=T.hop(mono,best[1]); disp=disp+best[2]
    assert not T.monomers(), "not closed"
    f=np.linalg.solve(Mt,disp)
    return T,f
def bipartite(T):
    B=T.bonds(); adj=[[] for _ in range(T.N)]
    for i,j in B: adj[i].append(j); adj[j].append(i)
    c=-np.ones(T.N,int); c[0]=0; q=deque([0]); ok=True
    while q:
        u=q.popleft()
        for v in adj[u]:
            if c[v]<0: c[v]=1-c[u]; q.append(v)
            elif c[v]==c[u]: ok=False
    return ok,c
if __name__=="__main__":
    for w in ((0,0),(1,0),(1,1),(2,0)):
        T,f=wound(*w); ok,c=bipartite(T)
        types={}
        for e in T.removed:
            k=tuple(sorted((T.color[e[0]],T.color[e[1]]))); types[k]=types.get(k,0)+1
        deg=np.bincount(T.bonds().ravel(),minlength=T.N)
        print(w,"winding",np.round(f,2),"bipartite",ok,"rhombus types",types,"degrees",np.bincount(deg))
