import numpy as np, pickle
from ushape import build
from wind import bipartite
from dice_string import minimize_spins
import ucheck
def inside_set(legs,se):
    T,m,o,p,arc,ch,nr=build(legs,se)
    for it in range(20):
        sh=[e for e in T.tri_edges[m] if o in T.edge_tris[e]]
        if sh: T.removed.add(sh[0]); break
        best=None
        for e in T.tri_edges[m]:
            a,b=T.edge_tris[e]; tp=b if a==m else a
            tpp,_=T.partner(tp); dd=np.linalg.norm(T.wrap(T.tri_cent[tpp]-T.tri_cent[o]))
            if best is None or dd<best[0]: best=(dd,e)
        m=T.hop(m,best[1])
    ok,c=bipartite(T); sig=np.where(T.color==0,-1,1)   # dice: rims +, hubs -
    cs=np.where(c==0,1,-1)
    if (cs==sig).sum()<(cs==-sig).sum(): cs=-cs
    return np.where(cs!=sig)[0], sig
res={}
for key,(legs,se) in {'A':([(30,2),(90,2),(150,2)],2),'B':([(0,2),(90,2),(180,2)],1)}.items():
    I,sig=inside_set(legs,se); T,m,o,p,arc,ch,nr=build(legs,se); B=T.bonds(); mu=np.full(27,0.5)
    print(key,"inside sites",I,"colors",T.color[I])
    hs=np.linspace(0.02,3.0,50); aI=[];aO=[];M=[];E=[]
    x=None
    for h in hs:
        best=None
        for sd in range(16):
            e,s,_=minimize_spins(B,mu,h=h,seed=sd,tol=1e-13)
            if best is None or e<best[0]: best=(e,s)
        s=best[1]; a=sig*0.5*s[:,2]
        aI.append(a[I].mean()); aO.append(np.delete(a,I).mean()); M.append(0.5*s[:,2].sum()); E.append(best[0])
    res[key]=dict(I=I,sig=sig,hs=hs,aI=np.array(aI),aO=np.array(aO),M=np.array(M),E=np.array(E),arc=arc,chord=ch)
    for h,ai,mm in zip(hs[::5],res[key]['aI'][::5],res[key]['M'][::5]): print(f"  h={h:.2f} a_inside={ai:+.3f} M={mm:.2f}")
pickle.dump(res,open('u_classical.pkl','wb'))
