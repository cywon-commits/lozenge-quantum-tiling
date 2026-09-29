import numpy as np
from ed_dice import TorusTiling
def build(legs, se=0, T=None):
    T=TorusTiling((3,3),(-3,6))
    e0=sorted(T.removed)[se]; ta,tb=T.create_pair(e0); mono,other=ta,tb
    start=T.tri_cent[mono].copy(); disp=np.zeros(2); path=[disp.copy()]
    for ang,n in legs:
        dv=np.array([np.cos(np.radians(ang)),np.sin(np.radians(ang))]); pv=np.array([-dv[1],dv[0]]); st=disp.copy()
        for k in range(n):
            best=None
            for e in T.tri_edges[mono]:
                a,b=T.edge_tris[e]; tp=b if a==mono else a
                if tp==other: continue
                tpp,_=T.partner(tp); d=T.wrap(T.tri_cent[tpp]-T.tri_cent[mono])
                sc=(d@dv>0.1)*100+d@dv-1.5*abs((disp+d-st)@pv)
                if best is None or sc>best[0]: best=(sc,e,d)
            mono=T.hop(mono,best[1]); disp=disp+best[2]; path.append(disp.copy())
    path=np.array(path); arc=np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1))
    chord=np.linalg.norm(T.wrap(T.tri_cent[mono]-T.tri_cent[other]))
    # rewired rhombi
    rew=[e for e in T.removed if {T.color[e[0]],T.color[e[1]]}!={1,2}]
    return T,mono,other,path,arc,chord,len(rew)
if __name__=="__main__":
    for legs in ([(30,2),(90,2),(150,2)],[(30,3),(90,2),(150,3)],[(30,2),(90,3),(150,2)],[(30,3),(90,3),(150,3)],[(0,2),(90,2),(180,2)]):
        for se in range(3):
            try:
                T,m,o,p,arc,ch,nr=build(legs,se)
                print(legs,se,"arc %.2f chord %.2f  rewired rhombi %d  monomers %s  end disp %s"%(arc,ch,nr,T.monomers(),np.round(p[-1],2)))
            except Exception as ex: print(legs,se,"fail",ex)
