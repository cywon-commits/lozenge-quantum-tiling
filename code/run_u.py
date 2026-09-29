import numpy as np, sys, pickle, time
from ushape import build
from ed_dice import lowest, szmap
from wind import bipartite
legs={'A':[(30,2),(90,2),(150,2)],'B':[(0,2),(90,2),(180,2)]}[sys.argv[1]]; se={'A':2,'B':1}[sys.argv[1]]
T,m,o,p,arc,ch,nr=build(legs,se); B=T.bonds()
print(sys.argv[1],"arc",arc,"chord",ch,"monomers",T.monomers(),flush=True)
out={}
for sz in np.arange(0.5,14,1.0):
    nup=int(round(13.5+sz)); t=time.time()
    if nup==27:
        out[sz]=dict(E=[len(B)/4],mz=[0.5]*27); continue
    w,v,st=lowest(B,27,nup,k=2 if nup<24 else 1)
    out[sz]=dict(E=w.tolist(),mz=szmap(v[:,0],st,27).tolist())
    print(f"Sz={sz} E={w} ({time.time()-t:.0f}s)",flush=True)
    pickle.dump(dict(legs=legs,se=se,bonds=B,out=out),open(f"u_{sys.argv[1]}.pkl","wb"))
