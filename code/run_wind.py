import numpy as np, sys, pickle, time
from wind import wound, bipartite
from ed_dice import lowest, szmap
w=tuple(int(x) for x in sys.argv[1].split(',')); se=int(sys.argv[2]); sectors=[float(x) for x in sys.argv[3].split(',')]
T,f=wound(*w,start_edge_index=se); B=T.bonds(); N=T.N; ok,c=bipartite(T)
print("winding",w,np.round(f,2),"bipartite",ok,"Lieb S" if ok else "", abs((c==0).sum()-(c==1).sum())/2 if ok else "",flush=True)
out={}
for sz in sectors:
    t=time.time(); w_,v,st=lowest(B,N,int(round(N/2+sz)),k=2)
    out[sz]=dict(E=w_.tolist(),mz=szmap(v[:,0],st,N).tolist())
    print(f"Sz={sz} E={w_} ({time.time()-t:.0f}s)",flush=True)
    pickle.dump(dict(w=w,bonds=B,color=T.color,out=out),open(f"wind_{w[0]}{w[1]}.pkl","wb"))
