import numpy as np, time, sys, json
from sse import run
from dice_string import Tiling
L=int(sys.argv[1]); betas=[float(x) for x in sys.argv[2].split(',')]; steps=[int(x) for x in sys.argv[3].split(',')]
nm=int(sys.argv[4]) if len(sys.argv)>4 else 20000
res=[]
for nst in steps:
    T=Tiling(L)
    if nst>=0:
        e0=sorted(T.removed,key=lambda e:np.linalg.norm(T.pos[e[0]]-np.array([L/4,L/8])))[0]
        ta,tb=T.create_pair(e0); mono,other=ta,tb; cur=T.tri_cent[mono].copy(); p0=cur.copy()
        dv=np.array([np.sqrt(3)/2,.5]); pv=np.array([-.5,np.sqrt(3)/2])
        for k in range(nst):
            mono,dd=T.move(mono,other,dv,offset=(cur-p0)@pv); cur=cur+dd
        d=np.linalg.norm(T.wrap(cur-T.tri_cent[other]))
    else: d=0.0
    B=T.bonds()
    init=np.where(T.color==0,-1,1)
    for beta in betas:
        t=time.time(); sg,m2,E=run(B,T.N,beta,2000,nm,7,init_spin=init)
        s=sg.mean(); nb=20; sb=sg.reshape(nb,-1).mean(1)
        mw=(m2*sg).reshape(nb,-1).mean(1)/sb
        r=dict(L=L,steps=nst,d=round(d,2),beta=beta,sign=s,sign_err=sb.std()/np.sqrt(nb),M2=float(mw.mean()),M2_err=float(mw.std()/np.sqrt(nb)),E=float((E*sg).mean()/s),t=time.time()-t)
        print(json.dumps(r),flush=True); res.append(r)
json.dump(res,open(f"qsign_L{L}_{sys.argv[3].replace(',','_')}.json","w"))
