import numpy as np, pickle, time, sys
from run_ed import config
from wind import wound
from ed_dice import lowest
cfgs={'dice':('p',-1),'pair_d0.58':('p',0),'pair_d1.15':('p',1),'pair_d2.31':('p',3),'wind10':('w',((1,0),0)),'wind11':('w',((1,1),0)),'wind20':('w',((2,0),1))}
res={}
for name,(kind,arg) in cfgs.items():
    T=config(arg)[0] if kind=='p' else wound(*arg[0],start_edge_index=arg[1])[0]
    B=T.bonds(); res[name]={}
    for sz in np.arange(5.5,14,1.0):
        nup=int(round(13.5+sz))
        if nup==27:
            # fully polarized: E = nb/4
            res[name][sz]=len(B)/4; continue
        w,_,_=lowest(B,27,nup,k=1); res[name][sz]=float(w[0])
    print(name,{k:round(v,4) for k,v in res[name].items()},flush=True)
pickle.dump(res,open('hi_sectors.pkl','wb'))
