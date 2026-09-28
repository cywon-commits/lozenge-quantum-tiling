import json, pickle, glob, numpy as np
from dice_string import Tiling
from tilt import strips
from tiling_moves import degrees
from qmc_tiling import lieb
rows=[]
def add(T,e0,lab):
    z=degrees(T); N=T.N; S=lieb(T)
    rows.append(([1,np.mean(z==4),np.mean(z==5),S/N],e0,lab))
# pure + strips (walls.json has e0 for strip configs)
for L,iv,e0 in ((18,[[0,18]],-0.66920),(24,[[0,24]],-0.66917),(18,[],-0.63794),(24,[],-0.63786)):
    T,_=strips(L,iv); add(T,e0,f'strip{L}{iv}')
for o in json.load(open('walls.json')):
    T,_=strips(o['L'],[list(x) for x in o['iv']]); add(T,o['e0'],f"strip{o['L']}{o['iv']}")
for f in ('qhull.log','qb24.log','qb30.log','qc.log'):
    for l in open(f):
        if l.startswith('{'):
            r=json.loads(l)
            if r.get('kind')!='pkl': continue
            T=Tiling(r['L']); T.removed=pickle.load(open(r['pkl'],'rb'))
            mL=r['S_lieb']/r['L']**2; add(T,r['e']+r['h']*(r['m']+mL)/2,r['pkl'])
X=np.array([r[0] for r in rows]); y=np.array([r[1] for r in rows])
c,res,rk,sv=np.linalg.lstsq(X,y,rcond=None); pred=X@c
print("n=",len(y),"coef const,eps4',eps5',c_m =",np.round(c,5)," rms=",np.sqrt(np.mean((pred-y)**2)))
for (x,yy,lab),p in zip(rows,pred):
    if abs(p-yy)>0.0006 or 'per' in lab: print(f"  {lab:35s} QMC {yy:.5f} model {p:.5f} diff {yy-p:+.5f}")
json.dump(dict(coef=c.tolist()),open('qmc_model.json','w'))
