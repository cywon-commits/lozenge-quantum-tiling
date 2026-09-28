import json, pickle, numpy as np
from dice_string import Tiling
from tilt import strips
from tiling_moves import degrees
from qmc_tiling import lieb
EPS={3:-0.8544,4:-0.6582,5:-0.4423,6:-0.2061}; CM=0.0304
def lsw_model(T):
    z=degrees(T); return sum(EPS[int(k)] for k in z)/T.N + CM*lieb(T)/T.N
X=[];Y=[];labs=[]
for L,iv,e0 in ((18,[[0,18]],-0.66920),(24,[[0,24]],-0.66917),(18,[],-0.63794),(24,[],-0.63786)):
    T,_=strips(L,iv); X.append(lsw_model(T)); Y.append(e0); labs.append('pure')
for o in json.load(open('walls.json')):
    T,_=strips(o['L'],[list(x) for x in o['iv']]); X.append(lsw_model(T)); Y.append(o['e0']); labs.append('strip')
for f in ('qhull.log','qb24.log','qb30.log','qc.log'):
    for l in open(f):
        if l.startswith('{'):
            r=json.loads(l)
            if r.get('kind')!='pkl': continue
            T=Tiling(r['L']); T.removed=pickle.load(open(r['pkl'],'rb'))
            mL=r['S_lieb']/r['L']**2; X.append(lsw_model(T)); Y.append(r['e']+r['h']*(r['m']+mL)/2); labs.append(r['pkl'])
X=np.array(X);Y=np.array(Y)
b,a=np.polyfit(X,Y,1); p=a+b*X
print("e_QMC = %.5f + %.4f * e_LSWTmodel ; rms %.5f"%(a,b,np.sqrt(np.mean((p-Y)**2))))
json.dump(dict(a=a,b=b),open('scaled_model.json','w'))
