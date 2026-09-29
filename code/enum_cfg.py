import numpy as np, itertools, json
from tilt import strips
from qmc_tiling import lieb
def enum(L, max_strips=3):
    res={}
    units=L//3
    # square strips as sets of 3-row units; represent config as binary string of length units (1=square unit)
    for bits in itertools.product([0,1],repeat=units):
        if bits[0]==1 and all(bits): pass
        # canonical under cyclic shift
        s=''.join(map(str,bits))
        can=min(s[i:]+s[:i] for i in range(units))
        if can in res: continue
        iv=[]; i=0
        while i<units:
            if bits[i]:
                j=i
                while j<units and bits[j]: j+=1
                iv.append([3*i,3*(j-i)]); i=j
            else: i+=1
        # merge wrap-around
        if len(iv)>1 and iv[0][0]==0 and iv[-1][0]+iv[-1][1]==L:
            iv=[[iv[-1][0],iv[-1][1]+iv[0][1]]]+iv[1:-1]
        ok=False
        for off in (0,0.5):
            T,ok=strips(L,[[a+off,w] for a,w in iv])
            if ok: break
        if not ok: res[can]=None; continue
        S=lieb(T)
        res[can]=dict(iv=[[a+off,w] for a,w in iv],S=S,nwalls=2*len(iv) if 0<sum(bits)<units else 0,frac_sq=sum(bits)/units)
    return res
for L in (18,24):
    r=enum(L); good={k:v for k,v in r.items() if v and v['S'] is not None}
    print("L",L,"valid bipartite:",len(good))
    for k,v in sorted(good.items(),key=lambda kv:(kv[1]['frac_sq'],kv[0])): print("  ",k,v)
    json.dump(good,open(f'cfg{L}.json','w'))
