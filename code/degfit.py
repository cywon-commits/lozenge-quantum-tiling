import numpy as np, time
from tilt import strips
from tiling_moves import site_edges, flippable, flip, degrees
from lswt import lswt_energy
rng=np.random.default_rng(0)
X=[];Y=[];info=[]
t=time.time()
for L,iv in ((18,[]),(18,[[0,6]]),(18,[[0,12]]),(18,[[0,18]]),(18,[[0,6],[9,6]])):
    T,ok=strips(L,iv); SE=site_edges(T)
    for nflip in (0,5,20,60,200):
        for rep in range(3):
            T,ok=strips(L,iv)
            for k in range(nflip):
                cand=[v for v in rng.choice(T.N,40,replace=False) if flippable(T,SE,v)]
                if cand: flip(T,SE,cand[0])
            assert not T.check_pairing()
            E,S=lswt_energy(T.N,T.bonds()); z=degrees(T)
            X.append([np.mean(z==k) for k in range(3,7)]); Y.append(E/T.N); info.append((str(iv),nflip,S/T.N))
X=np.array(X);Y=np.array(Y)
print("n tilings",len(Y),"time",time.time()-t)
coef,res,rk,sv=np.linalg.lstsq(X,Y,rcond=None)
pred=X@coef; print("eps(z=3..6) =",np.round(coef,4)," rms resid",np.sqrt(np.mean((pred-Y)**2)), " spread of e",Y.std())
# add net moment term
X2=np.c_[X,[i[2] for i in info]]; c2=np.linalg.lstsq(X2,Y,rcond=None)[0]; p2=X2@c2
print("with |m| term:",np.round(c2,4)," rms",np.sqrt(np.mean((p2-Y)**2)))
np.savez('degfit.npz',X=X,Y=Y)
