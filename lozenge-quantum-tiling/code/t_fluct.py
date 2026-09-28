import numpy as np, time
from tilt import strips
from lswt_fluct import fluct
from lswt import signs
for L,iv,lab in ((24,[[0,24]],'square'),(24,[],'dice')):
    T,ok=strips(L,iv); B=T.bonds(); c=signs(T.N,B)
    M=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
    f=np.array([np.linalg.solve(M,p) for p in T.pos]); reg=np.where((f[:,0]%1)<0.5)[0]
    t=time.time(); o=fluct(T.N,B,region=reg)
    maj = c==(1 if c.sum()>=0 else -1)
    print(lab,"zero modes",o['nzero'],"<n>=%.4f  n_maj=%.4f n_min=%.4f  s_site=%.4f  S_half/(2L)=%.4f  t=%.1fs"%(o['n'].mean(),o['n'][maj].mean(),o['n'][~maj].mean(),o['s_site'].mean(),o['S_region']/(2*L),time.time()-t))
    for zz in np.unique(o['z']): print("   z=%d  n=%.4f"%(zz,o['n'][o['z']==zz].mean()))
