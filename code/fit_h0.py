import numpy as np, pickle, sys
from torus import G_torus
D=pickle.load(open(sys.argv[1],'rb')); L=D['L']; k=0.5
out=D['out']; 
rs=[o['cur']-D['pother'] for o in out]; d=np.array([o['d'] for o in out]); E=np.array([o['dE_frust'] for o in out])
G,area=G_torus(rs,L)
F=4*np.pi**2*k**2*(-G)+2*np.pi**2*k**2*d**2/area
sel=d>4
a,c=np.polyfit(F[sel],E[sel],1)
res=E[sel]-(a*F[sel]+c)
print("fit rho_s =",a," (theory 2/sqrt3*mu_h*mu_r =", 2/np.sqrt(3)*D['mu'].max()*D['mu'].min() if False else None,") max resid",np.abs(res).max())
# plain log fit small d
sel2=(d>4)&(d<L/6)
print("plain log slope small d:",np.polyfit(np.log(d[sel2]),E[sel2],1)[0])
np.save(sys.argv[1]+'.fit.npy',np.c_[d,E,a*F+c])
