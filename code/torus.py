import numpy as np
A1=np.array([1.0,0.0]); A2=np.array([0.5,np.sqrt(3)/2])
def G_torus(rs, L, M=300, sig=0.25):
    # periodic Green's function of -Laplacian (minus zero mode) on torus L*A1, L*A2
    Amat=np.array([A1,A2])*L
    B=2*np.pi*np.linalg.inv(Amat).T   # rows b1,b2
    area=abs(np.linalg.det(Amat))
    m,n=np.meshgrid(np.arange(-M,M+1),np.arange(-M,M+1),indexing='ij')
    q=m.ravel()[:,None]*B[0]+n.ravel()[:,None]*B[1]
    q2=(q**2).sum(1); ok=q2>0; q=q[ok]; q2=q2[ok]
    w=np.exp(-q2*sig**2/2)/q2/area
    return np.array([np.sum(w*np.cos(q@r)) for r in rs]), area
