"""Corrected Gaussian (LSWT) half-torus entanglement: covariance matrices of the quadratures, zero modes (omega<wc) removed
consistently from <xx> and <pp>. Check: entropy of the whole system = 0."""
import sys, json, pickle, numpy as np
sys.path.insert(0,'/home/claude/w/lozenge-quantum-tiling/code')
from dice_string import Tiling
def cov(N,bonds,S=0.5,eps=1e-7,wc=1e-2):
    bonds=np.asarray(bonds); Adj=np.zeros((N,N)); Adj[bonds[:,0],bonds[:,1]]=1; Adj[bonds[:,1],bonds[:,0]]=1
    z=Adj.sum(1); A=S*np.diag(z)+eps*np.eye(N); B=S*Adj; Kx=A+B; Kp=A-B
    w,Q=np.linalg.eigh(Kx); Kxh=Q@np.diag(np.sqrt(w))@Q.T; Kxmh=Q@np.diag(1/np.sqrt(w))@Q.T
    w2,Q2=np.linalg.eigh(Kxh@Kp@Kxh); om=np.sqrt(np.clip(w2,1e-14,None)); k=om>wc
    X=0.5*Kxmh@(Q2[:,k]*om[k])@Q2[:,k].T@Kxmh; P=0.5*Kxh@(Q2[:,k]/om[k])@Q2[:,k].T@Kxh
    return X,P
def ent(X,P,r):
    nu=np.sqrt(np.clip(np.linalg.eigvals(X[np.ix_(r,r)]@P[np.ix_(r,r)]).real,0.25,None))
    return float(np.sum((nu+.5)*np.log(nu+.5)-np.where(nu>.5+1e-12,(nu-.5)*np.log(np.maximum(nu-.5,1e-300)),0)))
S=pickle.load(open('/home/claude/w/lozenge-quantum-tiling/data/paper/structs_L36.pkl','rb')); L=36; out={}
Mf=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
for key in ('square','M6','M3','M2','M23','dice'):
    T=Tiling(L); T.removed=set(S[key]); B=T.bonds(); X,P=cov(T.N,B)
    f=np.array([np.linalg.solve(Mf,x) for x in T.pos]); reg=np.where((f[:,1]%1)<0.5)[0]
    sh=ent(X,P,reg); out[key]=dict(S_half_per_length=sh/(2*L), S_all=ent(X,P,np.arange(T.N)) if key=='square' else None)
    print(key, out[key], flush=True)
json.dump(out,open('/home/claude/k5/ent_fix.json','w'),indent=1)
