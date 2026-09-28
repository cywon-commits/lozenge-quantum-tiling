import json, pickle, numpy as np, time
from dice_string import Tiling
from tilt import strips
from periodic import Cell, replicate
from lswt import lswt_energy, signs
from lswt_fluct import fluct, bogoliubov
from tiling_moves import degrees
L=36
def frompkl(p): T=Tiling(L); T.removed=pickle.load(open(p,'rb')); assert not T.check_pairing(); return T
S={}
S['square']=strips(L,[[0,L]])[0]
O=json.load(open('periodic_all_lswt.json')); O=[o for o in O if abs(o['m_rep']-1/18)<1e-4]; O.sort(key=lambda o:o['e_lswt'])
for o in O:
    try: S['M3']=replicate(Cell(*o['cell']),[tuple(x) for x in o['tiling']],L); print('M3 cell',o['cell'],o['hist']); break
    except AssertionError: pass
S['M6']=frompkl('big_m36_L36.pkl'); S['M2']=frompkl('big_m12_L36.pkl'); S['M23']=frompkl('big_m9_L36.pkl')
S['dice']=strips(L,[])[0]
pickle.dump({k:set(T.removed) for k,T in S.items()},open('paper/structs_L36.pkl','wb'))
A1=np.array([1.0,0]); A2=np.array([0.5,np.sqrt(3)/2]); Bm=2*np.pi*np.linalg.inv(np.array([A1,A2])).T
Mf=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
res={}; sq={}
wg=np.linspace(0,2.4,241); sg=np.arange(-L,L+1)
for k,T in S.items():
    t=time.time(); B=T.bonds(); z=degrees(T)
    f=np.array([np.linalg.solve(Mf,x) for x in T.pos]); reg=np.where((f[:,1]%1)<0.5)[0]
    o=fluct(T.N,B,region=reg); E,Sl=lswt_energy(T.N,B)
    res[k]=dict(m=Sl/T.N,e_lswt=E/T.N,n=float(o['n'].mean()),s_site=float(o['s_site'].mean()),S_half=o['S_region']/(2*L),
               n_z={int(q):float(o['n'][z==q].mean()) for q in np.unique(z)},frac_z={int(q):float(np.mean(z==q)) for q in np.unique(z)})
    om,U,V,zz,keep=bogoliubov(T.N,B); om=om[keep]; C=U+V
    for name,bv in (('b1',Bm[0]),('b12',Bm[0]+Bm[1])):
        qp=np.outer(sg,bv/L); ph=np.exp(-1j*qp@T.pos.T); amp=np.abs(ph@C)**2*0.25/T.N
        eta=0.03; lor=eta/np.pi/((wg[:,None]-om[None,:])**2+eta**2); sq[f'{k}_{name}']=amp@lor.T
    print(k,res[k]['m'],res[k]['n'],res[k]['s_site'],f"{time.time()-t:.0f}s",flush=True)
json.dump(res,open('paper/fluct_structs.json','w'),indent=1)
np.savez_compressed('paper/sqw_structs.npz',wg=wg,sg=sg,**sq)
