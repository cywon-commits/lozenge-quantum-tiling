"""LSWT data for the crystals added by the unrestricted (D6-class) enumeration: 1/7, 1/5, 1/4 and the hub-free 3/5 crystal.
Fluctuations, corrected half-torus entanglement (covariance method), local moments, S^xx(q,w) at h=0 and in a field (3/5 only)."""
import sys, json, pickle, numpy as np, time
sys.path.insert(0,'/home/claude/w/lozenge-quantum-tiling/code'); sys.path.insert(0,'/home/claude/rw'); sys.path.insert(0,'/home/claude/k5')
from tocell import to_tiling
from lswt import lswt_energy, signs
from lswt_fluct import bogoliubov
from tiling_moves import degrees
import importlib.util
src=open('/home/claude/k5/ent_fix.py').read().split("S=pickle.load")[0]; exec(src)   # cov(), ent()
D='/mnt/user-data/outputs/.qdec/'
todo=[('M19','x0p1111_cell',27),('M111','x0p0909_cell',33),('M17','x0p1429_cell',42),('M15','x0p2_cell',30),('M14','x0p25_cell',24),('M35','m35_cell',60)]
only=sys.argv[1:] or [k for k,_,_ in todo]
A1=np.array([1.0,0]); A2=np.array([0.5,np.sqrt(3)/2]); Bm=2*np.pi*np.linalg.inv(np.array([A1,A2])).T
wg=np.linspace(0,2.4,241); eta=0.03; Sp=0.5
lor=lambda om: eta/np.pi/((wg[:,None]-om[None,:])**2+eta**2)
for k,f,L in todo:
    if k not in only: continue
    t0=time.time(); r=json.load(open(D+f+'.json')); T=to_tiling(r['cell'],r['tiling'],L)
    pickle.dump(dict(L=L,removed=set(T.removed),cell=r['cell']),open(f'/home/claude/rw/struct_{k}.pkl','wb'))
    B=np.array(T.bonds()); z=degrees(T); N=T.N
    E,Sl=lswt_energy(N,B)
    om,U,V,_,keep=bogoliubov(N,B); n=(V**2).sum(1)
    sent=(n+1)*np.log(n+1)-np.where(n>0,n*np.log(np.where(n>0,n,1)),0)
    sg=np.array(signs(N,B)); sg=sg if sg.sum()>0 else -sg; sz=sg*(0.5-n)
    X,P=cov(N,B); Mf=np.array([[1,0.5],[0,np.sqrt(3)/2]])*L
    fr=np.array([np.linalg.solve(Mf,x) for x in T.pos]); reg=np.where((fr[:,1]%1)<0.5)[0]
    Sh=ent(X,P,reg)/(2*L)
    res=dict(L=L,cell=r['cell'],m=Sl/N,e_lswt=E/N,n=float(n.mean()),s_site=float(sent.mean()),S_half=Sh,
             n_z={int(q):float(n[z==q].mean()) for q in np.unique(z)},frac_z={int(q):float(np.mean(z==q)) for q in np.unique(z)},
             sz=sz.tolist(),z=z.tolist(),sg=sg.tolist(),
             classes={f'z{q}{"+" if s>0 else "-"}':round(float(sz[(z==q)&(sg==s)].mean()),4) for q in np.unique(z) for s in (1,-1) if ((z==q)&(sg==s)).any()})
    if k=='M35':
        sgr=np.arange(-L,L+1); qp=np.outer(sgr,Bm[0]/L); ph=np.exp(-1j*qp@T.pos.T)
        omk=om[keep] if len(om)!=U.shape[1] else om
        C=U+V; amp=np.abs(ph@C)**2*0.25/N; sq={'sg':sgr,'b1':amp@lor(omk).T}
        dSz=((U**2-V**2)*(-sg)[:,None]).sum(0); up=sg>0; h=0.22
        Cp=np.where(up[:,None],V,U); Cm=np.where(up[:,None],U,V)
        for ch,Cc,shift,sel in (('p',Cp,-h,dSz>0.5),('m',Cm,+h,dSz<-0.5)):
            sq[ch]=(np.abs(ph@Cc[:,sel])**2*(Sp/2)/N)@lor(omk[sel]+shift).T
        res['gap_plus']=float(omk[dSz>0.5].min()); res['h_field']=h
        np.savez_compressed('/home/claude/rw/sqw_M35.npz',**sq)
    json.dump(res,open(f'/home/claude/rw/lswt_{k}.json','w'))
    print(k,L,N,'m',round(res['m'],5),'e',round(res['e_lswt'],6),'n',round(res['n'],4),'s1',round(res['s_site'],4),'Sh',round(Sh,4),res['classes'],res.get('gap_plus'),f'{time.time()-t0:.0f}s',flush=True)
