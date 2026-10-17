import json, numpy as np
from features import NAMES, SHELLS, CLASSES
R=json.load(open('hubfree_lswt.json')); X=np.load('X.npy'); y=np.array([r['e_inf'] for r in R])
rng=np.random.default_rng(0); fold=rng.integers(0,5,len(y))
def cv(A):
    A1=np.c_[np.ones(len(y)),A]; p=np.zeros(len(y))
    for k in range(5):
        t=fold!=k; w=np.linalg.lstsq(A1[t],y[t],rcond=None)[0]; p[~t]=A1[~t]@w
    w=np.linalg.lstsq(A1,y,rcond=None)[0]; return np.sqrt(np.mean((y-p)**2)), w
# re-express shell features in the (1, kk', ee', kk'ee') basis: pair sums of 1, k k', eta eta', sigma sigma'
sh=[s for s in SHELLS if s<=7]
def basis(X):
    out=[]
    for s in sh:
        b=2+4*SHELLS.index(s); c=X[:,b:b+4]   # classes (+,+),(+,-),(-,+),(-,-) in (kk, ee)
        kk=np.array([1,1,-1,-1]); ee=np.array([1,-1,1,-1])
        out += [c.sum(1), c@kk, c@ee, c@(kk*ee)]
    return np.array(out).T
Bf=basis(X); nnb=X[:,[2+4*len(SHELLS),3+4*len(SHELLS)]]
full=np.c_[X[:,[0,1]],Bf,nnb]
r,w=cv(full); print('full r2<=7 in new basis: CV rms %.2fe-6'%(1e6*r))
lab=['n_def','m']+[f'{q}(r2={s})' for s in sh for q in ('N','KK','EE','SS')]+['nnb++','nnb+-']
for l,v in zip(lab,w[1:]): print(f'{l:12s} {1e3*v:+8.3f}e-3')
# compact models
for name,cols in [('KK only',[i for i,l in enumerate(lab) if l.startswith('KK')]),
                  ('KK+SS',[i for i,l in enumerate(lab) if l[:2] in ('KK','SS')]),
                  ('KK+SS+N',[i for i,l in enumerate(lab) if l[:2] in ('KK','SS','N(')]),
                  ('KK+SS+N+EE',[i for i,l in enumerate(lab) if l[:2] in ('KK','SS','N(','EE')])]:
    r,w2=cv(np.c_[X[:,[0,1]],Bf[:,[c-2 for c in cols]]]); print(f'{name:12s} CV rms {1e6*r:6.2f}e-6', [round(1e3*v,3) for v in w2[1:]])
