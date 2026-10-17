import json, numpy as np
from features import NAMES, SHELLS, CLASSES
R = json.load(open('hubfree_lswt.json')); X = np.load('X.npy'); y = np.array([r['e_inf'] for r in R])
m = X[:, 1]; nd = X[:, 0]
rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(y))
def cv(A, lam=1e-9):
    A1 = np.c_[np.ones(len(y)), A]; pred = np.zeros(len(y))
    for k in range(5):
        tr = fold != k
        w = np.linalg.solve(A1[tr].T @ A1[tr] + lam * np.eye(A1.shape[1]), A1[tr].T @ y[tr]); pred[~tr] = A1[~tr] @ w
    w = np.linalg.solve(A1.T @ A1 + lam * np.eye(A1.shape[1]), A1.T @ y)
    return pred, w
def rms(x): return np.sqrt(np.mean(x ** 2))
out = {}
print(f"N structures {len(y)}, spread of e: {1e3*y.std():.3f}e-3 J")
for name, cols in [('coordination (n_def, m)', [0, 1])] + \
        [(f'+ pairs up to r2<={s}', [0, 1] + [2 + 4 * SHELLS.index(t) + c for t in SHELLS if t <= s for c in range(4)] + [2 + 4 * len(SHELLS), 3 + 4 * len(SHELLS)])
         for s in SHELLS]:
    p, w = cv(X[:, cols]); r = y - p
    # lowest structures per m (near the hull) -> the ones that matter
    low = np.zeros(len(y), bool)
    for mm in np.unique(np.round(m, 6)):
        idx = np.where(np.round(m, 6) == mm)[0]; low[idx[np.argsort(y[idx])[:20]]] = True
    print(f"{name:28s} CV rms {1e6*rms(r):7.2f}e-6  max {1e6*np.abs(r).max():7.1f}e-6  | low-20/m rms {1e6*rms(r[low]):6.2f}e-6   nfeat {len(cols)}")
    out[name] = dict(cols=cols, w=w.tolist(), cv_rms=rms(r), cv_rms_low=rms(r[low]))
json.dump(out, open('fit.json', 'w'), indent=1)
