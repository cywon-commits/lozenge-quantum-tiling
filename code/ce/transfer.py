import json, numpy as np
from features import features
F = json.load(open('fit.json'))['+ pairs up to r2<=7']; cols = F['cols']; w = np.array(F['w'])
ce = lambda r: w[0] + features(r['cell'], r['tiling'])[cols] @ w[1:]
R = json.load(open('rank_result.json'))['candidates']
R = [r for r in R if '6' not in r['hist']]
d = np.array([ce(r) - r['e_inf'] for r in R])
print(f"48-site out-of-sample: n={len(R)}  rms {1e6*np.sqrt(np.mean(d**2)):.1f}e-6  max {1e6*np.abs(d).max():.1f}e-6  mean {1e6*d.mean():+.1f}e-6")
