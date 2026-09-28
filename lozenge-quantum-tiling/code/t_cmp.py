import numpy as np
from sse_h import run as run_h
import importlib.util,sys
exec(open('test_sse_s.py').read().split('bb=np.array')[0])
bb=np.array([[0,1],[1,2],[2,3],[3,0],[0,4],[4,5],[5,2]]); N=6
Ee,Me=ed(bb,N,0.5,3.0,0.3)
sg,mz,E=run_h(bb,N,3.0,2000,50000,1,h=0.3)
print("ED",Ee,Me,"old SSE",E.mean(),mz.mean())
