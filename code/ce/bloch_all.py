import json,sys; import os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from multiprocessing import Pool
from bloch_lswt import energy; from periodic import Cell
R=[r for r in json.load(open('distinct.json')) if '6' not in r['hist']]
def f(r):
    E,m=energy(Cell(*r['cell']),[tuple(x) for x in r['tiling']],nk=16); return E
if __name__=='__main__':
    Es=[]
    with Pool(2) as p:
        for i,E in enumerate(p.imap(f,R,chunksize=20)):
            Es.append(E)
            if i%500==0: print(i,flush=True)
    for r,E in zip(R,Es): r['e_inf']=E
    json.dump(R,open('hubfree_lswt.json','w')); print('done',len(R))
