import json, sys
from periodic import Cell, enumerate_tilings, analyse
cells=[(6,6,s) for s in range(6)]+[(4,9,s) for s in range(4)]
res=[]; KEEP=4
for (p,q,s) in cells:
    C=Cell(p,q,s); tl=enumerate_tilings(C); best={}; n=0
    for t in tl:
        a=analyse(C,t)
        if a is None: continue
        n+=1; k=round(a['m'],6); lst=best.setdefault(k,[])
        if len(lst)<KEEP or a['e_model']<lst[-1][0]:
            lst.append((a['e_model'],t,a)); lst.sort(key=lambda x:x[0]); del lst[KEEP:]
    for k,lst in best.items():
        for em,t,a in lst: res.append(dict(cell=[p,q,s],tiling=[list(x) for x in t],m=a['m'],e_model=em,hist=a['hist']))
    print(f"cell {p}x{q} s={s}: {len(tl)} tilings (cap 800k), {n} bipartite, distinct m {len(best)}",flush=True)
    del tl; json.dump(res,open('periodic_part2.json','w'))
