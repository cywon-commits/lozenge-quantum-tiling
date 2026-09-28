import numpy as np, random
from collections import Counter
from sse_s import wvert
S2=2; C=1.25
# dimer, 2 diag/offdiag ops on the same bond. legs: vertex p: 0 lo_i,1 lo_j,2 up_i,3 up_j
# links: site i: up_i(p0)=2 <-> lo_i(p1)=4 ; up_i(p1)=6 <-> lo_i(p0)=0 ; site j: 3<->5, 7<->1
link={2:4,4:2,6:0,0:6,3:5,5:3,7:1,1:7}
def W(leg,p): b=4*p; return wvert(leg[b],leg[b+1],leg[b+2],leg[b+3],S2,C,0.,0.)
leg=[1,1,1,1, 1,1,1,1]
random.seed(1); cnt=Counter()
for t in range(200000):
    for _ in range(3):
        j0=random.randrange(8); d0=random.choice((1,-1))
        if not (0<=leg[j0]+d0<=S2): continue
        j,d=j0,d0
        while True:
            p=j//4; l=j%4; base=4*p; leg[base+l]+=d
            dx=[ -d if (x==l or x//2==l//2) else d for x in range(4)]
            pr=[]
            for x in range(4):
                leg[base+x]+=dx[x]; pr.append(W(leg,p)); leg[base+x]-=dx[x]
            r=random.random()*sum(pr); x=0; acc=pr[0]
            while acc<r: x+=1; acc+=pr[x]
            leg[base+x]+=dx[x]; jn=link[base+x]; dn=dx[x]
            if (jn==j0 and dn==d0) or (base+x==j0 and dn==-d0): break
            j,d=jn,dn
    key=(leg[0],leg[1],leg[2]-leg[0],leg[6]-leg[4]); cnt[key]+=1
tot=sum(cnt.values()); Z=0; Wd={}
for n0 in range(3):
  for n1 in range(3):
    for d1 in (0,1,-1):
        m=(n0+d1,n1-d1)
        w=wvert(n0,n1,m[0],m[1],S2,C,0.,0.)*wvert(m[0],m[1],n0,n1,S2,C,0.,0.)
        if w>0: Wd[(n0,n1,d1,-d1)]=w
Z=sum(Wd.values())
for k in sorted(Wd): print(k, round(Wd[k]/Z,4), round(cnt[k]/tot,4))
