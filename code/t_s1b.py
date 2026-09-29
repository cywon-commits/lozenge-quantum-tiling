import numpy as np
exec(open('test_sse_s.py').read().split('bb=np.array')[0])
for bb,N in ((np.array([[0,1]]),2),(np.array([[0,1],[1,2],[2,3],[3,0]]),4)):
    for beta in (0.5,2.0):
        Ee,Me=ed(bb,N,1.0,beta,0.0)
        E,M=run(bb,N,beta,S=1.0,h=0.0,ntherm=5000,nmeas=300000,seed_=5)
        nb=30; Eb=E.reshape(nb,-1).mean(1)
        print(N,beta,"ED",round(Ee,4),"QMC",round(Eb.mean(),4),"+-",round(Eb.std()/np.sqrt(nb),4))
