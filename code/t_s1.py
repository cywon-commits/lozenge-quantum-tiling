import numpy as np
exec(open('test_sse_s.py').read().split('for S in')[0])
for beta,h in ((3.0,0.3),(1.0,0.5)):
    Ee,Me=ed(bb,N,1.0,beta,h)
    E,M=run(bb,N,beta,S=1.0,h=h,ntherm=5000,nmeas=400000,seed_=3)
    nb=40; Eb=E.reshape(nb,-1).mean(1); Mb=M.reshape(nb,-1).mean(1)
    print(beta,h,"ED",round(Ee,4),round(Me,4),"QMC",round(Eb.mean(),4),"+-",round(Eb.std()/np.sqrt(nb),4),round(Mb.mean(),4),"+-",round(Mb.std()/np.sqrt(nb),4))
