import numpy as np, pickle, sys
C=pickle.load(open('u_classical.pkl','rb'))
def analyze(key):
    D=pickle.load(open(f'u_{key}.pkl','rb')); out=D['out']; I=C[key]['I']; sig=C[key]['sig']
    szs=sorted(out); E=np.array([out[s]['E'][0] for s in szs])
    rows=[]
    for s in szs:
        mz=np.array(out[s]['mz']); a=sig*mz
        rows.append((s,out[s]['E'][0],a[I].mean(),np.delete(a,I).mean()))
    hs=np.linspace(0,3,601); G=E[None,:]-hs[:,None]*np.array(szs)[None,:]; gi=G.argmin(1)
    steps=[(hs[k],szs[gi[k-1]],szs[gi[k]]) for k in range(1,len(hs)) if gi[k]!=gi[k-1]]
    return rows,steps,hs,np.array(szs)[gi],np.array([rows[i][2] for i in gi])
if __name__=="__main__":
    for key in sys.argv[1:]:
        rows,steps,_,_,_=analyze(key)
        print(key); [print(f"  Sz={s:4.1f} E={e:9.4f}  a_inside={ai:+.3f}  a_outside={ao:+.3f}") for s,e,ai,ao in rows if s<9]
        print("  steps (h, from, to):",[(round(h,3),a,b) for h,a,b in steps])
