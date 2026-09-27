"""Figure B1 (ESM) — Calibration of the evolutionary regime.
A: coinfection optimum of g3h over the k2 x s3 grid (kappa = 1.2, mu3 = 0.2, K2 = 300);
B: optimum of g3h against the cost kappa; C: fitness curves at the reference point.
Output: figures/FigureB1.png"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from evolution import base_theta, FIG
from deterministic import simulate_det
from models import build_coinfection, build_hp_single
b=base_theta(); tg=np.linspace(0,30,31); gg=np.linspace(0,0.5,51)
def fit(pr,g3h,kappa,coinf=True):
    th=dict(b); th.update(pr); th['g3h']=g3h; th['r2']=b['r2']*(1-kappa*g3h)
    if th['r2']<=0: return 0.0
    builder=build_coinfection if coinf else build_hp_single
    idx=1 if coinf else 0
    y=simulate_det(builder,th,tg)
    return 0.0 if np.any(~np.isfinite(y)) else float(np.trapezoid(np.maximum(y[:,idx],0),tg))
def ess(pr,kappa,coinf):
    w=np.array([fit(pr,g,kappa,coinf) for g in gg]); return gg[w.argmax()]
fig,axes=plt.subplots(1,3,figsize=(15,4.4))
k2s=np.array([4,6,8,12,20]); s3s=np.array([1,3,6,10,20]); Z=np.zeros((len(k2s),len(s3s)))
for i,k2 in enumerate(k2s):
    for j,s3 in enumerate(s3s): Z[i,j]=ess({'k2':k2,'s3':s3,'mu3':0.2,'K2':300},1.2,True)
im=axes[0].imshow(Z,origin='lower',aspect='auto',cmap='viridis',extent=[0,len(s3s),0,len(k2s)])
axes[0].set_xticks(np.arange(len(s3s))+0.5); axes[0].set_xticklabels(s3s)
axes[0].set_yticks(np.arange(len(k2s))+0.5); axes[0].set_yticklabels(k2s)
axes[0].set_xlabel(r"$s_3$ (regulatory damping efficacy)"); axes[0].set_ylabel(r"$k_2$ (anti-worm killing)")
for i in range(len(k2s)):
    for j in range(len(s3s)):
        axes[0].text(j+.5,i+.5,"%.2f"%Z[i,j],ha='center',va='center',color='white' if Z[i,j]<0.04 else 'black',fontsize=8)
plt.colorbar(im,ax=axes[0],label=r"optimum $g_{3h}$"); axes[0].plot(2.5,1.5,'r*',ms=18,mec='k')
REF={'k2':6.0,'s3':6.0,'mu3':0.2,'K2':300.0}
kaps=np.array([0.6,0.9,1.2,1.5,2.0,2.4,3.0])
ec=[ess(REF,k,True) for k in kaps]; es=[ess(REF,k,False) for k in kaps]
axes[1].plot(kaps,es,'C0-o',label="worm alone"); axes[1].plot(kaps,ec,'C3-s',label="coinfection")
axes[1].fill_between(kaps,ec,es,color='C0',alpha=.12); axes[1].axvline(1.2,color='k',ls=':',lw=1,label=r"reference $\kappa$ = 1.2")
axes[1].set_xlabel(r"$\kappa$ (cost of immunosuppression)"); axes[1].set_ylabel(r"optimum $g_{3h}$")
axes[1].legend(); axes[1].grid(alpha=.3)
wc=np.array([fit(REF,g,1.2,True) for g in gg]); ws=np.array([fit(REF,g,1.2,False) for g in gg])
axes[2].plot(gg,ws/ws.max(),'C0-',label="worm alone (opt=%.3f)"%gg[ws.argmax()])
axes[2].plot(gg,wc/wc.max(),'C3-',label="coinfection (opt=%.3f)"%gg[wc.argmax()])
axes[2].axvline(gg[ws.argmax()],color='C0',ls=':',alpha=.6); axes[2].axvline(gg[wc.argmax()],color='C3',ls=':',alpha=.6)
axes[2].set_xlabel(r"$g_{3h}$ (worm immunosuppression)"); axes[2].set_ylabel("relative fitness")
axes[2].legend(); axes[2].grid(alpha=.3)
for _a,_lt in zip(axes,"ABC"):
    _a.text(0.01,1.02,_lt,transform=_a.transAxes,fontsize=12,fontweight="bold")
fig.tight_layout(); p=os.path.join(FIG,"FigureB1.png"); fig.savefig(p,dpi=130); print("->",p)
print("ref: SI=%.3f COI=%.3f (kappa=1.2)"%(gg[ws.argmax()],gg[wc.argmax()]))
print("cost sweep COI:",[round(float(x),3) for x in ec]); print("cost sweep SI :",[round(float(x),3) for x in es])
