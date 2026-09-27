# -*- coding: utf-8 -*-
"""
Epidemiological consequences (population scale) of the interactions
estimated on data. Everything is in relative units:
 - time is normalized by the mean Py infection duration (1/(mu+gammaPy+alphaPy)),
 - transmission is parameterized by BASE (single-infection) R0 values that are swept,
 - only the 3 coupling factors phi carry an ESTIMATED value (fits/phi_interhost.json).
No absolute base rate is set to a made-up number.
"""
import numpy as np, json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import interhost as ih

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = os.path.join(HERE, "fits"); FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

def derive_phi(tmax=45.0, thr=0.01, save=True):
    """Derives the inter-host coupling factors from the intra-host RATIOS
    (coinfected / single) of the FITTED model — no made-up number.

    The onward Py transmission per infection ∝ cumulative parasitemia (∫). This
    total = (instantaneous infectiousness) × (infectious period). We DECOMPOSE it so
    the duration part can drive coinfection co-occurrence at the population scale:
      Cr = ∫para_c / ∫para_s                      (total transmission ratio)
      Dr = period_c / period_s                    (infectious-period ratio, >1: prolonged)
           period = span where parasitemia > thr (transmission-relevant density)
      phi_gamma  = 1/Dr   (<1)  -> recovery rate slowed, Pi,Hi sojourn lengthened
      phi_betaPy = Cr/Dr        -> instantaneous infectiousness (so β·period = Cr preserved)
      phi_betaHp = P2c[-1]/P2s[-1]   Hp transmission ∝ eggs ∝ worm load (plateau)
    (phi_gamma REPLACES the old phi_alpha: prolonging the period, not adding mortality.)
    """
    import estimate as e
    from deterministic import simulate_det
    from models import build_coinfection
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    single = json.load(open(os.path.join(FITS, "single.json")))
    worm   = json.load(open(os.path.join(FITS, "worm.json")))
    coinf  = json.load(open(os.path.join(FITS, "coinf.json")))
    tt = np.linspace(0, tmax, int(tmax * 10) + 1)
    P1s, Rs = e._sim_single(single, tt); para_s = e.frac(P1s, Rs)
    thc = e._theta_worm_ss(dict(single) | dict(e.WORM_FIXED) | dict(worm) | dict(coinf))
    y = simulate_det(build_coinfection, thc, tt)
    P1c, P2c, Rc = y[:, 0], y[:, 1], y[:, 2]; para_c = e.frac(P1c, Rc)
    P2s = e._sim_hp(thc, tt)
    def period(par):
        m = par > thr
        return float(tt[m].max() - tt[m].min()) if m.any() else 0.0
    Cr = float(trapz(para_c, tt) / trapz(para_s, tt))
    Dr = period(para_c) / period(para_s)
    phi = dict(phi_betaPy=Cr / Dr,
               phi_gamma =1.0 / Dr,
               phi_betaHp=float(P2c[-1] / P2s[-1]))
    if save:
        json.dump(phi, open(os.path.join(FITS, "phi_interhost.json"), "w"), indent=2)
    return phi

_phi_path = os.path.join(FITS, "phi_interhost.json")
phi = json.load(open(_phi_path)) if os.path.exists(_phi_path) else derive_phi()
PHI_BPY=phi['phi_betaPy']; PHI_BHP=phi['phi_betaHp']; PHI_G=phi['phi_gamma']

# --- relative units: we fix the time scale by the Py infection duration ---
# We work with dimensionless rates referred to (mu+gammaPy+alphaPy)=1.
# Choice of internal partition of this duration (demography/recovery/mortality) in
# proportions, WITHOUT absolute value: gamma dominates (recovery), mu and alpha small.
BASE = dict(
    mu=0.02, gammaPy=0.93, alphaPy=0.05,   # sum=1.0 (time unit = Py infection duration)
    deltaHp=0.10, tau=0.0,                 # chronic Hp: slow loss; tau=deworming
    omega=0.10,                            # Py immunity WANING rate (SIRS): 1/omega=10
                                           # infection-durations ~8 months (mouse). FREE
                                           # epidemiological param -> swept below; omega=0=SIR.
    phi_betaPy=PHI_BPY, phi_betaHp=PHI_BHP, phi_gamma=PHI_G,
)
def params_from_R0(R0Py_base, R0Hp_base, tau=0.0):
    p=dict(BASE); p['tau']=tau
    # R0_base single = beta*duration ; we invert to obtain beta.
    # IMPORTANT: transmissibility beta is a biological constant -> it must be pinned
    # to the UNTREATED (tau=0) duration. Deworming (tau>0) then acts by SHORTENING
    # the effective Hp infection (mu+deltaHp+tau), lowering effective R0_Hp below
    # R0Hp_base. (Folding tau into dur_Hp here would rescale beta and exactly cancel
    # the treatment -> Hp prevalence invariant to tau, which is wrong.)
    dur_Py=1.0/(p['mu']+p['gammaPy']+p['alphaPy'])
    dur_Hp0=1.0/(p['mu']+p['deltaHp'])            # baseline (tau=0) duration
    p['betaPy']=R0Py_base/dur_Py
    p['betaHp']=R0Hp_base/dur_Hp0
    return p

# ================= ANALYSIS 1: R0 and invasion thresholds =================
def analysis_R0():
    R0Py=np.linspace(0.5,3.0,26); R0Hp=np.linspace(0.5,3.0,26)
    # Py invasion threshold: effective R0_Py >1. Depends on the established Hp prevalence.
    # For each (R0Py_base,R0Hp_base), we compute the endemic Hp prevalence then
    # the effective R0_Py in the presence of this prevalence.
    Zpy=np.zeros((len(R0Hp),len(R0Py)))
    Zco=np.zeros((len(R0Hp),len(R0Py)))
    for i,rh in enumerate(R0Hp):
        for j,rp in enumerate(R0Py):
            p=params_from_R0(rp,rh)
            e=ih.endemic(p,tmax=6000)
            hp_prev=e['prev_Hp']
            Zpy[i,j]=ih.R0_Py(p,hp_prev)            # effective R0 Py
            Zco[i,j]=e['prev_co']                    # coinfection prevalence
    return R0Py,R0Hp,Zpy,Zco

# ============= ANALYSIS 2: deworming (Hp removal) =============
def analysis_deworming():
    taus=np.linspace(0.0,2.0,21)                    # deworming intensity (relative)
    scenarios={'Py low (R0=1.3)':1.3,'Py medium (R0=1.8)':1.8,'Py high (R0=2.5)':2.5}
    R0Hp_base=2.0                                    # established endemic Hp
    out={}
    for name,rp in scenarios.items():
        pv_py=[]; pv_co=[]; pv_hp=[]
        for tau in taus:
            p=params_from_R0(rp,R0Hp_base,tau=tau)
            e=ih.endemic(p,tmax=6000)
            pv_py.append(e['prev_Py']); pv_co.append(e['prev_co']); pv_hp.append(e['prev_Hp'])
        out[name]=(np.array(pv_py),np.array(pv_co),np.array(pv_hp))
    return taus,out

# ================= FIGURE 7 =================
def figure7():
    """Figure 7: effective R0 of Py (A), coinfected fraction (B), deworming (C)."""
    print("Estimated coupling factors: phi_betaPy=%.3f phi_betaHp=%.3f phi_gamma=%.3f"
          %(PHI_BPY,PHI_BHP,PHI_G))
    R0Py,R0Hp,Zpy,Zco=analysis_R0()
    taus,dw=analysis_deworming()

    fig,ax=plt.subplots(1,3,figsize=(16,4.6))
    # Panel A: effective R0_Py, threshold=1 line
    im=ax[0].contourf(R0Py,R0Hp,Zpy,levels=20,cmap='RdBu_r',vmin=Zpy.min(),vmax=Zpy.max())
    cs=ax[0].contour(R0Py,R0Hp,Zpy,levels=[1.0],colors='k',linewidths=2)
    ax[0].clabel(cs,fmt='threshold = 1')
    ax[0].set_xlabel(r"base $R_0$ of Py (single infection)")
    ax[0].set_ylabel(r"base $R_0$ of Hp (single infection)")
    plt.colorbar(im,ax=ax[0],label=r"effective $R_0$ of Py")
    # Panel B: coinfection prevalence
    im2=ax[1].contourf(R0Py,R0Hp,Zco,levels=20,cmap='viridis')
    ax[1].set_xlabel(r"base $R_0$ of Py (single infection)")
    ax[1].set_ylabel(r"base $R_0$ of Hp (single infection)")
    plt.colorbar(im2,ax=ax[1],label="coinfected fraction")
    # Panel C: deworming vs Py prevalence
    for name,(py,co,hp) in dw.items():
        lab=name.replace('Py ','').replace('(R0=',r'(base $R_0$ = ')
        ax[2].plot(taus,py/py[0],'-o',ms=3,label=lab)
    ax[2].set_xlabel(r"deworming intensity $\tau$ (relative)")
    ax[2].set_ylabel(r"Py prevalence (relative to $\tau$ = 0)")
    ax[2].axhline(1.0,color='gray',ls=':'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    for a,lt in zip(ax,"ABC"):
        a.text(0.01,1.02,lt,transform=a.transAxes,fontsize=12,fontweight="bold")
    fig.tight_layout(); _p=os.path.join(FIG,"Figure7.png"); fig.savefig(_p,dpi=130)
    plt.close(fig); print("saved",_p)

    # key figures for the text
    print("\n--- Deworming: relative change in Py prevalence between τ=0 and τ=2 ---")
    for name,(py,co,hp) in dw.items():
        print("  %s: prev Py %.3f -> %.3f  (%+.0f%%)"%(name,py[0],py[-1],100*(py[-1]/py[0]-1)))
    print("\n--- Py invasion threshold: minimal R0Py_base for R0_eff>1 as a function of R0Hp ---")
    for i,rh in [(0,'R0Hp=0.5'),(12,'R0Hp~1.7'),(25,'R0Hp=3.0')]:
        col=Zpy[i,:]
        idx=np.argmax(col>=1.0)
        print("  %s: R0Py_base threshold ≈ %.2f"%(rh,R0Py[idx]))
    return _p

def sirs_sensitivity():
    """Numbers quoted in the text: Py immunity waning omega (R0Py=1.8, R0Hp=2.0)."""
    print("\n--- SIRS sensitivity: Py immunity waning omega (R0Py=1.8, R0Hp=2.0) ---")
    print("  omega=0 -> pure SIR (permanent immunity). time unit = Py infection duration.")
    for w in (0.0, 0.05, 0.10, 0.20, 0.50):
        p=params_from_R0(1.8,2.0); p['omega']=w
        e=ih.endemic(p,tmax=10000)
        life="inf" if w==0 else "%.0f"%(1.0/w)
        print("  omega=%.2f (immunity ~%s durations): prev_Py=%.3f  prev_CO=%.4f"
              %(w,life,e['prev_Py'],e['prev_co']))

if __name__=="__main__":
    figure7()
    sirs_sensitivity()
