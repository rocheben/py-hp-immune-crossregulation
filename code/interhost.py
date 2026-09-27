# -*- coding: utf-8 -*-
"""
Between-host model (population scale) coupled to the within-host model.

Compartment structure, crossing of two statuses:
  Py:  S -> I -> R -> S   [SIRS]  immunity WANES at rate omega (R->S);
       omega=0 recovers the pure SIR (permanent immunity).
  Hp:  s (susceptible) -> i (infected) -> s              [SIS, chronicity=slow return]

Joint state (9 compartments), notation X_yz with y in {S,I,R} for Py,
z in {s,i} for Hp:
  SS Si  IS Ii  RS Ri  (+ the crossed S/I/R)  -> we enumerate the 6 useful
  combinations: SS, Ss? To stay readable we write:
    y ∈ {Ps=Py-susceptible, Pi=Py-infected, Pr=Py-immune}
    z ∈ {Hs=Hp-susceptible, Hi=Hp-infected}
  => 6 compartments: (Ps,Hs),(Ps,Hi),(Pi,Hs),(Pi,Hi),(Pr,Hs),(Pr,Hi)

Within->between coupling (factors ESTIMATED from the within-host model, without evolution):
  - Py transmissibility (instantaneous infectiousness) increased if also Hp-infected:
        effective β_Py *= phi_betaPy   in hosts (·,Hi)
  - Hp transmissibility increased if the host is also infected by Py:
        effective β_Hp *= phi_betaHp   in hosts (Pi,·)
  - Py infectious PERIOD PROLONGED under coinfection (the within-host data show a
    fatter/longer parasitemia tail in coinfection): the Py recovery rate is reduced,
        gamma_Py *= phi_gamma  (phi_gamma < 1)   in hosts (Pi,Hi).
    This lengthening of the Pi,Hi sojourn is what creates POSITIVE coinfection
    co-occurrence (without it, Py and Hp are acquired independently and prev_co is
    just prev_Py*prev_Hp). It replaces the earlier phi_alpha (extra mortality), which
    SHORTENED the coinfected period — the wrong sign relative to the data.
These phi are passed as arguments (computed in run_interhost from the ratios).

Demography: disease-induced Py deaths (alpha) are compensated by births into the
naive (Ps,Hs) class, so the total population N stays constant (proper endemic eq.).

Deworming: treatment rate tau applies i->s on Hp (worm removal).
"""
import numpy as np
from scipy.integrate import solve_ivp

# indices of the 6 compartments
PsHs,PsHi,PiHs,PiHi,PrHs,PrHi = range(6)
LABELS=["Ps,Hs","Ps,Hi","Pi,Hs","Pi,Hi","Pr,Hs","Pr,Hi"]

def deriv(t, x, p):
    Ps_Hs,Ps_Hi,Pi_Hs,Pi_Hi,Pr_Hs,Pr_Hi = x
    N = x.sum()
    # infecting prevalences
    I_Py = Pi_Hs + Pi_Hi                     # Py-infected hosts
    I_Hp = Ps_Hi + Pi_Hi + Pr_Hi             # Hp-infected hosts
    # forces of infection (transmission ∝ number of infected, weighted by transmissibility)
    # Py: coinfected hosts (Pi,Hi) transmit phi_betaPy times more
    lam_Py = p['betaPy']*(Pi_Hs + p['phi_betaPy']*Pi_Hi)/N
    # Hp: coinfected hosts (Pi,Hi) transmit phi_betaHp times more
    lam_Hp = p['betaHp']*(Ps_Hi + Pr_Hi + p['phi_betaHp']*Pi_Hi)/N

    g=p['gammaPy']; d=p['deltaHp']; tau=p['tau']; mu=p['mu']; aPy=p['alphaPy']
    w=p.get('omega',0.0)             # Py immunity WANING rate (R->S); w=0 -> pure SIR
    gC=g*p['phi_gamma']              # coinfection PROLONGS the Py infectious period (phi_gamma<1)
    births=mu*N + aPy*(Pi_Hs+Pi_Hi)  # replace natural AND disease deaths -> N constant

    dx=np.zeros(6)
    # (Ps,Hs)  <- all births enter here (naive) + waned immunes (Pr,Hs)
    dx[PsHs]= births - lam_Py*Ps_Hs - lam_Hp*Ps_Hs - mu*Ps_Hs \
              + (d+tau)*Ps_Hi + w*Pr_Hs
    # (Ps,Hi)  <- waned immunes (Pr,Hi)
    dx[PsHi]= lam_Hp*Ps_Hs - lam_Py*Ps_Hi - (d+tau)*Ps_Hi - mu*Ps_Hi + w*Pr_Hi
    # (Pi,Hs)
    dx[PiHs]= lam_Py*Ps_Hs - g*Pi_Hs - (mu+aPy)*Pi_Hs \
              - lam_Hp*Pi_Hs + (d+tau)*Pi_Hi
    # (Pi,Hi)  recovery slowed (gC) -> longer coinfected sojourn; same virulence aPy
    dx[PiHi]= lam_Py*Ps_Hi + lam_Hp*Pi_Hs - gC*Pi_Hi - (mu+aPy)*Pi_Hi \
              - (d+tau)*Pi_Hi
    # (Pr,Hs)  immunity wanes at w
    dx[PrHs]= g*Pi_Hs - lam_Hp*Pr_Hs - mu*Pr_Hs + (d+tau)*Pr_Hi - w*Pr_Hs
    # (Pr,Hi)
    dx[PrHi]= gC*Pi_Hi + lam_Hp*Pr_Hs - (d+tau)*Pr_Hi - mu*Pr_Hi - w*Pr_Hi
    return dx

def run(p, tmax=4000, x0=None):
    if x0 is None:
        N0=1.0
        x0=np.array([N0-2e-3, 1e-3, 1e-3, 0,0,0])
    sol=solve_ivp(deriv,[0,tmax],x0,args=(p,),method='LSODA',
                  rtol=1e-8,atol=1e-10,dense_output=True)
    # LSODA can occasionally return a non-finite tail on stiff runs (e.g. strong
    # deworming tau); fall back to a fully implicit stiff solver in that case.
    if not np.all(np.isfinite(sol.y[:,-1])):
        sol=solve_ivp(deriv,[0,tmax],x0,args=(p,),method='Radau',
                      rtol=1e-8,atol=1e-10,dense_output=True)
    return sol

def endemic(p, **kw):
    sol=run(p,**kw)
    xf=sol.y[:,-1]
    N=xf.sum()
    prev_Py=(xf[PiHs]+xf[PiHi])/N
    prev_Hp=(xf[PsHi]+xf[PiHi]+xf[PrHi])/N
    prev_co=xf[PiHi]/N
    return dict(prev_Py=prev_Py, prev_Hp=prev_Hp, prev_co=prev_co,
                immune_Py=(xf[PrHs]+xf[PrHi])/N, x=xf)

# ---- R0 for Py, with/without coinfection present ----
def R0_Py(p, hp_prev):
    """R0 of Py in a population where a fraction hp_prev is already infected by Hp.
    Mean transmission weighted by the coinfection over-factor, duration=1/(mu+gamma+alpha)."""
    # in Hp+ hosts, Py transmissibility is phi_betaPy and the infectious period is
    # LENGTHENED (slower recovery gamma*phi_gamma, phi_gamma<1)
    dur_single=1.0/(p['mu']+p['gammaPy']+p['alphaPy'])
    dur_co    =1.0/(p['mu']+p['gammaPy']*p['phi_gamma']+p['alphaPy'])
    beta_eff = p['betaPy']*((1-hp_prev)*dur_single + hp_prev*p['phi_betaPy']*dur_co)
    return beta_eff
