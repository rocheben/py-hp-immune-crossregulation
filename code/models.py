"""
Within-host models: reaction definitions.

Two structural features are essential for
reproducing the end of the observed kinetics (decline of parasitemia + rebound
of RBC around day 21):

  (1) Compensatory erythropoiesis  :  a_R_prod = rho*(Rmax - R)
      The old logistic form rho*R*(1-R/Rmax) vanishes when R->0: the
      RBC could never rebound after anemia. The compensatory form
      (production stimulated by the deficit, EPO-like — cf. epo_summary.csv) is
      nonzero at low R and allows the observed rebound.

  (2) ADAPTIVE immunity (clonal expansion): a_I_prod = b*I*P/(Kc+P)
      The old production b*P slaved the effector to the burden (no
      memory, no resolution). Clonal expansion + slow decay
      (small mu) makes immunity peak AFTER the parasite -> delayed
      clearance -> resolution of the infection, hence rebound of the RBC.

Each model is returned as a list of (propensity, stoichiometry) pairs. This single
formulation defines both an exact stochastic jump process and its mean-field limit;
all results in the manuscript are obtained from the mean-field limit, integrated by
deterministic.simulate_det.
"""
import numpy as np

# ---------- MODEL 1 : Py single (P1, R=RBC, I1, I3) ----------
def build_py_single(theta):
    x0 = np.array([theta['P1_0'], theta['R0'], theta['I1_0'], theta['I3_0']], float)
    Kc = theta.get('Kc', 50.0)
    def a_repl(x,th,t):        return th['r1']*x[0]*x[1]/(th['Rh']+x[1])
    def a_clear_immune(x,th,t):return th['k1']*x[2]*x[0]/(1.0+th['s3']*x[3])
    def a_clear_bg(x,th,t):    return th['d1']*x[0]
    def a_R_prod(x,th,t):      return th['rho']*(th['Rmax']-x[1])            # (1)
    def a_R_cons(x,th,t):      return th['c1']*x[0]*x[1]/(th['Rh']+x[1])
    def a_I1_prod(x,th,t):     return th['b1']*x[2]*x[0]/(Kc+x[0])/(1.0+th['s3']*x[3])  # (2)
    def a_I1_decay(x,th,t):    return th['mu1']*x[2]
    def a_I3_prod(x,th,t):     return th['g3']*x[2]
    def a_I3_decay(x,th,t):    return th['mu3']*x[3]
    reactions = [
        (a_repl,          np.array([+1,0,0,0])),
        (a_clear_immune,  np.array([-1,0,0,0])),
        (a_clear_bg,      np.array([-1,0,0,0])),
        (a_R_prod,        np.array([0,+1,0,0])),
        (a_R_cons,        np.array([0,-1,0,0])),
        (a_I1_prod,       np.array([0,0,+1,0])),
        (a_I1_decay,      np.array([0,0,-1,0])),
        (a_I3_prod,       np.array([0,0,0,+1])),
        (a_I3_decay,      np.array([0,0,0,-1])),
    ]
    return x0, reactions

# ---------- MODEL 2 : Hp single (P2, I2, I3) — chronic burden ----------
# H. polygyrus = chronic nematode: self-limited LOGISTIC growth
# (capacity K2) -> stable burden, consistent with the flat lfec plateau observed.
# No RBC resource for the worm (unlike Py).
def build_hp_single(theta):
    x0 = np.array([theta['P2_0'], theta['I2_0'], theta['I3_0']], float)
    Kc = theta.get('Kc', 50.0)
    def a_repl(x,th,t):        return th['r2']*x[0]*(1.0 - x[0]/th['K2'])         # logistic
    def a_clear_immune(x,th,t):return th['k2']*x[1]*x[0]/(1.0+th['s3']*x[2])
    def a_clear_bg(x,th,t):    return th['d2']*x[0]
    def a_I2_prod(x,th,t):     return th['b2']*x[1]*x[0]/(Kc+x[0])/(1.0+th['s3']*x[2])  # adaptive
    def a_I2_decay(x,th,t):    return th['mu2']*x[1]
    Kh = theta.get('Kh', 80.0)
    def a_I3_prod(x,th,t):     return th['g3h']*x[0]/(Kh+x[0]) + th['g3']*x[1]  # worm -> I3 (saturated)
    def a_I3_decay(x,th,t):    return th['mu3']*x[2]
    reactions = [
        (a_repl,          np.array([+1,0,0])),
        (a_clear_immune,  np.array([-1,0,0])),
        (a_clear_bg,      np.array([-1,0,0])),
        (a_I2_prod,       np.array([0,+1,0])),
        (a_I2_decay,      np.array([0,-1,0])),
        (a_I3_prod,       np.array([0,0,+1])),
        (a_I3_decay,      np.array([0,0,-1])),
    ]
    return x0, reactions

# ---------- MODEL 3 : Coinfection (P1,P2,R,I1,I2,I3) ----------
def build_coinfection(theta):
    x0 = np.array([theta['P1_0'], theta['P2_0'], theta['R0'],
                   theta['I1_0'], theta['I2_0'], theta['I3_0']], float)
    Kc = theta.get('Kc', 50.0); Kh = theta.get('Kh', 80.0)
    s12 = theta.get('s12', 0.0)   # I1 (Th1 anti-Py) suppresses I2 (Th2 anti-worm)
    s21 = theta.get('s21', 0.0)   # I2 (Th2) suppresses I1 (Th1)  -> Th1/Th2 antagonism
    def a_P1_repl(x,th,t):  return th['r1']*x[0]*x[2]/(th['Rh']+x[2])
    def a_P1_imm(x,th,t):   return th['k1']*x[3]*x[0]/(1.0+th['s3']*x[5])
    def a_P1_bg(x,th,t):    return th['d1']*x[0]
    # worm: logistic growth (K2). NO competition (Hp data = facilitation).
    def a_P2_repl(x,th,t):  return th['r2']*x[1]*(1.0 - x[1]/th['K2'])
    def a_P2_imm(x,th,t):   return th['k2']*x[4]*x[1]/(1.0+th['s3']*x[5])
    def a_P2_bg(x,th,t):    return th['d2']*x[1]
    def a_R_prod(x,th,t):   return th['rho']*(th['Rmax']-x[2])                       # (1)
    def a_R_cons(x,th,t):   return th['c1']*x[0]*x[2]/(th['Rh']+x[2])
    # Th1/Th2 antagonism: each immune axis represses the other -> mutual FACILITATION
    def a_I1_prod(x,th,t):  return th['b1']*x[3]*x[0]/(Kc+x[0])/(1.0+th['s3']*x[5]+s21*x[4])
    def a_I1_decay(x,th,t): return th['mu1']*x[3]
    def a_I2_prod(x,th,t):  return th['b2']*x[4]*x[1]/(Kc+x[1])/(1.0+th['s3']*x[5]+s12*x[3])
    def a_I2_decay(x,th,t): return th['mu2']*x[4]
    # I3 (regulation/Treg): induction saturated by the worm (evolvable trait g3h) + effectors
    def a_I3_prod(x,th,t):  return th['g3h']*x[1]/(Kh+x[1]) + th['g3']*(x[3]+x[4])
    def a_I3_decay(x,th,t): return th['mu3']*x[5]
    reactions = [
        (a_P1_repl, np.array([+1,0,0,0,0,0])),
        (a_P1_imm,  np.array([-1,0,0,0,0,0])),
        (a_P1_bg,   np.array([-1,0,0,0,0,0])),
        (a_P2_repl, np.array([0,+1,0,0,0,0])),
        (a_P2_imm,  np.array([0,-1,0,0,0,0])),
        (a_P2_bg,   np.array([0,-1,0,0,0,0])),
        (a_R_prod,  np.array([0,0,+1,0,0,0])),
        (a_R_cons,  np.array([0,0,-1,0,0,0])),
        (a_I1_prod, np.array([0,0,0,+1,0,0])),
        (a_I1_decay,np.array([0,0,0,-1,0,0])),
        (a_I2_prod, np.array([0,0,0,0,+1,0])),
        (a_I2_decay,np.array([0,0,0,0,-1,0])),
        (a_I3_prod, np.array([0,0,0,0,0,+1])),
        (a_I3_decay,np.array([0,0,0,0,0,-1])),
    ]
    return x0, reactions
