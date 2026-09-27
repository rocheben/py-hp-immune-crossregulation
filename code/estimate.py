"""
Estimation of the within-host model parameters (models.py)
on the data, + data/model goodness-of-fit figures + uncertainty.

Three steps (cf. --help):

  1. single   : JOINT calibration of Py parasitemia + RBC in single infection
                (SE-weighted least squares, multi-start, deterministic ODE).
  2. worm     : calibration of the worm-alone egg plateau (r2, k2, egg scale).
  3. coinf    : estimation of the coinfection INTERACTION parameters
                (K2 = worm ceiling; s12, s21 = Th1/Th2 cross-antagonism),
                the shared intra-host parameters being fixed to the earlier fits.
  4. profile  : confidence intervals by LIKELIHOOD PROFILE
                (Delta chi2 < 3.84 -> 95% CI, 1 df) for r1, s12 and s21.

Each step writes its figures into figures/ and its parameters into fits/.

--------------------------------------------------------------------------
HOW TO ESTIMATE THE PARAMETERS AND GENERATE THE FIGURES
--------------------------------------------------------------------------
  python estimate.py single      # calibrates Py single, writes figures/Figure1.png
  python estimate.py worm        # calibrates Hp alone, writes figures/Figure2.png
  python estimate.py coinf       # estimates K2,s12,s21, writes figures/Figure3.png
  python estimate.py profile     # profile CIs, writes figures/FigureA1.png
  python estimate.py all         # everything, in order (recommended)

Options: --restarts N (multi-start starting points), --grid N (profile points).

Principle: the ODE (deterministic.py) gives a SMOOTH objective -> `least_squares`
(Levenberg/TRF) converges in a fraction of a second. We minimize the
NORMALIZED residuals (model-obs)/SE: this is a chi-2, directly interpretable
(chi2/df ~ 1 => fit compatible with the observation noise).
"""
import argparse, json, os, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from deterministic import simulate_det
from models import build_py_single, build_coinfection

HERE = os.path.dirname(os.path.abspath(__file__))
DP   = os.path.join(HERE, "data_processed")
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)
FITS = os.path.join(HERE, "fits");    os.makedirs(FITS, exist_ok=True)

# ======================================================================== #
#  Data
# ======================================================================== #
def load_para(treat):
    p = pd.read_csv(os.path.join(DP, "py_parasitemia_summary.csv"))
    s = p[p.Treatment == treat].sort_values("time_post_infection")
    return (s.time_post_infection.values.astype(float),
            s["mean"].values.astype(float), s["se"].values.astype(float))

def load_rbc(treat):
    r = pd.read_csv(os.path.join(DP, "rbc_summary.csv"))
    s = r[r.Treatment == treat].sort_values("time_post_infection")
    return (s.time_post_infection.values.astype(float),
            s["mean"].values.astype(float), s["se"].values.astype(float))

def load_lfec(treat):
    d = pd.read_csv(os.path.join(DP, "hp_lfec_summary.csv"))
    s = d[d.Treatment == treat].sort_values("time_post_infection")
    return (s.time_post_infection.values.astype(float),
            s["mean"].values.astype(float), s["se"].values.astype(float))

T_SINGLE = "Single Py"                                    # single parasitemia
T_COINF  = "coinfected (simultaneous and Hp first)"       # coinf parasitemia
T_RBC_S  = "single Py"                                    # single RBC (distinct label!)
T_LFEC_S = "single Hp"                                    # eggs, worm alone
T_LFEC_C = "Py-14+Hp"                                     # eggs, coinfection (Py first)

# ======================================================================== #
#  Parameters: fixed, estimated, bounds
# ======================================================================== #
FIXED_S = dict(R0=1000, Rmax=1000, Rh=200, mu3=0.4, g3=0.02, s3=0.05,
               P1_0=5, I1_0=1, I3_0=1, Kc=50.0)
KEYS_S  = ["r1", "k1", "b1", "c1", "rho", "d1", "mu1"]
BOUNDS  = {                    # bounds (natural scale): multi-start + profile + constraint
    "r1": (0.3, 1.5), "k1": (0.005, 0.5), "b1": (0.03, 2.0),
    "c1": (1.0, 200.0), "rho": (1.0, 100.0), "d1": (0.01, 0.4), "mu1": (0.03, 2.0),
    "r2": (0.1, 3.0), "k2": (0.05, 3.0), "beta_e": (1.0, 20000.0),
    "K2": (30.0, 4000.0),                    # ceiling (sets the magnitude of Hp facilitation)
    "s12": (1e-3, 20.0), "s21": (1e-3, 20.0),  # Th1/Th2 antagonism
    "g3h": (1e-3, 1.0),                      # evolvable trait (fixed at the fit, varied in evolution)
}

def log_bounds(keys):
    lb = np.log([BOUNDS[k][0] for k in keys]); ub = np.log([BOUNDS[k][1] for k in keys])
    return lb, ub

def clip_log(x0, keys):
    lb, ub = log_bounds(keys)
    return np.clip(x0, lb + 1e-9, ub - 1e-9)
# worm (Hp): chronic burden, PERSISTENT anti-worm immunity (b2>mu2) -> plateau
# maintained BELOW the K2 ceiling by Th2 control. In coinfection, the Th1/Th2
# antagonism (Py represses Th2) releases this control -> the worm rises toward K2
# (bounded facilitation). r2,k2,beta_e calibrated on "single Hp"; K2,s12,s21 on coinfection.
WORM_FIXED = dict(P2_0=20, d2=0.02, b2=0.7, mu2=0.4, I2_0=1, Kh=80.0, g3h=0.02)
KEYS_W = ["r2", "k2", "beta_e"]      # worm alone: immuno-controlled level + egg scale
KEYS_C = ["K2", "s12", "s21"]        # interaction: ceiling + Th1/Th2 antagonism

# ======================================================================== #
#  Model predictions
# ======================================================================== #
def _sim_single(theta, tt):
    y = simulate_det(build_py_single, theta, tt)
    return y[:, 0], y[:, 1]                 # P1, R

def _sim_coinf(theta, tt):
    y = simulate_det(build_coinfection, theta, tt)
    return y[:, 0], y[:, 2]                 # P1, R

def frac(P1, R):
    return P1 / (P1 + R + 1e-9)

# ======================================================================== #
#  Normalized residuals (= chi)
# ======================================================================== #
def residuals_single(logx, keys, fixed, data):
    (tp, yp, sp), (tr, yr, sr), R0obs = data
    th = dict(fixed)
    for k, v in zip(keys, np.exp(logx)):
        th[k] = v
    tt = np.union1d(tp, tr)
    P1, R = _sim_single(th, tt)
    if np.any(~np.isfinite(P1)):
        return np.full(len(tp) + len(tr), 1e3)
    # RBC compared to R (UNinfected red blood cells = resource): couples the RBC
    # recovery to parasite clearance (recovery => fewer parasites), which
    # imposes the resolution observed at day 21. (R+P1 would decouple the two.)
    r_para = (np.interp(tp, tt, frac(P1, R)) - yp) / sp
    r_rbc  = (np.interp(tr, tt, R / th["R0"]) - yr / R0obs) / (sr / R0obs)
    return np.concatenate([r_para, r_rbc])

def _sim_hp(theta, tt):
    from models import build_hp_single
    y = simulate_det(build_hp_single, theta, tt)
    return y[:, 0]                          # P2

def lfec(P2, beta_e):
    return np.log1p(beta_e * np.maximum(P2, 0.0))

def _theta_worm_ss(theta):
    """Initialize the worm at its chronic SET-POINT (steady state of the worm alone).
    The eggs are only measured in the chronic phase (no establishment data),
    so we start from the established worm and only model the coinfection perturbation."""
    from models import build_hp_single
    ss = simulate_det(build_hp_single, theta, np.array([0.0, 400.0]))[-1]  # [P2,I2,I3]
    th = dict(theta)
    if np.all(np.isfinite(ss)):
        th["P2_0"], th["I2_0"], th["I3_0"] = float(ss[0]), float(ss[1]), float(ss[2])
    return th

def residuals_worm(logx, keys, fixed, data):
    tl, yl, sl = data
    th = dict(fixed)
    for k, v in zip(keys, np.exp(logx)):
        th[k] = v
    P2 = _sim_hp(_theta_worm_ss(th), tl)    # worm at the chronic set-point -> flat plateau
    if np.any(~np.isfinite(P2)):
        return np.full(len(tl), 1e3)
    return (lfec(P2, th["beta_e"]) - yl) / sl

def residuals_coinf(logx, keys, fixed, data):
    """data = (para_coinf, lfec_coinf, lfec_single): JOINT fit
       Py parasitemia (coinf) + Hp eggs (coinf) + Hp eggs (worm alone = anchor)."""
    (tp, yp, sp), (tl, yl, sl), (ts, ys, ss) = data
    th = dict(fixed)
    for k, v in zip(keys, np.exp(logx)):
        th[k] = v
    th = _theta_worm_ss(th)                     # worm initialized at its chronic set-point
    tt = np.union1d(tp, tl)
    yy = simulate_det(build_coinfection, th, tt)
    P2s = _sim_hp(th, ts)                        # worm alone, SAME params (including K2)
    if np.any(~np.isfinite(yy)) or np.any(~np.isfinite(P2s)):
        return np.full(len(tp) + len(tl) + len(ts), 1e3)
    P1, P2, R = yy[:, 0], yy[:, 1], yy[:, 2]
    r_para  = (np.interp(tp, tt, frac(P1, R)) - yp) / sp
    r_lfecC = (np.interp(tl, tt, lfec(P2, th["beta_e"])) - yl) / sl
    r_lfecS = (lfec(P2s, th["beta_e"]) - ys) / ss
    return np.concatenate([r_para, r_lfecC, r_lfecS])

# ======================================================================== #
#  Multi-start optimizer
# ======================================================================== #
def multistart(resid_fn, keys, fixed, data, restarts, seed=0, x0_first=None):
    rng = np.random.default_rng(seed)
    lb, ub = log_bounds(keys)
    best = None
    for r in range(restarts):
        if r == 0 and x0_first is not None:
            x0 = clip_log(np.log(x0_first), keys)
        else:
            x0 = np.log([rng.uniform(*BOUNDS[k]) for k in keys])
        try:
            res = least_squares(resid_fn, x0, args=(keys, fixed, data),
                                bounds=(lb, ub),           # CONSTRAINED optimization
                                method="trf", xtol=1e-12, ftol=1e-12, max_nfev=4000)
        except Exception:
            continue
        if best is None or np.sum(res.fun**2) < np.sum(best.fun**2):
            best = res
    theta = dict(fixed)
    for k, v in zip(keys, np.exp(best.x)):
        theta[k] = v
    return theta, float(np.sum(best.fun**2)), best

# ======================================================================== #
#  Step 1 — single calibration (parasitemia + RBC)
# ======================================================================== #
def fit_single(restarts=10, seed=0):
    tp, yp, sp = load_para(T_SINGLE)
    tr, yr, sr = load_rbc(T_RBC_S)
    data = ((tp, yp, sp), (tr, yr, sr), yr[0])
    t0 = time.time()
    theta, chi2, _ = multistart(residuals_single, KEYS_S, FIXED_S, data,
                                restarts, seed, x0_first=[0.6,0.05,0.3,0.5,0.3,0.05,0.2])
    dt = time.time() - t0
    ndat = len(tp) + len(tr); ddl = ndat - len(KEYS_S)
    print(f"[single] {dt:.2f}s  chi2={chi2:.1f}  chi2/ddl={chi2/ddl:.2f}  (n={ndat})")
    for k in KEYS_S:
        print(f"    {k:4s}= {theta[k]:.4f}")
    json.dump({k: theta[k] for k in KEYS_S} | {k: FIXED_S[k] for k in FIXED_S},
              open(os.path.join(FITS, "single.json"), "w"), indent=2)
    _fig_single(theta, data)
    return theta, chi2

def _fig_single(theta, data):
    (tp, yp, sp), (tr, yr, sr), R0obs = data
    tt = np.linspace(0, max(tp[-1], tr[-1]), 200)
    P1, R = _sim_single(theta, tt)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(tt, frac(P1, R), "C0-", lw=2, label="model")
    ax[0].errorbar(tp, yp, yerr=sp, fmt="ks", capsize=3, label="data")
    ax[0].set_ylabel("parasitaemia (iRBC fraction)")
    ax[1].plot(tt, R / theta["R0"], "C2-", lw=2, label="model")
    ax[1].errorbar(tr, yr / R0obs, yerr=sr / R0obs, fmt="ks", capsize=3, label="data")
    ax[1].set_ylabel("uninfected RBC (normalised to day 0)")
    for a in ax:
        a.set_xlabel("days post-infection"); a.grid(alpha=.3); a.legend()
    fig.tight_layout(); p = os.path.join(FIG, "Figure1.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"    -> {p}")

# ======================================================================== #
#  Step 2a — worm calibration (Hp alone) on faecal eggs
# ======================================================================== #
def fit_worm(restarts=14, seed=2):
    tl, yl, sl = load_lfec(T_LFEC_S)
    # K2 fixed large here: the worm alone is immuno-controlled well BELOW the ceiling,
    # so K2 is not identifiable from the "single" eggs -> estimated in coinfection.
    fixed = dict(WORM_FIXED, K2=4000.0, s3=0.05, g3=0.02, mu3=0.4, I3_0=1, Kc=50.0)
    data = (tl, yl, sl)
    t0 = time.time()
    theta, chi2, _ = multistart(residuals_worm, KEYS_W, fixed, data,
                                restarts, seed, x0_first=[0.6, 0.8, 800.0])
    dt = time.time() - t0
    ddl = max(len(tl) - len(KEYS_W), 1)
    print(f"[worm  ] {dt:.2f}s  chi2={chi2:.1f}  chi2/ddl={chi2/ddl:.2f}  (n={len(tl)})")
    for k in KEYS_W:
        print(f"    {k:6s}= {theta[k]:.4f}")
    json.dump({k: theta[k] for k in KEYS_W}, open(os.path.join(FITS, "worm.json"), "w"), indent=2)
    _fig_worm(theta, data)
    return theta, chi2

def _fig_worm(theta, data):
    tl, yl, sl = data
    tt = np.linspace(0, tl[-1], 200)
    P2 = _sim_hp(_theta_worm_ss(theta), tt)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(tt, lfec(P2, theta["beta_e"]), "C1-", lw=2, label="model")
    ax.errorbar(tl, yl, yerr=sl, fmt="ks", capsize=3, label="data (single Hp)")
    ax.set_xlabel("days post-infection (Hp)"); ax.set_ylabel("log faecal egg count")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "Figure2.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"    -> {p}")

# ======================================================================== #
#  Step 2b - coinfection interaction (K2, s12, s21): JOINT fit Py + Hp
# ======================================================================== #
def _fixed_coinf():
    """Assemble the shared parameters: Py (single) + worm + fixed worm immunity."""
    if not os.path.exists(os.path.join(FITS, "single.json")):
        fit_single()
    if not os.path.exists(os.path.join(FITS, "worm.json")):
        fit_worm()
    base = json.load(open(os.path.join(FITS, "single.json")))
    worm = json.load(open(os.path.join(FITS, "worm.json")))
    return dict(base) | dict(WORM_FIXED) | worm, base, worm

def fit_coinf(restarts=12, seed=1):
    fixed, base, worm = _fixed_coinf()
    tp, yp, sp = load_para(T_COINF)        # Py parasitemia in coinfection
    tl, yl, sl = load_lfec(T_LFEC_C)       # Hp eggs in coinfection
    ts, ys, ss = load_lfec(T_LFEC_S)       # Hp eggs worm alone (anchor)
    data = ((tp, yp, sp), (tl, yl, sl), (ts, ys, ss))
    t0 = time.time()
    # x0 in the STRONG-facilitation basin (s12,s21 high): the objective has a
    # second, WEAKER local optimum near s12=s21=1 with nearly identical chi2
    # (~5.05 vs 4.97) but almost no interaction. Seeding at [~K2, 8, 4] lands on
    # the true global optimum (s12 saturates high, s21~4 -> mutual facilitation).
    theta, chi2, _ = multistart(residuals_coinf, KEYS_C, fixed, data,
                                restarts, seed, x0_first=[112.0, 8.0, 4.0])
    dt = time.time() - t0
    ndat = len(tp) + len(tl) + len(ts); ddl = ndat - len(KEYS_C)
    print(f"[coinf ] {dt:.2f}s  chi2={chi2:.1f}  chi2/ddl={chi2/ddl:.2f}  (n={ndat}: "
          f"{len(tp)} Py + {len(tl)} eggs-coinf + {len(ts)} eggs-single)")
    for k in KEYS_C:
        print(f"    {k:4s}= {theta[k]:.4f}")
    json.dump({k: theta[k] for k in KEYS_C}, open(os.path.join(FITS, "coinf.json"), "w"), indent=2)
    _fig_coinf(theta, base, worm)
    return theta, chi2

def _fig_coinf(theta_c, base, worm):
    tp, yp, sp = load_para(T_SINGLE); tpc, ypc, spc = load_para(T_COINF)
    tls, yls, sls = load_lfec(T_LFEC_S); tlc, ylc, slc = load_lfec(T_LFEC_C)
    tt = np.linspace(0, max(tpc[-1], tlc[-1]), 240)
    P1s, Rs = _sim_single(base, tt)
    fixed = dict(base) | dict(WORM_FIXED) | dict(worm) | {k: theta_c[k] for k in KEYS_C}
    fixed = _theta_worm_ss(fixed)               # worm at the chronic set-point
    yy = simulate_det(build_coinfection, fixed, tt)
    P1c, P2c, Rc = yy[:, 0], yy[:, 1], yy[:, 2]
    P2single = _sim_hp(fixed, tt)               # worm alone, SAME params (including fitted K2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # -- Py parasitemia: single vs coinf --
    ax[0].plot(tt, frac(P1s, Rs), "C0-", lw=2, label="model single")
    ax[0].errorbar(tp, yp, yerr=sp, fmt="C0o", capsize=3, label="data single")
    ax[0].plot(tt, frac(P1c, Rc), "C3-", lw=2, label="model coinfected")
    ax[0].errorbar(tpc, ypc, yerr=spc, fmt="C3s", capsize=3, label="data coinfected")
    ax[0].set_ylabel("Py parasitaemia (iRBC fraction)")
    # -- Hp eggs: worm alone vs coinfection --
    ax[1].plot(tt, lfec(P2single, worm["beta_e"]), "C1-", lw=2, label="model worm alone")
    ax[1].plot(tt, lfec(P2c, worm["beta_e"]), "C3-", lw=2, label="model coinfected")
    ax[1].errorbar(tls, yls, yerr=sls, fmt="C1o", capsize=3, label="data worm alone")
    ax[1].errorbar(tlc, ylc, yerr=slc, fmt="C3s", capsize=3, label="data coinfected")
    ax[1].set_ylabel("Hp log faecal egg count")
    for a in ax:
        a.set_xlabel("days post-infection"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(FIG, "Figure3.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"    -> {p}")

# ======================================================================== #
#  Step 3 — likelihood profile (95% CI)
# ======================================================================== #
def _profile(param, keys, fixed, data, resid_fn, mle_theta, chi2_min, grid_n):
    """Fix `param` on a grid (centered on the MLE), re-optimize the others (bounded)."""
    blo, bhi = BOUNDS[param]
    lo = max(blo, mle_theta[param] * 0.4)
    # covers at least ~40% of the admissible interval -> shows the flatness
    # (non-identifiability) when the MLE is stuck against a bound
    hi = min(bhi, max(mle_theta[param] * 2.0, lo + 0.4 * (bhi - lo)))
    grid = np.linspace(lo, hi, grid_n)
    others = [k for k in keys if k != param]
    lb, ub = log_bounds(others) if others else (None, None)
    rng = np.random.default_rng(0)
    chis = []
    for gv in grid:
        fx = dict(fixed); fx[param] = gv
        if not others:
            chis.append(float(np.sum(resid_fn(np.array([]), [], fx, data)**2)))
            continue
        # profile = MIN over the nuisance parameters -> several starts, keep the min
        best = np.inf
        starts = [clip_log(np.log([max(mle_theta[k], 1e-6) for k in others]), others)]
        for _ in range(4):
            starts.append(np.log([rng.uniform(*BOUNDS[k]) for k in others]))
        for x0 in starts:
            try:
                r = least_squares(resid_fn, x0, args=(others, fx, data), bounds=(lb, ub),
                                  method="trf", xtol=1e-11, ftol=1e-11, max_nfev=3000)
                best = min(best, float(np.sum(r.fun**2)))
            except Exception:
                pass
        chis.append(best if np.isfinite(best) else np.nan)
    return grid, np.array(chis)

def _ci_from_profile(grid, chis, chi2_min, ddl):
    # overdispersion: observation SE too tight (chi2/df>1) -> we scale the
    # threshold by the reduced chi2 s2 (equivalent to inflating the variance).
    s2 = max(chi2_min / ddl, 1.0)
    thr = chi2_min + 3.84 * s2                 # chi2(0.95, 1 df) scaled
    below = grid[chis <= thr]
    return (below.min(), below.max()) if len(below) else (np.nan, np.nan), thr

def profile(grid_n=15):
    # --- r1 (single) ---
    tp, yp, sp = load_para(T_SINGLE); tr, yr, sr = load_rbc(T_RBC_S)
    data_s = ((tp, yp, sp), (tr, yr, sr), yr[0])
    base = json.load(open(os.path.join(FITS, "single.json"))) if os.path.exists(
        os.path.join(FITS, "single.json")) else fit_single()[0]
    mle_s = {k: base[k] for k in KEYS_S}
    chi2min_s = float(np.sum(residuals_single(np.log([mle_s[k] for k in KEYS_S]),
                                              KEYS_S, FIXED_S, data_s)**2))
    ddl_s = (len(tp) + len(tr)) - len(KEYS_S)
    g_r1, c_r1 = _profile("r1", KEYS_S, FIXED_S, data_s, residuals_single,
                          mle_s, chi2min_s, grid_n)
    ci_r1, thr_r1 = _ci_from_profile(g_r1, c_r1, chi2min_s, ddl_s)
    print(f"[profile] r1  = {base['r1']:.3f}  95% CI = [{ci_r1[0]:.3f}, {ci_r1[1]:.3f}]")

    # --- s12, s21 (Th1/Th2 antagonism, joint fit Py + Hp) ---
    if not os.path.exists(os.path.join(FITS, "coinf.json")):
        fit_coinf()
    coinf = json.load(open(os.path.join(FITS, "coinf.json")))
    fixed_c, _, _ = _fixed_coinf()
    tpc, ypc, spc = load_para(T_COINF); tlc, ylc, slc = load_lfec(T_LFEC_C)
    tsc, ysc, ssc = load_lfec(T_LFEC_S)
    data_c = ((tpc, ypc, spc), (tlc, ylc, slc), (tsc, ysc, ssc))
    mle_c = {k: coinf[k] for k in KEYS_C}
    chi2min_c = float(np.sum(residuals_coinf(np.log([mle_c[k] for k in KEYS_C]),
                                            KEYS_C, fixed_c, data_c)**2))
    ddl_c = (len(tpc) + len(tlc) + len(tsc)) - len(KEYS_C)
    prof = {}
    for name in ("s12", "s21"):
        g, c = _profile(name, KEYS_C, fixed_c, data_c, residuals_coinf,
                        mle_c, chi2min_c, grid_n)
        ci, thr = _ci_from_profile(g, c, chi2min_c, ddl_c)
        prof[name] = (g, c, thr, ci)
        print(f"[profile] {name:3s} = {coinf[name]:.3f}  95% CI = [{ci[0]:.3f}, {ci[1]:.3f}]")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    panels = [(g_r1, c_r1, thr_r1, ci_r1, r"$r_1$ (Py replication rate)", base["r1"]),
              (*prof["s12"], r"$s_{12}$ (Th1 repression of Th2)", coinf["s12"]),
              (*prof["s21"], r"$s_{21}$ (Th2 repression of Th1)", coinf["s21"])]
    for a, (g, c, thr, ci, name, mle) in zip(ax, panels):
        a.plot(g, c, "C0-o", ms=3)
        a.axhline(thr, color="C3", ls="--", label="95% CI threshold (scaled)")
        a.axvline(mle, color="k", ls=":", label="MLE")
        if np.isfinite(ci[0]): a.axvspan(ci[0], ci[1], color="C0", alpha=.12)
        a.set_xlabel(name); a.set_ylabel(r"profile $\chi^2$"); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); p = os.path.join(FIG, "FigureA1.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"    -> {p}")

# ======================================================================== #
def main():
    ap = argparse.ArgumentParser(description="Estimation and goodness-of-fit figures")
    ap.add_argument("step", nargs="?", default="all",
                    choices=["single", "worm", "coinf", "profile", "all"])
    ap.add_argument("--restarts", type=int, default=10)
    ap.add_argument("--grid", type=int, default=15)
    a = ap.parse_args()
    if a.step in ("single", "all"):
        fit_single(a.restarts)
    if a.step in ("worm", "all"):
        fit_worm(a.restarts)
    if a.step in ("coinf", "all"):
        fit_coinf(a.restarts)
    if a.step in ("profile", "all"):
        profile(a.grid)
    print("\nDone. Figures in figures/, parameters in fits/.")

if __name__ == "__main__":
    main()
