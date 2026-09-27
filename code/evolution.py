"""
Evolutionary layer on the deterministic within-host model: library + runner.

Hybrid principle (validated with the user):
  - the within-host FITNESS of the worm is evaluated by the mean ODE (deterministic.py): fast, without Monte-Carlo noise -> clean selection gradients;
  - INVASION remains STOCHASTIC (fixation of a rare mutant = true demographic
    stochasticity), the only component where the SSA is justified.

Evolved trait: g3h = strength with which the worm (P2) induces the regulatory pathway I3
(immunosuppression, Hp-TGM type).
  Benefit: I3 ↑ -> immune killing of the worm (k2·I2) ↓ -> load/eggs ↑.
  Cost   : r2_eff = r2·(1 − kappa·g3h)  (producing the suppressor cuts into
           replication).

IMPORTANT NB — immune regime of the worm.
For suppression to be ADAPTIVE, the worm must be immuno-CONTROLLED (load
kept BELOW its capacity by Th2 immunity), otherwise relaxing immunity
brings nothing. We therefore adopt here a worm parameterization with persistent immunity
(b2 > mu2), consistent with the premise of the WP. This is a choice for the
evolutionary analysis, distinct from the acute-phase regime used to identify the interaction
parameters (estimate.py), where the flat egg data do not discriminate between the two.

Comparison of experimental regimes:
  - SI  (single-infection lines)  : worm alone -> build_hp_single
  - COI (coinfection lines)       : worm + Py  -> build_coinfection

Execution (produces figures/Figure4_..., Figure5_..., Figure6_...):
  python evolution.py                      # default kappa
  python evolution.py --kappa 1.5 --ngen 80 --reps 6
"""
import argparse, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deterministic import simulate_det
from models import build_hp_single, build_coinfection

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = os.path.join(HERE, "fits")
FIG  = os.path.join(HERE, "figures"); os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Assembling the base parameterization for the evolution
# --------------------------------------------------------------------------- #
def _load(name):
    p = os.path.join(FITS, name)
    return json.load(open(p)) if os.path.exists(p) else {}

def base_theta():
    """Parameterization fitted to data: Py (single.json) + worm (worm.json) +
    Th1/Th2 interaction & ceiling (coinf.json).

    EVOLUTIONARY REGIME (EVO overrides below, distinct from the acute fit): for
    the evolvable trait g3h (Treg/I3 induction by the worm) to be relevant, we place
    the regulatory pathway in an ACTIVE regime (s3 stronger than at the acute peak of Py, where I3
    counted little) and we give the worm a capacity margin (K2). The INTERACTION
    parameters (s12, s21 = Th1/Th2 antagonism) remain those ESTIMATED on data
    -> it is through them that coinfection modifies selection on g3h."""
    py   = _load("single.json")
    worm = _load("worm.json")            # r2, k2, beta_e
    coinf = _load("coinf.json")          # K2, s12, s21 (estimated interaction)
    worm_fixed = dict(P2_0=20, I2_0=1, d2=0.02, b2=0.7, mu2=0.4, g3h=0.02)
    defaults = dict(R0=1000, Rmax=1000, Rh=200, mu3=0.4, g3=0.02,
                    Kc=50.0, P1_0=5, I1_0=1, I3_0=1,
                    r1=0.8, k1=0.05, b1=0.5, c1=40.0, rho=15.0, d1=0.2, mu1=0.4,
                    r2=0.6, k2=0.2, beta_e=800.0, s12=8.0, s21=4.0)
    # Evolutionary regime validated by 2D sweep (appendix): the regulatory pathway must
    # be efficient enough (high s3) and the anti-worm killing strong enough (k2) for
    # the trait g3h to have a real marginal benefit -> INTERIOR coinfection ESS and
    # strictly positive (instead of degenerating to 0). Moderate Kh, K2.
    EVO = dict(s3=6.0, k2=6.0, mu3=0.2, Kh=30.0, K2=300.0)
    th = {**defaults, **py, **worm, **coinf, **worm_fixed}
    th.update(EVO)                            # the evo regime takes priority over K2/s3 from the acute fit
    return th

# --------------------------------------------------------------------------- #
#  Deterministic fitness of the worm
# --------------------------------------------------------------------------- #
def worm_fitness(g3h, base, coinfection, kappa, t_grid):
    """Cumulative transmissive output of the worm ~ ∫ P2 dt (proxy for eggs)."""
    th = dict(base)
    th['g3h'] = g3h
    th['r2']  = base['r2'] * (1.0 - kappa * g3h)     # cost on replication
    if th['r2'] <= 0:
        return 0.0
    builder = build_coinfection if coinfection else build_hp_single
    P2idx   = 1 if coinfection else 0                # state: COI=[P1,P2,...], SI=[P2,...]
    y = simulate_det(builder, th, t_grid)
    if np.any(~np.isfinite(y)):
        return 0.0
    return float(np.trapz(np.maximum(y[:, P2idx], 0.0), t_grid)) \
        if hasattr(np, "trapz") else float(np.trapezoid(np.maximum(y[:, P2idx], 0.0), t_grid))

# --------------------------------------------------------------------------- #
#  Evolution: deterministic gradient + stochastic fixation
# --------------------------------------------------------------------------- #
def evolve(base, coinfection, kappa, t_grid,
           g0=0.02, n_gen=60, mut_sd=0.01, seed=0):
    rng = np.random.default_rng(seed)
    g = g0
    traj = [g]
    for _ in range(n_gen):
        gm = max(0.0, g + rng.normal(0, mut_sd))                  # mutant
        wr = worm_fitness(g,  base, coinfection, kappa, t_grid)   # resident
        wm = worm_fitness(gm, base, coinfection, kappa, t_grid)   # mutant
        # stochastic invasion: fixation probability of a beneficial mutant ~ 2s
        # (selection coefficient s = (wm-wr)/wr, Haldane/Kimura approx.)
        s = (wm - wr) / (wr + 1e-12)
        if s > 0 and rng.random() < min(1.0, 2.0 * s):
            g = gm
        traj.append(g)
    return np.array(traj)

def selection_gradient(base, coinfection, kappa, t_grid, g_grid):
    """Fitness (deterministic) along a g3h axis -> evolutionary optimum (approx. ESS)."""
    return np.array([worm_fitness(g, base, coinfection, kappa, t_grid) for g in g_grid])

# --------------------------------------------------------------------------- #
#  Figures: worm alone (SI) vs coinfection (COI)
#  One function per figure; make_figure4/5/6.py call them, main() runs all three.
# --------------------------------------------------------------------------- #
G_GRID = np.linspace(0, 0.5, 51)

def _t_grid(tmax):
    return np.linspace(0, tmax, int(tmax) + 1)

def figure4(kappa=0.8, tmax=30.0):
    """Figure 4: selection gradient (fitness vs g3h) at kappa and 2*kappa."""
    base = base_theta(); t_grid = _t_grid(tmax); g_grid = G_GRID
    print("Selection gradient (fitness ∫P2 vs g3h):")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for j, kap in enumerate([kappa, 2 * kappa]):
        for coi, col, lab in [(False, "C0", "SI"), (True, "C3", "COI")]:
            w = selection_gradient(base, coi, kap, t_grid, g_grid)
            ess = g_grid[np.argmax(w)]
            nm = "worm alone" if lab == "SI" else "coinfection"
            ax[j].plot(g_grid, w / w.max(), col + "-",
                       label=nm + f" (optimum {ess:.3f})")
            ax[j].axvline(ess, color=col, ls=":", alpha=.6)
            print(f"  kappa={kap:.2f}  {lab:3s}: ESS g3h ≈ {ess:.3f}")
        ax[j].set_xlabel(r"$g_{3h}$ (worm immunosuppression)")
        ax[j].set_ylabel(r"relative fitness ($\int P_2\,\mathrm{d}t$)")
        ax[j].legend(title=r"$\kappa$ = %.2f" % kap); ax[j].grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "Figure4.png")
    fig.savefig(p, dpi=130); plt.close(fig); print(f"    -> {p}")
    return p

def figure5(kappa=0.8, ngen=400, reps=8, g0=0.02, tmax=30.0):
    """Figure 5: stochastic evolutionary trajectories of g3h.
    Started from an ANCESTRAL state of negligible suppression (g0) and run long
    enough for the lines to climb toward their deterministic optimum. Starting
    above the optima would only show a slow downward drift and would not show
    which regime selects the higher trait."""
    base = base_theta(); t_grid = _t_grid(tmax); g_grid = G_GRID
    ess = {lab: g_grid[np.argmax(selection_gradient(base, coi, kappa, t_grid, g_grid))]
           for coi, lab in [(False, "SI"), (True, "COI")]}
    print(f"Evolutionary trajectories (kappa={kappa}, {ngen} gen, {reps} lines, g0={g0}):")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ends = {"SI": [], "COI": []}
    for coi, col, lab in [(False, "C0", "SI"), (True, "C3", "COI")]:
        for r in range(reps):
            tr = evolve(base, coi, kappa, t_grid, g0=g0,
                        n_gen=ngen, mut_sd=0.02, seed=100 * coi + r)
            ax.plot(tr, col + "-", alpha=.4, lw=1)
            ends[lab].append(tr[-1])
        ax.axhline(ess[lab], color=col, ls=":", alpha=.7, lw=1.2)
        nm = "worm alone" if lab == "SI" else "coinfection"
        ax.plot([], [], col + "-", label=nm +
                r" (final $g_{3h}$ = %.3f $\pm$ %.3f)" % (np.mean(ends[lab]), np.std(ends[lab])))
    ax.plot([], [], "k:", lw=1.2, label="deterministic optimum")
    ax.set_xlabel("generation"); ax.set_ylabel(r"evolved $g_{3h}$")
    ax.legend(loc="lower right", title=r"$\kappa$ = %.2f" % kappa); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "Figure5.png")
    fig.savefig(p, dpi=130); plt.close(fig)

    for lab in ("SI", "COI"):
        print("  %-3s final %.3f +/- %.3f  (ESS %.3f)"
              % (lab, np.mean(ends[lab]), np.std(ends[lab]), ess[lab]))
    d = np.mean(ends["COI"]) - np.mean(ends["SI"])
    print("  -> COI selects for %s suppression" %
          ("STRONGER" if d > 0.005 else "WEAKER (relaxed)" if d < -0.005 else "~ equivalent"))
    print(f"    -> {p}")
    return p

def figure6(kappa=0.8, tmax=30.0):
    """Figure 6: sweep of the Th1/Th2 antagonism (s12), ESS SI vs COI.
    s12 = strength with which anti-Py Th1 represses anti-worm Th2 (estimated on
    data). The stronger s12, the more FREE immunosuppression the worm receives
    in coinfection -> the less incentive it has to evolve its own suppression (g3h)."""
    base = base_theta(); t_grid = _t_grid(tmax); g_grid = G_GRID
    print(f"Sweep of s12 (Th1/Th2 antagonism) — ESS SI vs COI, kappa={kappa}:")
    s12s = np.linspace(0, max(base.get("s12", 8.0), 20.0), 13)
    ess_si, ess_coi = [], []
    for sv in s12s:
        b = base_theta(); b["s12"] = sv
        ess_si.append(g_grid[np.argmax(selection_gradient(b, False, kappa, t_grid, g_grid))])
        ess_coi.append(g_grid[np.argmax(selection_gradient(b, True, kappa, t_grid, g_grid))])
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(s12s, ess_si, "C0-o", label="worm alone")
    ax.plot(s12s, ess_coi, "C3-s", label="coinfection")
    ax.fill_between(s12s, ess_coi, ess_si, color="C0", alpha=.12)
    ax.axvline(base.get("s12", 8.0), color="k", ls=":", lw=1,
               label=r"estimated $s_{12}$")
    ax.set_xlabel(r"$s_{12}$, Th1 repression of Th2 (cross-antagonism)")
    ax.set_ylabel(r"selected $g_{3h}$ at the evolutionary optimum")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); p = os.path.join(FIG, "Figure6.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  s12=0 -> SI={ess_si[0]:.3f} COI={ess_coi[0]:.3f} ; "
          f"s12={s12s[-1]:.1f} -> SI={ess_si[-1]:.3f} COI={ess_coi[-1]:.3f}")
    print(f"    -> {p}")
    return p

def main():
    ap = argparse.ArgumentParser(description="Evolutionary analysis (Figures 4, 5, 6)")
    ap.add_argument("--kappa", type=float, default=0.8, help="cost of suppression")
    ap.add_argument("--ngen",  type=int,   default=400)
    ap.add_argument("--g0",    type=float, default=0.02, help="ancestral trait value")
    ap.add_argument("--reps",  type=int,   default=8, help="replicate lines per regime")
    ap.add_argument("--tmax",  type=float, default=30.0)
    a = ap.parse_args()
    figure6(a.kappa, a.tmax)
    figure4(a.kappa, a.tmax)
    figure5(a.kappa, a.ngen, a.reps, a.g0, a.tmax)

if __name__ == "__main__":
    main()
