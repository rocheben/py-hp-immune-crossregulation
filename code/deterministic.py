"""
Deterministic (mean-field) limit of the same reaction models.

Since models.py describes each model as a list of reactions
    (propensity a_j(x,theta,t), stoichiometry nu_j),
the mean trajectory obeys exactly the ODE
    dx/dt = sum_j  nu_j * a_j(x, theta, t)
We therefore reuse the same model definitions, without
rewriting them: simulate_det takes a builder as an argument.

Advantage: smooth calibration objective -> gradient-based optimization (least_squares),
in seconds, with direct likelihood (no particle filter).
"""
import numpy as np
from scipy.integrate import solve_ivp


def rhs(t, x, reactions, theta):
    dx = np.zeros_like(x)
    for a_fn, nu in reactions:
        dx += nu * max(0.0, a_fn(x, theta, t))
    return dx


def simulate_det(builder, theta, t_grid):
    """Integrate the mean ODE and return the state at times t_grid: (len(t_grid), n_states)."""
    x0, reactions = builder(theta)
    t_grid = np.asarray(t_grid, float)
    sol = solve_ivp(rhs, (0.0, t_grid[-1]), x0, t_eval=t_grid,
                    args=(reactions, theta), method="LSODA",
                    rtol=1e-6, atol=1e-8)
    if not sol.success:
        return np.full((len(t_grid), len(x0)), np.nan)
    return sol.y.T
