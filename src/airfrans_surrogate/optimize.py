"""Multi-start gradient-based search that only accepts converged, feasible results."""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class CallCounter:
    """Counts surrogate calls across every function wrapped with it (objective, constraint, Jacobian)."""

    def __init__(self):
        self.n = 0

    def wrap(self, f):
        def counted(*args, **kwargs):
            self.n += 1
            return f(*args, **kwargs)
        return counted

    def reset(self):
        n, self.n = self.n, 0
        return n


def is_feasible(x, bounds, constraints=(), tol=1e-6):
    """True if x lies inside the bounds and satisfies every inequality constraint (fun(x) >= 0)."""
    x = np.asarray(x, dtype=float)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    if np.any(x < lo - tol) or np.any(x > hi + tol):
        return False
    for c in constraints:
        if c.get('type', 'ineq') != 'ineq':
            raise ValueError('only inequality constraints are supported')
        if np.any(np.asarray(c['fun'](x)) < -tol):
            return False
    return True


def multistart(fun_grad, bounds, n_starts, seed=0, method='L-BFGS-B', constraints=(), maxiter=200, tol=1e-6):
    """Minimise fun_grad (returning value and gradient) from Latin-hypercube starting points.

    A start is accepted only if the optimizer reports success and the result is feasible; the best
    accepted result is returned. Raises if no start is accepted.
    """
    starts = qmc.scale(qmc.LatinHypercube(len(bounds), seed=seed).random(n_starts),
                       [b[0] for b in bounds], [b[1] for b in bounds])
    runs, best = [], None
    for s in starts:
        r = minimize(fun_grad, s, jac=True, method=method, bounds=bounds,
                     constraints=constraints, options=dict(maxiter=maxiter))
        feasible = is_feasible(r.x, bounds, constraints, tol)
        runs.append(dict(f=float(r.fun), success=bool(r.success), feasible=feasible))
        if r.success and feasible and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        raise RuntimeError(f'none of the {n_starts} starts converged to a feasible point')
    return dict(x=best.x, f=float(best.fun),
                starts=np.array([r['f'] for r in runs]),
                accepted=np.array([r['success'] and r['feasible'] for r in runs]))
