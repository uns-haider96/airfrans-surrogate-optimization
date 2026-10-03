"""Surface ordering, pressure-force integration and the global shape descriptor."""

import numpy as np

# Chord stations of the camber-line part of the shape descriptor (Phase 1b configuration)
STATIONS = np.array([0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


def order_surface(pos, normals):
    """Return indices that walk the airfoil surface as a closed loop:
    upper surface leading edge -> trailing edge, then lower surface trailing edge -> leading edge.
    Works whichever way the normals point (inward or outward)."""
    g1 = np.where(normals[:, 1] < 0)[0]
    g2 = np.where(normals[:, 1] >= 0)[0]
    # The upper-surface group is the one sitting higher on average
    upper, lower = (g1, g2) if pos[g1, 1].mean() > pos[g2, 1].mean() else (g2, g1)
    upper = upper[np.argsort(pos[upper, 0])]     # LE -> TE
    lower = lower[np.argsort(-pos[lower, 0])]    # TE -> LE
    return np.concatenate([upper, lower]), len(upper)


def pressure_force_coefficients(pos, normals, p, u_inf, aoa_rad):
    """Integrate surface pressure (kinematic, p/rho) to pressure-only lift and drag coefficients.
    pos, normals, p: surface nodes only. Returns (cd_p, cl_p)."""
    idx, _ = order_surface(pos, normals)
    P, N, pp = pos[idx], normals[idx], p[idx].ravel()
    # Length "owned" by each node = half of each neighbouring segment on the closed loop
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    L = 0.5 * (seg + np.roll(seg, 1))
    # Make normals point OUT of the body: on average they should point away from the centroid
    sign = np.sign(np.mean(np.sum((P - P.mean(axis=0)) * N, axis=1)))
    n_out = sign * N / np.linalg.norm(N, axis=1, keepdims=True)
    F = -(pp[:, None] * n_out * L[:, None]).sum(axis=0)       # force / rho
    drag_dir = np.array([np.cos(aoa_rad), np.sin(aoa_rad)])
    lift_dir = np.array([-np.sin(aoa_rad), np.cos(aoa_rad)])
    q = 0.5 * u_inf**2
    return F @ drag_dir / q, F @ lift_dir / q


def surface_tangents(pos, normals):
    """Unit tangent at each surface node pointing from the leading edge towards the trailing edge.

    Returned in the original node order, with a boolean mask marking the upper surface.
    """
    idx, n_up = order_surface(pos, normals)
    P = pos[idx]
    t = np.zeros_like(P)
    for part in (slice(0, n_up), slice(n_up, None)):
        Q = P[part]
        d = np.gradient(Q, axis=0)
        t[part] = d / np.linalg.norm(d, axis=1, keepdims=True)
    t[n_up:] *= -1                                 # lower surface was walked TE -> LE
    out, upper = np.empty_like(t), np.zeros(len(pos), bool)
    out[idx], upper[idx[:n_up]] = t, True
    return out, upper


def near_wall_reversal(pos, normals, sdf, velocity, surf):
    """Flag reversed flow next to the wall, as a measure of separation.

    For each surface node, the velocity at the nearest off-wall mesh node (the first cell, about 2 µm
    away) is projected on the local surface tangent pointing from leading to trailing edge; a negative
    value means the near-wall flow runs upstream. Near the leading edge this also flags the short
    stretch between the geometric nose and the stagnation point, so callers should ignore x/c < 0.05.

    Returns (reversed, upper, x) for the surface nodes, in the order of pos[surf].
    """
    from scipy.spatial import cKDTree
    surf = np.asarray(surf, bool).ravel()
    sp, sn = pos[surf], normals[surf]
    t, upper = surface_tangents(sp, sn)
    off = np.where(~surf & (np.asarray(sdf).ravel() > 0))[0]
    _, k = cKDTree(pos[off]).query(sp)
    u_t = np.sum(velocity[off[k], :2] * t, axis=1)
    return u_t < 0, upper, sp[:, 0]


def shape_descriptor(naca, stations=STATIONS):
    """Thickness ratio and camber-line height at the chord stations, for NACA 4- or 5-digit parameters.

    Uses the camber line shipped with `airfrans`, which also generated the dataset geometry.
    Note that for a 4-digit airfoil with P = 0 that camber line is identically zero, whatever M.
    """
    from airfrans.naca_generator import camber_line
    naca = np.asarray(naca, dtype=float)
    if len(naca) == 3:
        params_c, t = naca[:2], naca[2] / 100
    elif len(naca) == 4:
        params_c, t = naca[:3], naca[3] / 100
    else:
        raise ValueError(f'unexpected NACA parameters {naca}')
    y_c, _ = camber_line(params_c, np.array(stations, dtype=float).copy())
    return np.concatenate([[t], y_c]).astype(np.float32)


def descriptor_torch(design, stations):
    """Differentiable shape descriptor of NACA 4-digit designs.

    design: (..., 3) tensor of [M (% chord), P (tenths of chord), XX (% chord)];
    stations: 1-D tensor of chord stations. Returns (..., 1 + len(stations)).

    Matches `airfrans.naca_generator.camber_line` exactly, including its convention that the camber
    line is zero when P = 0 or P = 10. Both branches are evaluated with safe divisors, so gradients
    stay finite on those bounds (torch.where would otherwise back-propagate the 0/0 of the unused
    branch as NaN).
    """
    import torch
    M, P, XX = design[..., 0:1], design[..., 1:2], design[..., 2:3]
    m, p, t = M / 100, P / 10, XX / 100
    x = stations.expand(*design.shape[:-1], len(stations))
    one = torch.ones_like(p)
    p_fore = torch.where(p > 0, p, one)
    p_aft = torch.where(p < 1, p, 0 * one)
    fore = (m / p_fore ** 2) * (2 * p * x - x ** 2)
    aft = (m / (1 - p_aft) ** 2) * ((1 - 2 * p) + 2 * p * x - x ** 2)
    y_c = torch.where(x < p, fore, aft)
    flat = (p <= 0) | (p >= 1)
    y_c = torch.where(flat.expand_as(y_c), torch.zeros_like(y_c), y_c)
    return torch.cat([t, y_c], dim=-1)
