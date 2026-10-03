import numpy as np
import pytest
import torch
from airfrans.naca_generator import camber_line, naca_generator

from airfrans_surrogate.data import family, parse_name
from airfrans_surrogate.geometry import (STATIONS, descriptor_torch, order_surface,
                                         pressure_force_coefficients, shape_descriptor,
                                         near_wall_reversal, surface_tangents)
from airfrans_surrogate.metrics import comparable_pairs, pair_accuracy
from airfrans_surrogate.models import ForceMLP, PointwiseMLP
from airfrans_surrogate.optimize import CallCounter, is_feasible, multistart


def synthetic_surface(params=(2.0, 4.0, 12.0), n=400):
    """Closed NACA surface with outward node normals, in shuffled order (as in the dataset)."""
    P = naca_generator(np.array(params), nb_samples=n, verbose=False)[:-1]
    t = np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)
    N = np.c_[t[:, 1], -t[:, 0]]
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    if np.mean(np.sum((P - P.mean(0)) * N, axis=1)) < 0:
        N = -N
    perm = np.random.default_rng(0).permutation(len(P))
    return P[perm], N[perm]


def test_parse_name_and_family():
    u, a, naca = parse_name('airFoil2D_SST_36.622_11.319_3.941_5.424_1.0_16.283')
    assert (u, a, naca) == (36.622, 11.319, [3.941, 5.424, 1.0, 16.283])
    assert family('airFoil2D_SST_50.0_2.0_2.0_4.0_12.0') == '4-digit'
    assert family('airFoil2D_SST_36.622_11.319_3.941_5.424_1.0_16.283') == '5-digit'


@pytest.mark.parametrize('flip', [False, True])
def test_pressure_integration_exact_for_linear_field(flip):
    # For p = y, the pressure force is -oint p n ds = -A y_hat (A = enclosed area), whichever way normals point
    P, N = synthetic_surface()
    if flip:
        N = -N
    idx, _ = order_surface(P, N)
    Q = P[idx]
    area = 0.5 * abs(np.sum(Q[:, 0] * np.roll(Q[:, 1], -1) - np.roll(Q[:, 0], -1) * Q[:, 1]))
    cd, cl = pressure_force_coefficients(P, N, P[:, 1], u_inf=np.sqrt(2.0), aoa_rad=0.0)
    assert abs(cd) < 1e-6
    assert cl == pytest.approx(-area, rel=2e-3)


def test_surface_tangents_point_downstream():
    P, N = synthetic_surface()
    t, upper = surface_tangents(P, N)
    mid = (P[:, 0] > 0.2) & (P[:, 0] < 0.8)
    assert np.all(t[mid, 0] > 0.9)
    assert upper.sum() > 0 and (~upper).sum() > 0


def test_near_wall_reversal_detects_upstream_flow():
    P, N = synthetic_surface()
    t, upper = surface_tangents(P, N)
    off = P + 1e-5 * N                                     # first off-wall layer
    vel = t.copy()
    sep = (~upper) & (P[:, 0] > 0.6)                       # reversed flow on the aft lower surface
    vel[sep] *= -1
    pos = np.r_[P, off]
    normals = np.r_[N, np.zeros_like(N)]
    sdf = np.r_[np.zeros(len(P)), np.full(len(P), 1e-5)]
    velocity = np.r_[np.zeros_like(vel), vel]
    surf = np.r_[np.ones(len(P), bool), np.zeros(len(P), bool)]
    rev, up, x = near_wall_reversal(pos, normals, sdf, velocity, surf)
    assert np.array_equal(rev, sep) and np.array_equal(up, upper)


def test_shape_descriptor_both_families():
    g4 = shape_descriptor([2.0, 4.0, 12.0])
    g5 = shape_descriptor([3.941, 5.424, 1.0, 16.283])
    assert g4.shape == g5.shape == (1 + len(STATIONS),)
    assert g4[0] == pytest.approx(0.12) and g5[0] == pytest.approx(0.16283)
    assert np.all(shape_descriptor([2.7, 0.0, 12.0])[1:] == 0)   # library convention at P = 0


def test_descriptor_torch_matches_library_including_bounds():
    rng = np.random.default_rng(0)
    d = rng.uniform([0.0, 0.0, 6.0], [7.0, 7.0, 20.0], size=(300, 3))
    d[:30, 1] = 0.0            # P exactly on the lower bound
    d[30:40, 1] = 1e-3         # just above it
    ours = descriptor_torch(torch.tensor(d, dtype=torch.float64),
                            torch.tensor(STATIONS, dtype=torch.float64)).numpy()
    lib = np.stack([np.r_[x[2] / 100, camber_line(x[:2], STATIONS.copy())[0]] for x in d])
    assert np.abs(ours - lib).max() < 1e-12


def test_descriptor_torch_gradient_finite_at_P0():
    d = torch.tensor([[3.0, 0.0, 12.0], [3.0, 4.0, 12.0]], dtype=torch.float64, requires_grad=True)
    g = descriptor_torch(d, torch.tensor(STATIONS, dtype=torch.float64))
    (gd,) = torch.autograd.grad(g.sum(), d)
    assert torch.isfinite(gd).all()


def test_models_shapes():
    assert PointwiseMLP(20, 4, 32, 3)(torch.zeros(5, 20)).shape == (5, 4)
    assert ForceMLP(15, 16)(torch.zeros(5, 15)).shape == (5, 2)


def test_pair_accuracy():
    aoa, re = np.array([0.0, 0.5, 5.0]), np.array([3e6, 3.1e6, 3e6])
    i, j, near = comparable_pairs(aoa, re, 1.0, 0.5e6)
    assert near.tolist() == [True, False, False]
    true, pred = np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 0.0])
    acc, acc_d, n_d = pair_accuracy(true, pred, i, j, np.ones_like(near), 0.05)
    assert acc == pytest.approx(1 / 3) and n_d == 3


def test_multistart_returns_best_feasible_and_counts_calls():
    calls = CallCounter()
    fg = calls.wrap(lambda x: (float((x[0] - 0.3) ** 2), np.array([2 * (x[0] - 0.3)])))
    con = calls.wrap(lambda x: x[0] - 0.5)                    # x >= 0.5 is binding
    cons = [dict(type='ineq', fun=con, jac=lambda x: np.array([1.0]))]
    r = multistart(fg, [(0.0, 1.0)], 4, seed=0, method='SLSQP', constraints=cons)
    assert r['x'][0] == pytest.approx(0.5, abs=1e-6)
    assert r['accepted'].all()
    assert calls.reset() > 0 and calls.n == 0


def test_multistart_rejects_infeasible():
    fg = lambda x: (float(x[0]), np.array([1.0]))
    cons = [dict(type='ineq', fun=lambda x: x[0] - 3.0, jac=lambda x: np.array([1.0]))]   # impossible within bounds
    with pytest.raises(RuntimeError):
        multistart(fg, [(0.0, 1.0)], 3, seed=0, method='SLSQP', constraints=cons)
    assert not is_feasible([0.5], [(0.0, 1.0)], cons)
