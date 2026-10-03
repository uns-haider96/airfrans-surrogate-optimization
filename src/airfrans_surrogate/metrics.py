"""Ranking metrics for comparing designs."""

import numpy as np


def comparable_pairs(aoa, re, max_daoa, max_dre):
    """All test-case pairs (i < j) and a mask of those at comparable operating conditions.

    Note that a pair within 1 degree can still differ in lift by ~0.1 from incidence alone, so this is
    a proxy for "same condition, different shape", not an exact match.
    """
    aoa, re = np.asarray(aoa), np.asarray(re)
    i, j = np.triu_indices(len(aoa), 1)
    near = (np.abs(aoa[i] - aoa[j]) <= max_daoa) & (np.abs(re[i] - re[j]) <= max_dre)
    return i, j, near


def pair_accuracy(true, pred, i, j, mask, min_rel):
    """Fraction of pairs ordered correctly: over `mask`, and over the distinguishable pairs within it
    (true values differing by more than `min_rel`). Returns (accuracy, accuracy_distinguishable, n_distinguishable)."""
    true, pred = np.asarray(true), np.asarray(pred)
    ok = np.sign(true[i] - true[j]) == np.sign(pred[i] - pred[j])
    dist = np.abs(true[i] - true[j]) > min_rel * np.maximum(np.abs(true[i]), np.abs(true[j]))
    frac = lambda sel: ok[sel].mean() if sel.any() else np.nan
    return frac(mask), frac(mask & dist), int((mask & dist).sum())
