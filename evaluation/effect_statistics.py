"""Episode-separated diagnostic readers; no model or policy updates."""
import numpy as np
from evaluation.probes import ridge_probe


def partition(episode_ids, fit_ids, validation_ids, test_ids):
    groups = [set(map(int, ids)) for ids in [fit_ids, validation_ids, test_ids]]
    if any(not g for g in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError('Calibration, validation and test episodes must be disjoint and nonempty')
    if set(map(int, episode_ids)) != set.union(*groups):
        raise ValueError('Every episode must belong to exactly one diagnostic split')
    masks = tuple(np.isin(episode_ids, list(g)) for g in groups)
    if not all(m.any() for m in masks):
        raise ValueError('Empty diagnostic split')
    return masks


def metrics(prediction, truth, calibration, scale):
    pred, y, cal = map(lambda a: np.asarray(a, np.float64), [prediction, truth, calibration])
    if pred.shape != y.shape or not all(np.isfinite(a).all() for a in (pred, y, cal, scale)):
        raise ValueError('Invalid prediction/target values')
    err = np.square((pred-y)/scale).mean()
    mean_err = np.square((y-cal.mean(0))/scale).mean()
    zero_err = np.square(y/scale).mean()
    return dict(mse=float(np.square(pred-y).mean()), standardized_mse=float(err),
                target_calibration_variance=float(np.var(cal/scale, axis=0).mean()),
                informative_target=bool(np.var(cal/scale, axis=0).mean() > 1e-12),
                mean_reference_standardized_mse=float(mean_err),
                zero_reference_standardized_mse=float(zero_err),
                skill_vs_mean=float(1-err/mean_err) if mean_err > 1e-12 else None,
                skill_vs_zero=float(1-err/zero_err) if zero_err > 1e-12 else None)


def fit_reader(features, target, splits, scale, alphas):
    fit, val, test = splits
    x, y = np.asarray(features, np.float64), np.asarray(target, np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError('Reader needs aligned [sample, coordinate] arrays')
    # Same target scaling in fitting, regularization selection, and scoring.
    pred, diag, _ = ridge_probe(x[fit], y[fit]/scale, x[val], y[val]/scale, x[test], alphas)
    pred = pred.astype(np.float64)*scale
    return pred, dict(**metrics(pred, y[test], y[fit], scale), **diag)


def target_edges(cube):
    """Bit zero denotes target replacement; bits 1..3 denote partners."""
    if cube.shape[1] != 16:
        raise ValueError('Expected complete four-agent assignment cube')
    return cube[:, 0::2] - cube[:, 1::2]


def single_partner_interactions(edges):
    """Matches c({i}) = g({i}) - g(empty), with fixed direction."""
    return np.stack([edges[:, k]-edges[:, 0] for k in (1, 2, 4)], axis=1)
