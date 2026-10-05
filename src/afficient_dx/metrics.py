import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import auc, average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from afficient_dx.data import require_two_classes

def fit_calibrator(y, logits, method):
    if method == 'none':
        return {'method': 'none', 'slope': 1.0, 'intercept': 0.0}
    require_two_classes(y, 'probability calibration')
    y, z = (np.asarray(y, dtype=float), np.asarray(logits, dtype=float))

    def objective(theta):
        a, b = theta
        value = a * z + b
        delta = expit(value) - y
        return (np.mean(np.logaddexp(0, value) - y * value), np.array([np.mean(delta * z), np.mean(delta)]))
    fit = minimize(objective, [1.0, 0.0], jac=True, method='L-BFGS-B', bounds=[(1e-06, 100.0), (-100.0, 100.0)])
    if not fit.success or not np.isfinite(fit.x).all():
        raise RuntimeError(f'Probability calibration failed: {fit.message}')
    return {'method': 'monotone_platt', 'slope': float(fit.x[0]), 'intercept': float(fit.x[1]), 'fit_records': len(y), 'fit_split': 'calibration', 'class_weighting': 'none'}

def apply_calibrator(logits, calibration):
    return expit(calibration['slope'] * logits + calibration['intercept'])

def threshold_from_validation(y, p, method='youden', target_sensitivity=0.9):
    if method == 'fixed':
        return 0.5
    require_two_classes(y, 'threshold fitting')
    fpr, tpr, thresholds = roc_curve(y, p, drop_intermediate=False)
    valid = np.isfinite(thresholds) & (thresholds >= 0) & (thresholds <= 1)
    fpr, tpr, thresholds = (fpr[valid], tpr[valid], thresholds[valid])
    thresholds = np.r_[np.nextafter(float(np.max(p)), np.inf), thresholds]
    fpr, tpr = (np.r_[0.0, fpr], np.r_[0.0, tpr])
    if method == 'sensitivity':
        candidates = np.flatnonzero(tpr >= target_sensitivity)
        return float(thresholds[candidates[0]])
    if method != 'youden':
        raise ValueError(f'Unknown threshold method: {method}')
    return float(thresholds[np.argmax(tpr - fpr)])

def reliability_table(y, p, bins=10):
    assignment = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for i in range(bins):
        mask = assignment == i
        rows.append({'lower': i / bins, 'upper': (i + 1) / bins, 'n': int(mask.sum()), 'mean_probability': float(p[mask].mean()) if mask.any() else None, 'af_fraction': float(y[mask].mean()) if mask.any() else None})
    return rows

def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None

def scores(y, p, threshold, bins=10, ranking_scores=None):
    y, p = (np.asarray(y), np.asarray(p, dtype=float))
    if len(y) == 0 or len(y) != len(p) or (not np.isfinite(p).all()):
        raise ValueError('Nonempty, aligned finite predictions are required')
    if not np.isin(y, [0, 1]).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Expected binary targets and probabilities in [0,1]')
    ranking = p if ranking_scores is None else np.asarray(ranking_scores, dtype=float)
    if ranking.shape != p.shape or not np.isfinite(ranking).all():
        raise ValueError('Ranking scores must be aligned and finite')
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    sensitivity, specificity = (divide(tp, tp + fn), divide(tn, tn + fp))
    curve_p, curve_r, _ = precision_recall_curve(y, ranking) if both else (None, None, None)
    reliability = reliability_table(y, p, bins)
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    return {'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()), 'auroc': float(roc_auc_score(y, ranking)) if both else None, 'auprc': float(average_precision_score(y, ranking)) if both else None, 'ranking_score': 'raw_logit' if ranking_scores is not None else 'probability', 'auprc_method': 'average_precision', 'random_auprc': float(y.mean()), 'pr_trapezoid_auc': float(auc(curve_r, curve_p)) if both else None, 'accuracy': float((tp + tn) / len(y)), 'balanced_accuracy': (sensitivity + specificity) / 2 if both else None, 'sensitivity': sensitivity, 'specificity': specificity, 'ppv': divide(tp, tp + fp), 'npv': divide(tn, tn + fn), 'f1': divide(2 * tp, 2 * tp + fp + fn), 'false_discovery_rate': divide(fp, tp + fp), 'false_positives_per_1000_ecgs': float(1000 * fp / len(y)), 'missed_af_per_1000_ecgs': float(1000 * fn / len(y)), 'referral_fraction': float((tp + fp) / len(y)), 'brier': float(np.mean((p - y) ** 2)), 'log_loss': float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))), 'ece': float(sum((row['n'] / len(y) * abs(row['mean_probability'] - row['af_fraction']) for row in reliability if row['n']))), 'ece_bins': bins, 'threshold': float(threshold), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

def cluster_resamples(y, patient_id, n, seed):
    patient_id = np.asarray(patient_id)
    patients = np.unique(patient_id)
    lookup = [np.flatnonzero(patient_id == patient) for patient in patients]
    rng = np.random.default_rng(seed)
    valid = 0
    for _ in range(max(20 * n, 1)):
        if valid >= n:
            return
        idx = np.concatenate([lookup[i] for i in rng.integers(0, len(patients), len(patients))])
        if len(np.unique(y[idx])) < 2:
            continue
        valid += 1
        yield idx

def bootstrap_ci(y, p, patient_id, n, seed, threshold=0.5, bins=10, ranking_scores=None):
    keys = ['auroc', 'auprc', 'sensitivity', 'specificity', 'ppv', 'f1', 'brier', 'ece']
    samples = {key: [] for key in keys}
    valid = 0
    for idx in cluster_resamples(y, patient_id, n, seed):
        metrics = scores(y[idx], p[idx], threshold, bins, ranking_scores[idx] if ranking_scores is not None else None)
        valid += 1
        for key in keys:
            if metrics[key] is not None:
                samples[key].append(metrics[key])
    return {'method': 'patient_cluster_percentile', 'requested': n, 'valid': valid, 'scope': 'conditional on this fitted model, calibrator and fixed threshold', 'intervals': {key: np.percentile(v, [2.5, 97.5]).tolist() if v else None for key, v in samples.items()}, 'valid_per_metric': {key: len(v) for key, v in samples.items()}}

def threshold_stability(y, p, patients, args):
    values = [threshold_from_validation(y[idx], p[idx], args.threshold_method, args.target_sensitivity) for idx in cluster_resamples(y, patients, args.threshold_bootstrap, args.split_seed + 1901)]
    return {'requested': args.threshold_bootstrap, 'valid': len(values), 'median': float(np.median(values)) if values else None, 'percentile_95_interval': np.percentile(values, [2.5, 97.5]).tolist() if values else None, 'scope': 'threshold-subset patient resampling; descriptive, no test retuning', 'bootstrap_thresholds': values}
