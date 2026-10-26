"""Held-out evaluation with patient-cluster resampling and validation-only calibration."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import log_loss
from .core import metrics, coverage_curve


def calibrate_temperature(labels, scores):
    logits = logit(np.clip(scores, 1e-6, 1-1e-6))
    fit = minimize_scalar(lambda t: log_loss(labels, expit(logits / t), labels=[0, 1]),
                          bounds=(.1, 10.), method='bounded')
    if not fit.success:
        raise ValueError('Validation calibration failed')
    return float(fit.x)


def calibrated(scores, temperature):
    return expit(logit(np.clip(scores, 1e-6, 1-1e-6)) / temperature)


def bootstrap(rows, scores, seed=42, repeats=200):
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['patient_id'], []).append(i)
    patients = list(groups)
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ('auroc', 'brier', 'sensitivity', 'specificity')}
    labels = np.array([r['label'] for r in rows])
    for _ in range(repeats):
        ids = [i for patient in rng.choice(patients, len(patients), replace=True) for i in groups[patient]]
        result = metrics(labels[ids], np.asarray(scores)[ids])
        for key in values:
            if result[key] is not None:
                values[key].append(result[key])
    return {key: dict(lower=float(np.quantile(vals, .025)), upper=float(np.quantile(vals, .975)),
                      valid_resamples=len(vals), requested_resamples=repeats) if vals else None
            for key, vals in values.items()}


def evaluate_test(rows, scores, seed=42):
    y = [r['label'] for r in rows]
    grouped = {}
    for row, score in zip(rows, scores):
        grouped.setdefault(row['patient_id'], []).append((row['label'], float(score)))
    # Explicit exploratory patient endpoint: any malignant image; maximum image score.
    patient_y = [max(v[0] for v in group) for group in grouped.values()]
    patient_p = [max(v[1] for v in group) for group in grouped.values()]
    density = {}
    for d in ('1', '2', '3', '4', ''):
        ids = [i for i, row in enumerate(rows) if row['density'] == d]
        if ids:
            density[d or 'unknown'] = metrics(np.array(y)[ids], np.array(scores)[ids])
    return dict(image_metrics=metrics(y, scores), patient_metrics=metrics(patient_y, patient_p),
                patient_aggregation='max image score / any malignant image label',
                patient_cluster_95ci=bootstrap(rows, scores, seed),
                coverage_curve=coverage_curve(y, scores), density_subgroups=density)
