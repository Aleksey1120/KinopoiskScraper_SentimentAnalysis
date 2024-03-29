import sklearn.metrics as metrics
import numpy as np
from sklearn.utils import compute_sample_weight

_metrics_proba_dict = {
    'roc_auc_ovo': lambda y_real, y_pred, sample_weights=None:
    metrics.roc_auc_score(y_real, y_pred, multi_class='ovo', sample_weight=sample_weights),
    'roc_auc_ovr': lambda y_real, y_pred, sample_weights=None:
    metrics.roc_auc_score(y_real, y_pred, multi_class='ovr', sample_weight=sample_weights),
    'average_precision': lambda y_real, y_pred, sample_weights=None:
    metrics.average_precision_score(y_real, y_pred, average='macro', sample_weight=sample_weights),
    'accuracy': lambda y_real, y_pred, sample_weights=None:
    metrics.accuracy_score(y_real, np.argmax(y_pred, axis=1), sample_weight=sample_weights),
    'f1_macro': lambda y_real, y_pred, sample_weights=None:
    metrics.f1_score(y_real, np.argmax(y_pred, axis=1), average='macro', sample_weight=sample_weights),
    'cm': lambda y_real, y_pred, sample_weights=None:
    metrics.confusion_matrix(y_real, np.argmax(y_pred, axis=1), sample_weight=sample_weights)
}


def get_metrics_list():
    return list(sorted(_metrics_proba_dict.keys()))


def evaluate_classification_model(model, x, y, metrics: list[str]):
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=y
    )
    result = {}
    pred = model.predict_proba(x)
    for m in metrics:
        if m.startswith('balanced_'):
            result[m] = _metrics_proba_dict[m.replace('balanced_', '')](y, pred, sample_weights)
        else:
            result[m] = _metrics_proba_dict[m](y, pred)
    return result
