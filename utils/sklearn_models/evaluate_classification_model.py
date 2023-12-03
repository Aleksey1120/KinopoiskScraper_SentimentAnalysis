import sklearn.metrics as metrics
import numpy as np

_metrics_proba_dict = {
    'roc_auc_ovo': lambda y_real, y_pred: metrics.roc_auc_score(y_real, y_pred, multi_class='ovo'),
    'roc_auc_ovr': lambda y_real, y_pred: metrics.roc_auc_score(y_real, y_pred, multi_class='ovr'),
    'ap_ovo': lambda y_real, y_pred: metrics.roc_auc_score(y_real, y_pred, average='macro', multi_class='ovo'),
    'ap_ovr': lambda y_real, y_pred: metrics.roc_auc_score(y_real, y_pred, average='macro', multi_class='ovr'),
    'accuracy': lambda y_real, y_pred: metrics.accuracy_score(y_real, np.argmax(y_pred, axis=1)),
    'f1_macro': lambda y_real, y_pred: metrics.f1_score(y_real, np.argmax(y_pred, axis=1), average='macro'),
    'cm': lambda y_real, y_pred: metrics.confusion_matrix(y_real, np.argmax(y_pred, axis=1))
}


def get_metrics_list():
    return list(sorted(_metrics_proba_dict.keys()))


def evaluate_classification_model(model, x, y, metrics: list[str]):
    pred = model.predict_proba(x)
    return {m: _metrics_proba_dict[m](y, pred) for m in metrics}
