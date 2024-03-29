from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import compute_sample_weight
import numpy as np


class AveragingModels(BaseEstimator, ClassifierMixin):
    def __init__(self, models, transformer=None, meta_classifier=None, drop_one_dim=False):
        self.models = models
        self.meta_classifier = meta_classifier
        self.transformer = transformer
        self.drop_one_dim = drop_one_dim

    def set_meta(self, meta_classifier):
        self.meta_classifier = meta_classifier

    def set_models(self, models):
        self.models = models

    def set_transformer(self, transformer):
        self.transformer = transformer

    def fit_base(self, x, y, fit_transformer=False, balanced=False):
        if self.transformer is not None:
            if fit_transformer:
                x = self.transformer.fit_transform(x)
            else:
                x = self.transformer.transform(x)
        if balanced:
            weights = compute_sample_weight(
                class_weight='balanced',
                y=y
            )
            for model in self.models:
                model.fit(x, y, sample_weight=weights)
        else:
            for model in self.models:
                model.fit(x, y)
        return self

    def predict_proba_base(self, x):
        if self.transformer is not None:
            x = self.transformer.transform(x)
        base_predictions = [
            model.predict_proba(x)[:, :-1] if self.drop_one_dim else model.predict_proba(x)
            for model in self.models
        ]
        return np.array(base_predictions)

    def fit_meta(self, x, y):
        assert self.meta_classifier is not None, 'Meta classifier must not be None.'
        self.meta_classifier.fit(x, y)
        return self

    def fit(self, x, y, fit_transformer=False, balanced=False):
        self.fit_base(x, y, fit_transformer=fit_transformer, balanced=balanced)
        if self.meta_classifier is not None:
            predicted = self.predict_proba_base(x)
            self.fit_meta(np.column_stack(predicted), y)
        return self

    def predict(self, x):
        base_predictions = self.predict_proba_base(x)

        if self.meta_classifier is not None:
            return self.meta_classifier.predict(np.column_stack(base_predictions))
        else:
            return np.mean(base_predictions, axis=0)

    def predict_proba(self, x):
        base_predictions = self.predict_proba_base(x)

        if self.meta_classifier is not None:
            return self.meta_classifier.predict_proba(np.column_stack(base_predictions))
        else:
            return np.mean(base_predictions, axis=0)
