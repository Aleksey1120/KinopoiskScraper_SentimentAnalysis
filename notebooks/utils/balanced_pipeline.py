from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight


class BalancedPipeline(Pipeline):
    def __init__(self, model_names: list[str] | str, steps, *, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)
        assert len(model_names), 'No model name'
        if isinstance(model_names, str):
            self.model_names = [model_names]
        elif isinstance(model_names, list):
            self.model_names = model_names
        else:
            raise 'Model names must be str or list[str]'

    def fit(self, X, y=None, **fit_params):
        assert y is not None, 'y must be not None'
        sample_weight = compute_sample_weight('balanced', y)
        for model in self.model_names:
            fit_params[f'{model}__sample_weight'] = sample_weight
        return super().fit(X, y, **fit_params)
