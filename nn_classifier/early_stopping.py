import torch
import operator
import os


class EarlyStopping:

    def __init__(self, model, options):
        self.trigger_times = 0
        self.best_value = torch.inf if options.target_metric == 'loss' else -torch.inf
        self.target_metric = options.target_metric
        self.save_best_model = options.save_best_model
        self.model = model
        self.early_stop = options.early_stop

        self.comparison_function = operator.gt if options.target_metric != 'loss' else operator.lt
        self.output_dir = options.output_dir
        self.model_comment = options.model_comment

    def __call__(self, validate_metrics):
        computed_metrics = validate_metrics.compute_metrics()
        if self.comparison_function(computed_metrics[self.target_metric], self.best_value):
            self.best_value = computed_metrics[self.target_metric]
            self.trigger_times = 0
            if self.save_best_model:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_dir, f'{self.model_comment}.bin.train'))
            return False
        if self.early_stop is None:
            return False
        self.trigger_times += 1
        if self.trigger_times >= self.early_stop:
            return True

    def rename_file(self):
        if self.save_best_model:
            if os.path.exists(os.path.join(self.output_dir, f'{self.model_comment}.bin')):
                os.remove(os.path.join(self.output_dir, f'{self.model_comment}.bin'))
            os.rename(os.path.join(self.output_dir, f'{self.model_comment}.bin.train'),
                      os.path.join(self.output_dir, f'{self.model_comment}.bin'))
