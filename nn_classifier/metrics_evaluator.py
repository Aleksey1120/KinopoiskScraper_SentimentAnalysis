import collections

import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, \
    average_precision_score

_metrics_functions = {
    'loss': lambda x: torch.mean(torch.tensor(x['losses'])).item(),
    'accuracy': lambda x: accuracy_score(x['labels'],
                                         torch.argmax(x['outputs'], dim=1)),
    'precision': lambda x: precision_score(x['labels'],
                                           torch.argmax(x['outputs'], dim=1),
                                           average='macro', zero_division=0),
    'recall': lambda x: recall_score(x['labels'],
                                     torch.argmax(x['outputs'], dim=1),
                                     average='macro', zero_division=0),
    'f1': lambda x: f1_score(x['labels'],
                             torch.argmax(x['outputs'], dim=1),
                             average='macro', zero_division=0),
    'average_precision': lambda x: average_precision_score(x['labels'],
                                                           torch.nn.Softmax(1)(x['outputs'].float()),
                                                           average='macro'),
    'roc_auc': lambda x: roc_auc_score(x['labels'],
                                       torch.nn.Softmax(1)(x['outputs'].float()),
                                       multi_class='ovo'),
}


class MetricsEvaluator:
    def __init__(self, required_metrics):
        self.records = collections.defaultdict(list)
        self.required_metrics = required_metrics

    def append(self, loss, output, label):
        self.records['losses'].append(loss)
        self.records['outputs'].append(output)
        self.records['labels'].append(label)

    def compute_metrics(self):
        records = dict()
        records['losses'] = self.records['losses']
        records['outputs'] = torch.cat(self.records['outputs'])
        records['labels'] = torch.cat(self.records['labels'])
        return {m: _metrics_functions[m](records) for m in self.required_metrics}
