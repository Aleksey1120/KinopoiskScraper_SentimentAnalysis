import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from nn_classifier.options.test_options import TestOptions
from nn_classifier.model import get_model_and_tokenizer
from nn_classifier.datasets import LabeledDataset
from nn_classifier.metrics_evaluator import MetricsEvaluator

from transformers import logging

logging.set_verbosity_error()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def print_metrics(metrics):
    print(f'{"Metric":^20}{"Value":^9}')
    for m, v in metrics.items():
        print(f'{m:<20}{v:<9.3f}')


def test(opt, model, test_loader, loss_function, device):
    model.eval()
    metrics_evaluator = MetricsEvaluator(opt.metrics)
    with torch.no_grad():
        for batch in test_loader:
            input_id, mask, label = batch
            mask = mask.to(device)
            input_id = input_id.squeeze(1).to(device)
            label = label.to(device)
            label = torch.zeros(label.shape[0], 3,
                                device=device).scatter_(1, label.unsqueeze(1).type(torch.int64), 1.0)

            with torch.cuda.amp.autocast(enabled=opt.fp16, dtype=torch.float16):
                output = model(input_id, attention_mask=mask).logits
                batch_loss = loss_function(output, label)
            metrics_evaluator.append(batch_loss.cpu().detach(),
                                     output.cpu().detach(),
                                     label.cpu().detach())
    computed_metrics = metrics_evaluator.compute_metrics()
    print_metrics(computed_metrics)


def main():
    train_options = TestOptions()
    opt = train_options.get_options()
    set_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    model, tokenizer = get_model_and_tokenizer(opt.model_name_or_path, opt.checkpoint_path, cache_dir=opt.cache_dir)
    loss_function = nn.CrossEntropyLoss()

    test_df = pd.read_csv(opt.test_data)
    test_dataset = LabeledDataset(test_df['review_text'], test_df['review_type'], tokenizer, opt.max_length)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=True)

    model = model.to(device)
    loss_function = loss_function.to(device)
    test(opt, model, test_loader, loss_function, device)


if __name__ == '__main__':
    main()
