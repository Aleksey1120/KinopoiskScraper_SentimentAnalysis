import os
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.utils import compute_class_weight
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from nn_classifier.early_stopping import EarlyStopping
from nn_classifier.options.train_options import TrainOptions
from nn_classifier.model import get_model_and_tokenizer
from nn_classifier.datasets import LabeledDataset, Fetcher
from nn_classifier.metrics_evaluator import MetricsEvaluator

from transformers import logging

logging.set_verbosity_error()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log_metrics(train_metrics, validate_metrics, train_writer, validate_writer, iter_number):
    for k, v in train_metrics.items():
        train_writer.add_scalar(k, v, iter_number + 1)
    for k, v in validate_metrics.items():
        validate_writer.add_scalar(k, v, iter_number + 1)


def train_step(model, optimizer, loss_function, batch, device, fp16):
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    optimizer.zero_grad()

    input_id, mask, train_label = batch
    mask = mask.to(device)
    input_id = input_id.squeeze(1).to(device)
    train_label = train_label.to(device)

    with torch.cuda.amp.autocast(enabled=fp16):
        output = model(input_id, attention_mask=mask).logits
        batch_loss = loss_function(output, train_label)

    if fp16:
        scaler.scale(batch_loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        batch_loss.backward()
        optimizer.step()
    return batch_loss.cpu().detach(), output.cpu().detach(), train_label.cpu().detach()


def validate_step(model, loss_function, batch, device):
    with torch.no_grad():
        model.eval()
        input_id, mask, val_label = batch
        mask = mask.to(device)
        input_id = input_id.squeeze(1).to(device)
        val_label = val_label.to(device)
        output = model(input_id, attention_mask=mask).logits
        batch_loss = loss_function(output, val_label)
    return batch_loss.cpu().detach(), output.cpu().detach(), val_label.cpu().detach()


def print_epoch_metrics(start_iter, end_iter, elapsed_time, train_metrics,
                        validate_metrics):
    iters = f'{start_iter:>6} - {end_iter:<6}'
    metric_columns = []
    for metric_name in train_metrics.keys():
        metric_columns.append(f'{train_metrics[metric_name]:^26.3f}|')
        metric_columns.append(f'{validate_metrics[metric_name]:^26.3f}|')
    print(f'|{iters}|{elapsed_time:^10.2f}|' + ''.join(metric_columns))


def print_result_table_headers(required_metrics):
    metric_headers = []
    for metric in required_metrics:
        metric_headers.append(f'{f"Train {metric}":^26}|')
        metric_headers.append(f'{f"Validate {metric}":^26}|')
    print(f'|     Iters     |   Time   |' + ''.join(metric_headers))


def train(opt, model, train_fetcher: Fetcher, validate_loader, optimizer, scheduler, loss_function, device):
    start_time = time.time()
    iter_start_time = time.time()
    early_stopping = EarlyStopping(model, opt)
    train_metrics_evaluator = MetricsEvaluator(opt.metrics)
    if opt.tb_dir is not None:
        train_writer = SummaryWriter(log_dir=os.path.join(opt.tb_dir, opt.tb_comment, 'train'))
        validate_writer = SummaryWriter(log_dir=os.path.join(opt.tb_dir, opt.tb_comment, 'validate'))

    if opt.verbose >= 2:
        print_result_table_headers(opt.metrics)

    for iter_number in range(opt.niter):
        train_batch = train_fetcher.load()
        train_loss, train_output, train_label = train_step(model, optimizer, loss_function, train_batch, device,
                                                           opt.fp16)
        train_metrics_evaluator.append(train_loss, train_output, train_label)

        if opt.save_every and (iter_number + 1) % opt.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(opt.output_dir, f'{opt.model_comment}_iter_{iter_number}.bin'))

        if opt.verbose >= 2 and (iter_number + 1) % opt.print_every == 0:
            validate_metrics_evaluator = MetricsEvaluator(opt.metrics)
            for validate_batch in validate_loader:
                validate_loss, validate_output, validate_label = validate_step(model, loss_function, validate_batch,
                                                                               device)
                validate_metrics_evaluator.append(validate_loss, validate_output, validate_label)
            train_metrics = train_metrics_evaluator.compute_metrics()
            validate_metrics = validate_metrics_evaluator.compute_metrics()
            print_epoch_metrics(iter_number + 1 - opt.print_every,
                                iter_number,
                                time.time() - iter_start_time,
                                train_metrics,
                                validate_metrics)
            if opt.tb_dir is not None:
                log_metrics(train_metrics, validate_metrics, train_writer, validate_writer, iter_number)
            iter_start_time = time.time()
            train_metrics_evaluator = MetricsEvaluator(opt.metrics)

            if early_stopping(validate_metrics):
                break
            scheduler.step()

    early_stopping.rename_file()
    if opt.verbose >= 1:
        print(f'Total fitting time: {time.time() - start_time:.2f}')


def main():
    train_options = TrainOptions()
    opt = train_options.get_options()
    if opt.seed is not None:
        set_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    model, tokenizer = get_model_and_tokenizer(opt.model_name_or_path, opt.checkpoint_path, cache_dir=opt.cache_dir)

    train_df = pd.read_csv(opt.train_data)
    validate_df = pd.read_csv(opt.validate_data)
    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(train_df['review_type']),
                                         y=train_df['review_type'])
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    train_dataset = LabeledDataset(train_df['review_text'],
                                   train_df['review_type'],
                                   tokenizer,
                                   opt.max_length,
                                   opt.use_cache)
    validate_dataset = LabeledDataset(validate_df['review_text'],
                                      validate_df['review_type'],
                                      tokenizer,
                                      opt.max_length,
                                      opt.use_cache)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    valid_loader = torch.utils.data.DataLoader(validate_dataset,
                                               batch_size=opt.batch_size,
                                               num_workers=opt.num_workers)

    train_fetcher = Fetcher(train_loader)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)
    loss_function = nn.CrossEntropyLoss(weight=class_weights if opt.balanced else None,
                                        label_smoothing=opt.label_smoothing).to(device)

    if opt.verbose >= 1:
        print(f'Train size: {train_df.shape[0]} Validate size: {validate_df.shape[0]}')
        train_options.print_options()
    train(opt, model, train_fetcher, valid_loader, optimizer, scheduler, loss_function, device)


if __name__ == '__main__':
    main()
