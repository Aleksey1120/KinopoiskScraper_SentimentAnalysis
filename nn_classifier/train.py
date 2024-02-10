import os
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from nn_classifier.options.train_options import TrainOptions
from nn_classifier.model import get_model_and_tokenizer
from nn_classifier.datasets import TrainDataset, Fetcher

from transformers import logging

logging.set_verbosity_error()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_step(model, optimizer, loss_function, batch, device):
    model.train()
    optimizer.zero_grad()

    input_id, mask, train_label = batch
    mask = mask.to(device)
    input_id = input_id.squeeze(1).to(device)
    train_label = train_label.to(device)
    train_label = torch.zeros(train_label.shape[0],
                              3, device=device).scatter_(1, train_label.unsqueeze(1).type(torch.int64), 1.0)
    output = model(input_id, attention_mask=mask).logits
    batch_loss = loss_function(output, train_label)

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
        val_label = torch.zeros(val_label.shape[0], 3,
                                device=device).scatter_(1, val_label.unsqueeze(1).type(torch.int64), 1.0)
        output = model(input_id, attention_mask=mask).logits
        batch_loss = loss_function(output, val_label)
    return batch_loss.cpu().detach(), output.cpu().detach(), val_label.cpu().detach()


def print_epoch_metrics(start_iter, end_iter, elapsed_time, train_loss, validate_loss):
    iters = f'{start_iter} - {end_iter}'
    print(f'|{iters:^15}|{elapsed_time: ^10.2f}|{train_loss:^12.3f}|{validate_loss:^15.3f}|')


def print_result_table_header():
    print(f'|     Iters     |   Time   | Train loss | Validate loss |')


def train(opt, model, train_fetcher: Fetcher, validate_loader, optimizer, loss_function, device):
    train_losses = []
    validate_losses = []
    start_time = time.time()

    if opt.verbose >= 2:
        print_result_table_header()
    for iter_number in range(opt.niter):
        train_batch = train_fetcher.load()
        train_loss, _, _ = train_step(model, optimizer, loss_function, train_batch, device)
        train_losses.append(train_loss)

        if opt.verbose >= 2 and (iter_number + 1) % opt.print_every == 0:
            for validate_batch in validate_loader:
                validate_loss, _, _ = validate_step(model, loss_function, validate_batch, device)
                validate_losses.append(validate_loss)

            print_epoch_metrics(iter_number + 1 - opt.print_every,
                                iter_number,
                                time.time() - start_time,
                                torch.mean(torch.stack(train_losses).to(torch.float)).item(),
                                torch.mean(torch.stack(validate_losses).to(torch.float)).item())
            train_losses = []
            validate_losses = []
            start_time = time.time()

        if (iter_number + 1) % opt.save_every == 0:
            torch.save(model.state_dict(),
                       os.path.join(opt.output_dir, f'{opt.model_comment}_iter_{iter_number}.bin'))


def main():
    opt = TrainOptions().get_options()
    set_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    model, tokenizer = get_model_and_tokenizer(opt.model_name_or_path, cache_dir=opt.cache_dir)
    loss_function = nn.CrossEntropyLoss()

    train_df = pd.read_csv(opt.train_data)
    validate_df = pd.read_csv(opt.validate_data)

    train_dataset = TrainDataset(train_df['review_text'], train_df['review_type'], tokenizer, opt.max_length)
    validate_dataset = TrainDataset(validate_df['review_text'], validate_df['review_type'], tokenizer, opt.max_length)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=opt.batch_size)

    train_fetcher = Fetcher(train_loader)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=opt.lr)
    loss_function = loss_function.to(device)

    if opt.verbose >= 1:
        print(f'Start {opt.model_comment} fitting. '
              f'Train size: {train_df.shape[0]} '
              f'Validate size: {validate_df.shape[0]}')
    train(opt, model, train_fetcher, valid_loader, optimizer, loss_function, device)


if __name__ == '__main__':
    main()
