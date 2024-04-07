# nn_classifier

This module is responsible for training and testing the neural network classifier.

## Usage

Training and testing of the classifier are performed through the command line.

Training:
```
usage: train.py [-h] --model_name_or_path MODEL_NAME_OR_PATH [--checkpoint_path CHECKPOINT_PATH] [--cache_dir CACHE_DIR] [--seed SEED] [--batch_size BATCH_SIZE]
                [--num_workers NUM_WORKERS] [--max_length MAX_LENGTH] [--metrics METRICS [METRICS ...]] [--cuda] [--fp16] --output_dir OUTPUT_DIR [--niter NITER]        
                [--early_stop EARLY_STOP] [--print_every PRINT_EVERY] [--target_metric TARGET_METRIC] [--verbose VERBOSE] [--save_best_model] [--save_every SAVE_EVERY]  
                --model_comment MODEL_COMMENT --train_data TRAIN_DATA --validate_data VALIDATE_DATA [--tb_dir TB_DIR] [--tb_comment TB_COMMENT] [--lr LR]
                [--gamma GAMMA] [--label_smoothing LABEL_SMOOTHING] [--balanced] [--use_cache]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Pretrained model name from Huggingface
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint path
  --cache_dir CACHE_DIR
                        Directory for saving the pretrained model(None for default)
  --seed SEED           Enables repeatable experiments by setting the seed for the random.
  --batch_size BATCH_SIZE
                        Batch size
  --num_workers NUM_WORKERS
                        Num workers for DataLoader
  --max_length MAX_LENGTH
                        Max sequence length
  --metrics METRICS [METRICS ...]
                        Tracked metrics
  --cuda                Use cuda if available
  --fp16                Use mixed precision
  --output_dir OUTPUT_DIR
                        Directory for saving the model and its checkpoints
  --niter NITER         Maximum number of iters
  --early_stop EARLY_STOP
                        Stop training after n evaluate iters without improvement
  --print_every PRINT_EVERY
                        Print every k iters
  --target_metric TARGET_METRIC
                        Metric for selecting the best model
  --verbose VERBOSE     Controls the verbosity
  --save_best_model     Save best model
  --save_every SAVE_EVERY
                        Save checkpoint every k iters
  --model_comment MODEL_COMMENT
                        Model comment
  --train_data TRAIN_DATA
                        Train dataset path
  --validate_data VALIDATE_DATA
                        Validate dataset path
  --tb_dir TB_DIR       TensorBoard dir path
  --tb_comment TB_COMMENT
                        TensorBoard comment. If not given, current time
  --lr LR               Learning rate
  --gamma GAMMA         Gamma for exponential scheduler
  --label_smoothing LABEL_SMOOTHING
                        Specifies the amount of smoothing when computing the loss.
  --balanced            Use balanced class weights
  --use_cache           Use tokens caching
  ```
Test:
```
usage: test.py [-h] --model_name_or_path MODEL_NAME_OR_PATH [--checkpoint_path CHECKPOINT_PATH] [--cache_dir CACHE_DIR] [--seed SEED] [--batch_size BATCH_SIZE]
               [--num_workers NUM_WORKERS] [--max_length MAX_LENGTH] [--metrics METRICS [METRICS ...]] [--cuda] [--fp16] --test_data TEST_DATA

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Pretrained model name from Huggingface
  --checkpoint_path CHECKPOINT_PATH
                        Checkpoint path
  --cache_dir CACHE_DIR
                        Directory for saving the pretrained model(None for default)
  --seed SEED           Enables repeatable experiments by setting the seed for the random.
  --batch_size BATCH_SIZE
                        Batch size
  --num_workers NUM_WORKERS
                        Num workers for DataLoader
  --max_length MAX_LENGTH
                        Max sequence length
  --metrics METRICS [METRICS ...]
                        Tracked metrics
  --cuda                Use cuda if available
  --fp16                Use mixed precision
  --test_data TEST_DATA
                        Test dataset path
```

## Model selection

cointegrated/rubert-tiny2 was chosen as the base model for classification.

It has several advantages:
- russian text support: specifically trained on Russian text data
- large input sequence size: long text sequences (up to 2048 tokens) allows to capture more context for accurate classification.
- fast inference: enables real-time classification tasks.
- fast learning: reduces training time compared to larger models.

## Performance comparison

The distribution of text lengths is shown in the diagram below

![plot](../imgs/Figure_1.png)

So a decision was made to compare performance of rubert-tiny2 with different sequence size. The table below shows it.

| Sequence size | Covering | ROC AUC | Average precision |
|:-------------:|:--------:|:-------:|:-----------------:|
|      512      |  60.4%   |  0.871  |       0.704       |
|     1024      |  93.1%   |  0.898  |       0.765       |

Increasing the sequence size noticeably improves performance.

## Learning tracking

Tensorboard can be used to track learning.
To do this, you must specify the options --tb_dir and --tb_comment and then run tensorboard using
```
tensorboard --logdir=tb_dir
```
