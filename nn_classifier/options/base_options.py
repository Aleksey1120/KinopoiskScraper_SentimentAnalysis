import argparse


class BaseOptions:

    def __init__(self):
        self.options = self.initialize(argparse.ArgumentParser()).parse_args()
        self.check_options_is_correct()
        self.modify_options()

    def initialize(self, parser):
        parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                            help='Pretrained model name from Huggingface')
        parser.add_argument('--checkpoint_path', default=None, type=str,
                            help='Checkpoint path')
        parser.add_argument('--cache_dir', default=None, type=str,
                            help='Directory for saving the pretrained model(None for default)')
        parser.add_argument('--seed', type=int, default=None,
                            help='Enables repeatable experiments by setting the seed for the random.')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--num_workers', type=int, default=0, help='Num workers for DataLoader')
        parser.add_argument('--max_length', type=int, default=512,
                            help='Max sequence length')
        parser.add_argument('--metrics', nargs='+', default=['loss'], help='Tracked metrics')
        parser.add_argument('--cuda', action='store_true', help='Use cuda if available')
        parser.add_argument(
            '--fp16',
            action='store_true',
            help='Use mixed precision',
        )

        return parser

    def check_options_is_correct(self):
        pass

    def modify_options(self):
        pass

    def get_options(self):
        return self.options

    def print_options(self):
        print('       Option               Value        ')
        for k, v in self.options.__dict__.items():
            print(f'{k:<20}{str(v):<21}')
