import argparse


class BaseOptions:

    def __init__(self):
        self.options = self.initialize(argparse.ArgumentParser()).parse_args()
        self.check_options_is_correct()

    def initialize(self, parser):
        parser.add_argument('--model_name_or_path', default=None, type=str, required=True,
                            help='Pretrained model name from Huggingface')
        parser.add_argument('--model_comment', type=str, required=True, help='Model comment')
        parser.add_argument('--cache_dir', default=None, type=str,
                            help='Directory for saving the pretrained model(None for default)')
        parser.add_argument('--seed', type=int, default=42,
                            help='Randomization fixation')
        parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
        parser.add_argument('--max_length', type=int, default=512,
                            help='Max sequence length')
        parser.add_argument('--metrics', nargs='+', default=['loss'], help='Tracked metrics')
        parser.add_argument('--cuda', action='store_true', help='Use cuda if available')

        return parser

    def check_options_is_correct(self):
        pass

    def get_options(self):
        return self.options

    # TODO: add print_options func
