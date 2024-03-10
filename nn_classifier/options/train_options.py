import datetime

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def check_options_is_correct(self):
        assert self.options.target_metric in self.options.metrics, \
            'Target metric must be in the list of tracked metrics'

        assert self.options.save_best_model or self.options.save_every, \
            'Use --save_every and/or --save_best_model to save the training result'
        assert self.options.niter >= 1, 'niter must be >= 1'

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            '--output_dir',
            type=str,
            required=True,
            help='Directory for saving the model and its checkpoints',
        )
        parser.add_argument('--niter', type=int, default=1000,
                            help='Maximum number of iters')
        parser.add_argument('--early_stop', type=int, default=None,
                            help='Stop training after n evaluate iters without improvement')
        parser.add_argument('--print_every', type=int, default=100,
                            help='Print every k iters')
        parser.add_argument('--target_metric', type=str, default='loss',
                            help='Metric for selecting the best model')
        parser.add_argument('--verbose', type=int, default=0, help='Controls the verbosity')
        parser.add_argument('--save_best_model', action='store_true',
                            help='Save best model')
        parser.add_argument('--save_every', type=int, default=None,
                            help='Save checkpoint every k iters')
        parser.add_argument('--model_comment', type=str, required=True, help='Model comment')
        parser.add_argument(
            '--train_data', type=str, required=True, help='Train dataset path'
        )
        parser.add_argument(
            '--validate_data', type=str, required=True, help='Validate dataset path'
        )
        parser.add_argument('--tb_dir', type=str, default=None,
                            help='TensorBoard dir path')
        parser.add_argument('--tb_comment', type=str, default=datetime.datetime.now().strftime('%d-%m-%y %H.%M'),
                            help='TensorBoard comment. If not given, current time')
        parser.add_argument('--lr', default=1e-6, type=float, help='Learning rate')
        parser.add_argument('--gamma', default=0.9, type=float, help='Gamma for exponential scheduler')
        parser.add_argument('--balanced', action='store_true', help='Use balanced class weights')
        parser.add_argument('--use_cache', action='store_true', help='Use tokens caching')
        return parser
