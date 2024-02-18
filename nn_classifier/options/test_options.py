from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            '--test_data', type=str, required=True, help='Test dataset path'
        )
        return parser
