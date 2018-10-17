from .base_args import BaseArgs


class TestArgs(BaseArgs):
  '''
  Arguments for testing.
  '''
  def __init__(self):
    super(TestArgs, self).__init__()

    self.is_train = False
    self.split = 'val'

    # hyperparameters
    self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    self.parser.add_argument('--which_epochs', type=int, nargs='+', default=[-1],
                             help='which epochs to evaluate, -1 to load latest checkpoint')
    self.parser.add_argument('--save_visuals', type=int, default=0, help='Save results to tensorboard')
    self.parser.add_argument('--save_all_results', type=int, default=0, help='Save results to tensorboard')
