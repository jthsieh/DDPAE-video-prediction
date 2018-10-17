from .base_args import BaseArgs


class TrainArgs(BaseArgs):
  '''
  Arguments specific for training.
  '''
  def __init__(self):
    super(TrainArgs, self).__init__()

    self.is_train = True
    self.split = 'train'

    self.parser.add_argument('--batch_size', type=int, default=4, help='batch size per gpu')
    self.parser.add_argument('--n_epochs', type=int, default=50, help='total # of epochs')
    self.parser.add_argument('--n_iters', type=int, default=0, help='total # of iterations')
    self.parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    self.parser.add_argument('--lr_init', type=float, default=1e-3, help='initial learning rate')
    self.parser.add_argument('--lr_decay', type=int, default=1, choices=[0, 1], help='whether to decay learning rate')
    self.parser.add_argument('--load_ckpt_dir', type=str, default='', help='directory of checkpoint')
    self.parser.add_argument('--load_ckpt_epoch', type=int, default=0, help='epoch to load checkpoint')
    self.parser.add_argument('--when_to_predict_only', type=float, default=0, help='when to set predict_loss_only to True.')
