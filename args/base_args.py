import argparse
import os


class BaseArgs:
  '''
  Arguments for data, model, and checkpoints.
  '''
  def __init__(self):
    self.is_train, self.split = None, None
    self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # hardware
    self.parser.add_argument('--n_workers', type=int, default=8, help='number of threads')
    self.parser.add_argument('--gpus', type=str, default='0', help='visible GPU ids, separated by comma')

    # data
    self.parser.add_argument('--dset_dir', type=str, default=os.path.join(os.environ['HOME'], 'slowbro'))
    self.parser.add_argument('--dset_name', type=str, default='moving_mnist')
    self.parser.add_argument('--image_size', type=int, nargs='+', default=[64, 64])
    self.parser.add_argument('--n_frames_input', type=int, default=10)
    self.parser.add_argument('--n_frames_output', type=int, default=10)
    # Moving MNIST
    self.parser.add_argument('--num_objects', type=int, nargs='+', default=[2],
                             help='Max number of digits in Moving MNIST videos.')

    # model
    self.parser.add_argument('--model', type=str, default='crop', help='Model name')
    self.parser.add_argument('--n_components', type=int, default=2)
    self.parser.add_argument('--image_latent_size', type=int, default=256,
                             help='Output size of image encoder')
    self.parser.add_argument('--content_latent_size', type=int, default=128,
                             help='Size of content vector')
    self.parser.add_argument('--pose_latent_size', type=int, default=3,
                             help='Size of pose vector')
    self.parser.add_argument('--hidden_size', type=int, default=64,
                             help='Hidden size of LSTM')
    self.parser.add_argument('--ngf', type=int, default=8,
                             help='number of channels in encoder and decoder')
    self.parser.add_argument('--stn_scale_prior', type=float, default=3,
                             help='The scale of the spatial transformer prior.')
    self.parser.add_argument('--independent_components', type=int, default=0,
                             help='Baseline: (if set to 1) independent prediction of each component.')

    # ckpt and logging
    self.parser.add_argument('--ckpt_dir', type=str, default=os.path.join(os.environ['HOME'], 'slowbro', 'ckpt'),
                             help='the directory that contains all checkpoints')
    self.parser.add_argument('--ckpt_name', type=str, default='ckpt', help='checkpoint name')
    self.parser.add_argument('--log_every', type=int, default=400, help='log every x steps')
    self.parser.add_argument('--save_every', type=int, default=50, help='save every x epochs')
    self.parser.add_argument('--evaluate_every', type=int, default=-1, help='evaluate on val set every x epochs')

  def parse(self):
    opt = self.parser.parse_args()

    assert opt.n_frames_input > 0 and opt.n_frames_output > 0
    # for convenience
    opt.is_train, opt.split = self.is_train, self.split
    opt.dset_path = os.path.join(opt.dset_dir, opt.dset_name)
    if opt.is_train:
      ckpt_name = '{:s}_NC{:d}_lr{:.01e}_bt{:d}_{:s}'.format(opt.model,
                      opt.n_components, opt.lr_init,
                      opt.batch_size, opt.ckpt_name)
    else:
      ckpt_name = opt.ckpt_name
    opt.ckpt_path = os.path.join(opt.ckpt_dir, opt.dset_name, ckpt_name)

    # Hard code
    if opt.dset_name == 'moving_mnist':
      opt.n_channels = 1
      opt.image_size = (64, 64)
    elif opt.dset_name == 'bouncing_balls':
      opt.n_channels = 1
      opt.image_size = (128, 128)
    else:
      raise NotImplementedError

    if opt.model == 'crop':
      opt.pose_latent_size = 3
    else:
      raise NotImplementedError

    log = ['Arguments: ']
    for k, v in sorted(vars(opt).items()):
      log.append('{}: {}'.format(k, v))

    return opt, log
