import numpy as np
import os
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from .misc import *

class Visualizer:
  def __init__(self, tb_path):
    self.tb_path = tb_path

    if os.path.exists(tb_path):
      if prompt_yes_no('{} already exists. Proceed?'.format(tb_path)):
        os.system('rm -r {}'.format(tb_path))
      else:
        exit(0)

    self.writer = SummaryWriter(tb_path)

  def add_scalar(self, scalar_dict, global_step=None):
    for tag, scalar in scalar_dict.items():
      if isinstance(scalar, dict):
        self.writer.add_scalars(tag, scalar, global_step)
      elif isinstance(scalar, list) or isinstance(scalar, np.ndarray):
        continue
      else:
        self.writer.add_scalar(tag, scalar, global_step)

  def add_images(self, image_dict, global_step=None, prefix=None):
    for tag, images in image_dict.items():
      if prefix is not None:
        tag = '{}/{}'.format(prefix, tag)
      images = torch.clamp(images, -1, 1)
      images = vutils.make_grid(images, nrow=images.size(0), normalize=True, range=(-1, 1))
      self.writer.add_image(tag, images, global_step)
