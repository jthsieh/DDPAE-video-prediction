import numpy as np
import os
import random
import torch

import args
from .logger import Logger
from .visualizer import Visualizer


def build(is_train, tb_dir=None):
  '''
  Parse arguments, setup logger and tensorboardX directory.
  '''
  opt, log = args.TrainArgs().parse() if is_train else args.TestArgs().parse()

  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
  os.makedirs(opt.ckpt_path, exist_ok=True)

  # Set seed
  torch.manual_seed(666)
  torch.cuda.manual_seed_all(666)
  np.random.seed(666)
  random.seed(666)

  logger = Logger(opt.ckpt_path, opt.split)

  if tb_dir is not None:
    tb_path = os.path.join(opt.ckpt_path, tb_dir)
    vis = Visualizer(tb_path)
  else:
    vis = None

  logger.print(log)

  return opt, logger, vis
