import logging
import os
import sys

from .misc import blue


class Logger:
  '''
  Logger to write logs to file.
  '''
  def __init__(self, ckpt_path, name='train'):
    self.logger = logging.getLogger()
    self.logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s',
                                  datefmt=blue('[%Y-%m-%d,%H:%M:%S]'))

    fh = logging.FileHandler(os.path.join(ckpt_path, '{}.log'.format(name)), 'w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    self.logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    self.logger.addHandler(ch)

  def print(self, log):
    if isinstance(log, list):
      self.logger.info('\n - '.join(log))
    else:
      self.logger.info(log)
