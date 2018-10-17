import torch
import numpy as np

def to_numpy(array):
  """
  :param array: Variable, GPU tensor, or CPU tensor
  :return: numpy
  """
  if isinstance(array, np.ndarray):
    return array
  if isinstance(array, torch.autograd.Variable):
    array = array.data
  if array.is_cuda:
    array = array.cpu()

  return array.numpy()

def blue(string):
  return '\033[94m'+string+'\033[0m'

def prompt_yes_no(question):
  '''
  Prompt user to type yes or no.
  '''
  i = input(question + ' [y/n]: ')
  if len(i) > 0 and (i[0] == 'y' or i[0] == 'Y'):
    return True
  else:
    return False
