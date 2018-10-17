import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils

class Metrics(object):
  '''
  Evaluation metric: BCE and MSE.
  '''
  def __init__(self):
    self.bce_loss = nn.BCELoss()
    self.mse_loss = nn.MSELoss()
    self.bce_results = []
    self.mse_results = []

  def update(self, gt, pred):
    """
    gt, pred are tensors of size (..., 1, H, W) in the range [0, 1].
    """
    C, H, W = gt.size()[-3:]
    if isinstance(gt, torch.Tensor):
      gt = Variable(gt)
    if isinstance(pred, torch.Tensor):
      pred = Variable(pred)

    mse_score = self.mse_loss(pred, gt)
    eps = 1e-4
    pred.data[pred.data < eps] = eps
    pred.data[pred.data > 1 - eps] = 1 -eps
    bce_score = self.bce_loss(pred, gt)
    bce_score = bce_score.item() * C * H * W
    mse_score = mse_score.item() * C * H * W
    self.bce_results.append(bce_score)
    self.mse_results.append(mse_score)

  def get_scores(self):
    bce_score = np.mean(self.bce_results)
    mse_score = np.mean(self.mse_results)
    scores = {'bce': bce_score, 'mse':mse_score}
    return scores

  def reset(self):
    self.bce_results = []
    self.mse_results = []


class VelocityMetrics(object):
  '''
  Evaluation metric for bounding balls: relative error and cosine similarity.
  '''
  def __init__(self, save_path='', shift=2):
    self.relative_errors = []
    self.cosine_similarities = []
    self.masks = []
    self.pred_positions = []
    self.gt_positions = []
    self.save_path = save_path
    self.shift = shift

  def update(self, gt, pose, n_frames_input):
    pred = utils.calculate_positions(pose.view(-1, pose.size(-1))).view_as(gt)
    pred = utils.to_numpy(pred)
    gt = utils.to_numpy(gt)
    self.pred_positions.append(pred)
    self.gt_positions.append(gt)
    self.calculate_metrics(pred, gt, n_frames_input)

  def calculate_metrics(self, pred, gt, n_frames_input):
    batch_size, n_frames_total, n_components, _ = pred.shape
    n_frames_output = n_frames_total - n_frames_input
    # Match gt an pred from input
    pred_input = pred[:, :n_frames_input, ...]
    gt_input = gt[:, :n_frames_input, ...]
    matching_indices, ambiguous = self.find_match(pred_input, gt_input)

    # Calculate velocity
    shift = self.shift
    pred = pred[:, (n_frames_input - shift):, ...] # Keep the last input frame
    # batch_size x n_frames_output x n_components x 2
    pred_vel = pred[:, shift:, ...] - pred[:, :-shift, ...]
    # batch_size x n_frames_output x n_components
    pred_norm = np.linalg.norm(pred_vel, axis=-1)

    gt = gt[:, (n_frames_input - shift):, ...] # Keep the last input frame
    # batch_size x n_frames_output x n_components x 2
    gt_vel = gt[:, shift:, ...] - gt[:, :-shift, ...]
    # batch_size x n_frames_output x n_components
    gt_norm = np.linalg.norm(gt_vel, axis=-1)

    mask = np.ones((batch_size, n_frames_output, n_components))
    rel_error = np.zeros((batch_size, n_frames_output, n_components))
    cosine_distance = np.ones((batch_size, n_frames_output, n_components))
    for i in range(batch_size):
      if ambiguous[i]:
        mask[i, :] = 0
        continue
      for j in range(n_components):
        gt_idx = matching_indices[i, j]
        vel1 = pred_vel[i, :, j, :]
        vel2 = gt_vel[i, :, gt_idx, :]
        norm1 = pred_norm[i, :, j]
        norm2 = gt_norm[i, :, gt_idx]
        # zero norms, can't calculate error and cosine distance if norm is 0.
        norm_is_zero = np.logical_or(norm1 < 1e-4, norm2 < 1e-4)
        mask[i, norm_is_zero, j] = 0
        norm1[norm_is_zero] = 1e-4
        norm2[norm_is_zero] = 1e-4

        dot_product = np.sum(vel1 * vel2, axis=-1)
        cosine = dot_product / norm1 / norm2
        rel = np.abs(1 - (norm1 / norm2))
        # Save results
        cosine_distance[i, :, j] = cosine
        rel_error[i, :, j] = rel

    # Average over the components
    self.relative_errors.append(rel_error)
    self.cosine_similarities.append(cosine_distance)
    self.masks.append(mask)

  def find_match(self, pred, gt):
    '''
    Match component to balls.
    '''
    batch_size, n_frames_input, n_components, _ = pred.shape
    diff = pred.reshape(batch_size, n_frames_input, n_components, 1, 2) - \
               gt.reshape(batch_size, n_frames_input, 1, n_components, 2)
    diff = np.sum(np.sum(diff ** 2, axis=-1), axis=1)
    # Direct indices
    indices = np.argmin(diff, axis=2)
    ambiguous = np.zeros(batch_size, dtype=np.int8)
    for i in range(batch_size):
      _, counts = np.unique(indices[i], return_counts=True)
      if not np.all(counts == 1):
        ambiguous[i] = 1
    return indices, ambiguous

  def get_scores(self):
    # Save positions
    if self.save_path != '':
      positions = np.array([np.concatenate(self.pred_positions, axis=0),
                            np.concatenate(self.gt_positions, axis=0)])
      np.save(os.path.join(self.save_path), positions)

    masks = np.concatenate(self.masks, axis=0)
    cosine = np.concatenate(self.cosine_similarities, axis=0)
    rel_error = np.concatenate(self.relative_errors, axis=0)

    numel = np.sum(masks == 1, axis=(0,2))
    rel_error = np.sum(rel_error * masks, axis=(0,2)) / numel
    cosine = np.sum(cosine * masks, axis=(0,2)) / numel
    return {'relative_errors': rel_error, 'cosine_similarities': cosine}

  def reset(self):
    self.__init__()
