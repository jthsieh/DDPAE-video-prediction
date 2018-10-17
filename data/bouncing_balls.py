from glob import glob
import os
import numpy as np
import random
import torch.utils.data as data
import json
import cv2

def make_dataset(root, is_train):
  if is_train:
    folder = 'balls_n4_t60_ex50000'
  else:
    folder = 'balls_n4_t60_ex2000'

  dataset = np.load(os.path.join(root, folder, 'dataset_info.npy'))
  return dataset

class BouncingBalls(data.Dataset):
  '''
  Bouncing balls dataset.
  '''
  def __init__(self, root, is_train, n_frames_input, n_frames_output, image_size,
               transform=None, return_positions=False):
    super(BouncingBalls, self).__init__()
    self.n_frames = n_frames_input + n_frames_output
    self.dataset = make_dataset(root, is_train)

    self.size = image_size
    self.scale = self.size / 800
    self.radius = int(60 * self.scale)

    self.root = root
    self.is_train = is_train
    self.n_frames_input = n_frames_input
    self.n_frames_output = n_frames_output
    self.transform = transform
    self.return_positions = return_positions

  def __getitem__(self, idx):
    # traj sizeL (n_frames, n_balls, 4)
    traj = self.dataset[idx]
    vid_len, n_balls = traj.shape[:2]
    if self.is_train:
      start = random.randint(0, vid_len - self.n_frames)
    else:
      start = 0

    n_channels = 1
    images = np.zeros([self.n_frames, self.size, self.size, n_channels], np.uint8)
    positions = []
    for fid in range(self.n_frames):
      xy = []
      for bid in range(n_balls):
        # each ball:
        ball = traj[start + fid, bid]
        x, y = int(round(self.scale * ball[0])), int(round(self.scale * ball[1]))
        images[fid] = cv2.circle(images[fid], (x, y), int(self.radius * ball[3]),
                                 255, -1)
        xy.append([x / self.size, y / self.size])
      positions.append(xy)

    if self.transform is not None:
      images = self.transform(images)

    input = images[:self.n_frames_input]
    if self.n_frames_output > 0:
      output = images[self.n_frames_input:]
    else:
      output = []

    if not self.return_positions:
      return input, output
    else:
      positions = np.array(positions)
      return input, output, positions

  def __len__(self):
    return len(self.dataset)
