import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def expand_pose(pose):
  '''
  param pose: N x 3
  Takes 3-dimensional vectors, and massages them into 2x3 affine transformation matrices:
  [s,x,y] -> [[s,0,x],
              [0,s,y]]
  '''
  n = pose.size(0)
  expansion_indices = Variable(torch.LongTensor([1, 0, 2, 0, 1, 3]).cuda(), requires_grad=False)
  zeros = Variable(torch.zeros(n, 1).cuda(), requires_grad=False)
  out = torch.cat([zeros, pose], dim=1)
  return torch.index_select(out, 1, expansion_indices).view(n, 2, 3)

def pose_inv(pose):
  '''
  param pose: N x 3
  [s,x,y] -> [1/s,-x/s,-y/s]
  '''
  N, _ = pose.size()
  ones = Variable(torch.ones(N, 1).cuda(), requires_grad=False)
  out = torch.cat([ones, -pose[:, 1:]], dim=1)
  out = out / pose[:, 0:1]
  return out

def pose_inv_full(pose):
  '''
  param pose: N x 6
  Inverse the 2x3 transformer matrix.
  '''
  N, _ = pose.size()
  b = pose.view(N, 2, 3)[:, :, 2:]
  # A^{-1}
  # Calculate determinant
  determinant = (pose[:, 0] * pose[:, 4] - pose[:, 1] * pose[:, 3] + 1e-8).view(N, 1)
  indices = Variable(torch.LongTensor([4, 1, 3, 0]).cuda())
  scale = Variable(torch.Tensor([1, -1, -1, 1]).cuda())
  A_inv = torch.index_select(pose, 1, indices) * scale / determinant
  A_inv = A_inv.view(N, 2, 2)
  # b' = - A^{-1} b
  b_inv = - A_inv.matmul(b).view(N, 2, 1)
  transformer_inv = torch.cat([A_inv, b_inv], dim=2)
  return transformer_inv

def image_to_object(images, pose, object_size):
  '''
  Inverse pose, crop and transform image patches.
  param images: (... x C x H x W) tensor
  param pose: (N x 3) tensor
  '''
  N, pose_size = pose.size()
  n_channels, H, W = images.size()[-3:]
  images = images.view(N, n_channels, H, W)
  if pose_size == 3:
    transformer_inv = expand_pose(pose_inv(pose))
  elif pose_size == 6:
    transformer_inv = pose_inv_full(pose)

  grid = F.affine_grid(transformer_inv,
                       torch.Size((N, n_channels, object_size, object_size)))
  obj = F.grid_sample(images, grid)
  return obj

def object_to_image(objects, pose, image_size):
  '''
  param images: (N x C x H x W) tensor
  param pose: (N x 3) tensor
  '''
  N, pose_size = pose.size()
  _, n_channels, _, _ = objects.size()
  if pose_size == 3:
    transformer = expand_pose(pose)
  elif pose_size == 6:
    transformer = pose.view(N, 2, 3)

  grid = F.affine_grid(transformer,
                       torch.Size((N, n_channels, image_size, image_size)))
  components = F.grid_sample(objects, grid)
  return components

def calculate_positions(pose):
  '''
  Get the center x, y of the spatial transformer.
  '''
  N, pose_size = pose.size()
  assert pose_size == 3, 'Only implemented pose_size == 3'
  # s, x, y
  s = pose[:, 0]
  xt = pose[:, 1]
  yt = pose[:, 2]
  x = (- xt / s + 1) / 2
  y = (- yt / s + 1) / 2
  return torch.stack([x, y], dim=1)

def bounding_box(z_where, x_size):
  """This doesn't take into account interpolation, but it's close
  enough to be usable."""
  s, x, y = z_where
  w = x_size / s
  h = x_size / s
  xtrans = -x / s * x_size / 2.
  ytrans = -y / s * x_size / 2.
  x = (x_size - w) / 2 + xtrans  # origin is top left
  y = (x_size - h) / 2 + ytrans
  return (x, y), w, h

def draw_components(images, pose):
  '''
  Draw bounding box for the given pose.
  images: size (N x C x H x W), range [0, 1]
  pose: N x 3
  '''
  images = (images.cpu().numpy() * 255).astype(np.uint8) # [0, 255]
  pose = pose.cpu().numpy()
  N, C, H, W = images.shape
  for i in range(N):
    if C == 1:
      img = images[i][0]
    else:
      img = images[i].transpose((1, 2, 0))
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    (x, y), w, h = bounding_box(pose[i], H)
    draw.rectangle([x, y, x + w, y + h], outline=128)
    new_img = np.array(img)
    new_img[0, ...] = 255 # Add line
    new_img[-1, ...] = 255 # Add line
    if C == 1:
      new_img = new_img[np.newaxis, :, :]
    else:
      new_img = new_img.transpose((2, 0, 1))
    images[i] = new_img

  # Back to torch tensor
  images = torch.FloatTensor(images / 255)
  return images
