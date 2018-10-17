import os
from collections import OrderedDict
import torch
import pyro

'''
Basic functions.
'''

def init_weights(m):
  class_name = m.__class__.__name__
  try:
    if class_name.find('Conv') != -1:
      m.weight.data.normal_(0.0, 0.02)
      if m.bias is not None:
        m.bias.data.fill_(0)
    elif class_name.find('Linear') != -1:
      m.weight.data.normal_(0.0, 0.02)
      m.bias.data.fill_(0)
    elif class_name.find('BatchNorm2d') != -1:
      m.weight.data.normal_(1.0, 0.02)
      m.bias.data.fill_(0)
  except:
    print('Exception in init_weights:', class_name)

class BaseModel:
  '''
  Base model that implements basic functions such as saving and loading checkpoints,
  saving results, update hyperparameters, etc.
  '''
  def __init__(self):
    self.nets, self.optimizers, self.schedulers = {}, {}, []
    self.video_dict = {} # For visualization

  def initialize_weights(self):
    for _, net in self.nets.items():
      net.apply(init_weights)

  def setup(self, is_train):
    for _, net in self.nets.items():
      if is_train:
        net.train()
      else:
        net.eval()

  def load(self, ckpt_path, epoch, load_optimizer=False):
    '''
    Load checkpoint.
    '''
    for name, net in self.nets.items():
      path = os.path.join(ckpt_path, 'net_{}_{}.pth'.format(name, epoch))
      if not os.path.exists(path):
        print('{} does not exist, ignore.'.format(path))
        continue
      ckpt = torch.load(path)
      if isinstance(net, torch.nn.DataParallel):
        module = net.module
      else:
        module = net

      try:
        module.load_state_dict(ckpt)
      except:
        print('net_{} and checkpoint have different parameter names'.format(name))
        new_ckpt = OrderedDict()
        for ckpt_key, module_key in zip(ckpt.keys(), module.state_dict().keys()):
          assert ckpt_key.split('.')[-1] == module_key.split('.')[-1]
          new_ckpt[module_key] = ckpt[ckpt_key]
        module.load_state_dict(new_ckpt)

    if load_optimizer:
      for name, optimizer in self.optimizers.items():
        path = os.path.join(ckpt_path, 'optimizer_{}_{}.pth'.format(name, epoch))
        if not os.path.exists(path):
          print('{} does not exist, ignore.'.format(path))
          continue
        ckpt = torch.load(path)
        optimizer.load_state_dict(ckpt)

  def save(self, ckpt_path, epoch):
    '''
    Save checkpoint.
    '''
    for name, net in self.nets.items():
      if isinstance(net, torch.nn.DataParallel):
        module = net.module
      else:
        module = net

      path = os.path.join(ckpt_path, 'net_{}_{}.pth'.format(name, epoch))
      torch.save(module.state_dict(), path)

    for name, optimizer in self.optimizers.items():
      path = os.path.join(ckpt_path, 'optimizer_{}_{}.pth'.format(name, epoch))
      torch.save(optimizer.state_dict(), path)

  def pyro_sample(self, name, fn, mu, sigma, sample=True):
    '''
    Sample with pyro.sample. fn should be dist.Normal.
    If sample is False, then return mean.
    '''
    if sample:
      return pyro.sample(name, fn(mu, sigma))
    else:
      return mu.contiguous()

  def save_visuals(self, gt, output, components, latent):
    '''
    Save data for visualization.
    Take the first result in the batch.
    '''
    videos = [gt.data[0].cpu()]
    for i in range(components.size(2)):
      images = components.data[0, :, i, ...].cpu()
      videos.append(images)

    videos.append(output.data[0].cpu())
    videos = torch.cat(videos, dim=2).clamp(0, 1)
    videos = videos * 2 - 1 # map to [-1, 1]
    self.video_dict.update({'results': videos})

  def get_visuals(self):
    return self.video_dict

  def update_hyperparameters(self, epoch, n_epochs):
    '''
    Update learning rate.
    Multiply learning rate by 0.1 halfway through training.
    '''
    # Learning rate
    lr = self.lr_init
    if self.lr_decay:
      if epoch >= n_epochs // 2:
        lr = self.lr_init * 0.1
      for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return {'lr': lr}
