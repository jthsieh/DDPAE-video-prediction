from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

import data.video_transforms as vtransforms
from .moving_mnist import MovingMNIST
from .bouncing_balls import BouncingBalls

def get_data_loader(opt):
  if opt.dset_name == 'moving_mnist':
    transform = transforms.Compose([vtransforms.ToTensor()])
    dset = MovingMNIST(opt.dset_path, opt.is_train, opt.n_frames_input,
                       opt.n_frames_output, opt.num_objects, transform)

  elif opt.dset_name == 'bouncing_balls':
    transform = transforms.Compose([vtransforms.Scale(opt.image_size),
                                    vtransforms.ToTensor()])
    dset = BouncingBalls(opt.dset_path, opt.is_train, opt.n_frames_input,
                         opt.n_frames_output, opt.image_size[0], transform)

  else:
    raise NotImplementedError

  dloader = data.DataLoader(dset, batch_size=opt.batch_size, shuffle=opt.is_train,
                            num_workers=opt.n_workers, pin_memory=True)
  return dloader
