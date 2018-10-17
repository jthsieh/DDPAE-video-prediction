import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
  '''
  Encodes images. Similar structure as DCGAN.
  '''
  def __init__(self, n_channels, output_size, ngf, n_layers):
    super(ImageEncoder, self).__init__()

    layers = [nn.Conv2d(n_channels, ngf, 4, 2, 1, bias=False),
              nn.LeakyReLU(0.2, inplace=True)]

    for i in range(1, n_layers - 1):
      layers += [nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(ngf * 2),
                 nn.LeakyReLU(0.2, inplace=True)]
      ngf *= 2

    layers += [nn.Conv2d(ngf, output_size, 4, 1, 0, bias=False)]

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    x = self.main(x)
    x = x.squeeze(3).squeeze(2)
    return x
