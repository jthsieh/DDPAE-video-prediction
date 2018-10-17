import torch
import torch.nn as nn

class ImageDecoder(nn.Module):
  '''
  Decode images from vectors. Similar structure as DCGAN.
  '''
  def __init__(self, input_size, n_channels, ngf, n_layers, activation='tanh'):
    super(ImageDecoder, self).__init__()

    ngf = ngf * (2 ** (n_layers - 2))
    layers = [nn.ConvTranspose2d(input_size, ngf, 4, 1, 0, bias=False),
              nn.BatchNorm2d(ngf),
              nn.ReLU(True)]

    for i in range(1, n_layers - 1):
      layers += [nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                 nn.BatchNorm2d(ngf // 2),
                 nn.ReLU(True)]
      ngf = ngf // 2

    layers += [nn.ConvTranspose2d(ngf, n_channels, 4, 2, 1, bias=False)]
    if activation == 'tanh':
      layers += [nn.Tanh()]
    elif activation == 'sigmoid':
      layers += [nn.Sigmoid()]
    else:
      raise NotImplementedError

    self.main = nn.Sequential(*layers)

  def forward(self, x):
    if len(x.size()) == 2:
      x = x.view(*x.size(), 1, 1)
    x = self.main(x)
    return x
