import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    ndf = 64
    self.main = nn.Sequential(
        spectral_norm(nn.Conv2d(1, ndf, 4, 2, 1, bias=True),eps=1e-4),
        nn.BatchNorm2d(ndf),
        nn.LeakyReLU(0.2, inplace=True),  # (ndf = 64) x 64 x 64
        spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=True),eps=1e-4),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),  # (ndf*2 = 128) x 32 x 32
        spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=True),eps=1e-4),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),  # (ndf*4 = 256) x 16 x 16
        spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=True),eps=1e-4),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),  # 512 x 8 x 8 
        nn.Conv2d(ndf * 8, 1, 8, 1, 0, bias=True),
        nn.Sigmoid()
    )
  def forward(self, x):
    judge = self.main(x)
    return judge