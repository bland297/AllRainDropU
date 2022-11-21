import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class DSC2_PLUS(nn.Module):
    def __init__(
        self, 
        in_channels:int=None, 
        out_channels:int=None, 
        ):
        super(DSC2_PLUS, self).__init__()
        self.PLUS = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0), eps=1e-4)
                )
        self.DSC2 = nn.Sequential(
                # spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1), eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels), eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0), eps=1e-4),
                nn.LeakyReLU(0.2, inplace=True),
                # spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1,groups=in_channels), eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0), eps=1e-4),
                )

    def forward(self, x):
        DSC2 = self.DSC2(x)
        PLUS = self.PLUS(x)
        out = DSC2 + PLUS
        return out
    
    
class UP_DSC_PLUS(nn.Module):
    def __init__(
        self,
        in_channels: int=None,
        out_channels: int=None,
        ):
        super(UP_DSC_PLUS, self).__init__() 
        self.PLUS = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)
        self.UP_DSC2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=2, stride=2, padding=0, bias=True),
            # spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1), eps=1e-4),
            spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1,padding_mode='reflect', groups=in_channels), eps=1e-4),
            spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0), eps=1e-4),
            nn.LeakyReLU(0.2, inplace=True),
            # spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), eps=1e-4),
            spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1,padding_mode='reflect', groups=in_channels), eps=1e-4),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0), eps=1e-4),
            )
        self.UP_PLUS = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=2, stride=2, padding=0, bias=True),
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0), eps=1e-4),
            )

    def forward(self, x):
        UP_DSC2 = self.UP_DSC2(x)
        PLUS = self.UP_PLUS(x)
        out = UP_DSC2 + PLUS
        return out

    
class C_DSC_PLUS(nn.Module):
    def __init__(
        self,
        in_channels: int=None,
        out_channels: int=None
        ):
        super(C_DSC_PLUS, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=True)
        self.DSC2 = nn.Sequential(
                # spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1), eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1,padding_mode='reflect', groups=in_channels),eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 1, 1, 0),eps=1e-4),
                nn.LeakyReLU(0.2, inplace=True),
                # spectral_norm(nn.Conv2d(in_channels, out_channels, 3, 1, 1), eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, in_channels, 3, 1, 1,padding_mode='reflect', groups=in_channels),eps=1e-4),
                spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0),eps=1e-4),
                )
        self.PLUS = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0)),
            )

    def forward(self, x):
        x = self.conv(x)
        DSC2 = self.DSC2(x)
        PLUS = self.PLUS(x)
        sum = DSC2 + PLUS
        out = self.relu(sum)
        return out