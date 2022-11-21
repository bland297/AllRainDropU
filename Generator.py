import torch
import torch.nn as nn
from Layer import DSC2_PLUS as DP
from Layer import UP_DSC_PLUS as UpDP
from Layer import C_DSC_PLUS as CDP
from torch.nn.utils.parametrizations import spectral_norm

class Generator(nn.Module):
    def __init__(
        self,
        in_step: int=4,
        ):
        super().__init__()
        self.in_step = in_step

        self.DP0 = nn.ModuleList([DP(4, 64) for _ in range(2)])
        self.DP1 = nn.ModuleList([DP(64,128) for _ in range(2)])
        self.DP2 = nn.ModuleList([DP(128,256) for _ in range(2)])
        self.DP3 = nn.ModuleList([DP(256,512) for _ in range(2)])
        self.DP4 = DP(512,512)

        self.conv0 = nn.Sequential(
                # spectral_norm(nn.Conv2d(192, 64, 3, 1, 1)),
                spectral_norm(nn.Conv2d(128, 128, 3, 1, 1, groups=128)),
                spectral_norm(nn.Conv2d(128, 64, 1, 1, 0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.conv1 = nn.Sequential(
                # spectral_norm(nn.Conv2d(384, 128, 3, 1, 1)),
                spectral_norm(nn.Conv2d(256, 256, 3, 1, 1, groups=256)),
                spectral_norm(nn.Conv2d(256, 128, 1, 1, 0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.conv2 = nn.Sequential(
                # spectral_norm(nn.Conv2d(768, 256, 3, 1, 1)),
                spectral_norm(nn.Conv2d(512, 512, 3, 1, 1, groups=512)),
                spectral_norm(nn.Conv2d(512, 256, 1, 1, 0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.conv3 = nn.Sequential(
                # spectral_norm(nn.Conv2d(1536, 512, 3, 1, 1)),
                spectral_norm(nn.Conv2d(1024, 1024, 3, 1, 1, groups=1024)),
                spectral_norm(nn.Conv2d(1024, 512, 1, 1, 0)),
                nn.LeakyReLU(0.2,inplace=True)
                )

        self.scaling = nn.AvgPool2d(2)

        #0layer:64x128x128 / 1layer:128x64x64 / 2layer:256x32x32 / 3layer:512x16x16 / 4layer:1024x8x8
        self.trans23 = nn.Sequential(
                nn.AvgPool2d(2),
                spectral_norm(nn.Conv2d(256,256,3,1,1,groups=256)),
                spectral_norm(nn.Conv2d(256,128,1,1,0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.trans33 = nn.Sequential(
                spectral_norm(nn.Conv2d(512,512,3,1,1,groups=512)),
                spectral_norm(nn.Conv2d(512,128,1,1,0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.trans12 = nn.Sequential(
                nn.AvgPool2d(2),
                spectral_norm(nn.Conv2d(128,128,3,1,1,groups=128)),
                spectral_norm(nn.Conv2d(128,128,1,1,0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.trans22 = nn.Sequential(
                spectral_norm(nn.Conv2d(256,256,3,1,1,groups=256)),
                spectral_norm(nn.Conv2d(256,128,1,1,0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.trans01 = nn.Sequential(
                nn.AvgPool2d(2),
                spectral_norm(nn.Conv2d(64,64,3,1,1,groups=64)),
                spectral_norm(nn.Conv2d(64,128,1,1,0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.trans11 = nn.Sequential(
                spectral_norm(nn.Conv2d(128,128,3,1,1,groups=128)),
                spectral_norm(nn.Conv2d(128,128,1,1,0)),
                nn.LeakyReLU(0.2,inplace=True)
                )
        self.trans001 = DP(64,128)
        self.trans002 = DP(1,128)

        self.CDP4 = CDP(512, 128)
        self.CDPa = CDP(384, 128)
        self.CDPb = CDP(128, 128)
        self.CDP0 = CDP(384, 64)

        self.CDP_up = UpDP(128, 128)

        self.conv_out = nn.Sequential(
                nn.Conv2d(64,64,1,1,0),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1, 1, 0)
                )

    def forward(self, input):
        temp_x0, temp_x1, temp_x2, temp_x3= [], [], [], []
        for block0, block1, block2, block3 in zip(self.DP0, self.DP1, self.DP2, self.DP3):
            d0 = block0(input)
            d1 = block1(self.scaling(d0))
            d2 = block2(self.scaling(d1))
            d3 = block3(self.scaling(d2))
            temp_x0.append(d0)
            temp_x1.append(d1)
            temp_x2.append(d2)
            temp_x3.append(d3)

        x0 = torch.cat(temp_x0, dim=1)
        x1 = torch.cat(temp_x1, dim=1)
        x2 = torch.cat(temp_x2, dim=1)
        x3 = torch.cat(temp_x3, dim=1)
        del temp_x0, temp_x1, temp_x2, temp_x3
        
        f0 = self.conv0(x0) # 64x128x128
        f1 = self.conv1(x1) # 128x64x64
        f2 = self.conv2(x2) # 256x32x32
        f3 = self.conv3(x3) # 512x16x16
        f4 = self.DP4(self.scaling(f3)) # 512x8x8

        f23 = self.trans23(f2)
        f33 = self.trans33(f3)
        f12 = self.trans12(f1)
        f22 = self.trans22(f2)
        f01 = self.trans01(f0)
        f11 = self.trans11(f1)
        f001 = self.trans001(f0)
        f002 = self.trans002(input[:,-1].unsqueeze(1))

        post_f4 = self.CDPb(self.CDP4(f4))

        f43 = self.CDP_up(post_f4)
        concat_f3 = torch.cat((f33,f43,f23),dim=1)
        post_f3 = self.CDPb(self.CDPa(concat_f3))

        f32 = self.CDP_up(post_f3)
        concat_f2 = torch.cat((f22,f32,f12),dim=1)
        post_f2 = self.CDPb(self.CDPa(concat_f2))

        f21 = self.CDP_up(post_f2)
        concat_f1 = torch.cat((f11,f21,f01),dim=1)
        post_f1 = self.CDPb(self.CDPa(concat_f1))

        f10 = self.CDP_up(post_f1)
        concat_f0 = torch.cat((f001,f10,f002),dim=1)
        post_f0 = self.CDP0(concat_f0)

        last = self.conv_out(post_f0)
        return last