import torch
import torch.nn as nn

class UXNetBlock(nn.Module):
    def __init__(self, h_dim,kernel_size=7,input_size=[32,32,32]):
        super().__init__()
        H, W, Z = input_size
        C = h_dim
        self.ConvLayer1 = nn.Sequential(
            nn.LayerNorm([C, H, W, Z]),
            nn.Conv3d(h_dim, h_dim, (kernel_size,kernel_size,1), 1,(int(kernel_size/2),int(kernel_size/2),0)),
        )
        # Normalize over the last four dimensions (i.e. the channel and spatial dimensions)
        self.ScalingLayer1 = nn.Sequential(
            nn.LayerNorm([C, H, W, Z]),
            nn.Conv3d(h_dim, h_dim, 1, 1),
            nn.GELU()
        )
        self.ConvLayer2 = nn.Sequential(
            nn.LayerNorm([C, H, W, Z]),
            nn.Conv3d(h_dim, h_dim, (kernel_size,kernel_size,3), 1,(int(kernel_size/2),int(kernel_size/2),1)),
        )
        self.ScalingLayer2 = nn.Sequential(
            nn.LayerNorm([C, H, W, Z]),
            nn.Conv3d(h_dim, h_dim, 1, 1),
            nn.GELU()
        )
    def forward(self, x):
        x1 = self.ConvLayer1(x)
        x2 = self.ScalingLayer1(x1+x)
        x3 = self.ConvLayer1(x2+x1)
        return self.ScalingLayer1(x3+x2)

class ResBlock3D(nn.Module):
    def __init__(self, in_channels,out_channels,stride=2,input_size=[32,32,32]):
        super().__init__()
        H, W, Z = input_size
        H, W, Z = int(H/stride), int(W/stride), int(Z)
        C = out_channels
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3,3,1), stride=(stride,stride,1),padding=(1,1,0),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,3), stride=1,padding=(1,1,1),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.skip_layer = nn.Conv3d(in_channels, out_channels, 1, 1)
        self.downsample = nn.Sequential()
        if stride != 1:
            self.downsample = nn.MaxPool3d((stride,stride,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        if self.same_channels:

            x_skip = self.downsample(x)
            out = x_skip + x2
            return out / 1.414
        else:
            x_skip = self.skip_layer(x)
            x_skip = self.downsample(x_skip)
            out = x_skip + x2
            return out / 1.414

class ResBlock3DLink(nn.Module):
    #仿照ACC-UNet: A Completely Convolutional UNet model for the 2020s的HANCBlock
    #https: // doi.org / 10.1007 / 978 - 3 - 031 - 43898 - 1_66
    def __init__(self, in_channels,inv_factor=2,input_size=[32,32,32]):
        super().__init__()
        H, W, Z = input_size
        C = in_channels*inv_factor
        '''
        standard ResNet style convolutional block
        '''

        self.conv0 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels*inv_factor, kernel_size=(3,3,1), stride=1,padding=(1,1,0),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )

        self.conv3c = nn.Sequential(
            nn.Conv3d(in_channels*inv_factor, in_channels*inv_factor, kernel_size=(3,3,3), stride=1,padding=(1,1,1),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.conv5c = nn.Sequential(
            nn.Conv3d(in_channels*inv_factor, in_channels*inv_factor, kernel_size=(3,3,5), stride=1,padding=(1,1,2),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.conv7c = nn.Sequential(
            nn.Conv3d(in_channels*inv_factor, in_channels*inv_factor, kernel_size=(3,3,7), stride=1,padding=(1,1,3),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.maxpool3c = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.maxpool5c = nn.MaxPool3d(kernel_size=(3, 3, 5), stride=1, padding=(1, 1, 2))
        self.maxpool7c = nn.MaxPool3d(kernel_size=(3, 3, 7), stride=1, padding=(1, 1, 3))

        self.avgpool3c = nn.AvgPool3d(kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1))
        self.avgpool5c = nn.AvgPool3d(kernel_size=(3, 3, 5), stride=1, padding=(1, 1, 2))
        self.avgpool7c = nn.AvgPool3d(kernel_size=(3, 3, 7), stride=1, padding=(1, 1, 3))

        self.pointwise_layer = nn.Conv3d(9*in_channels*inv_factor, in_channels, 1, 1)

        self.downsample = nn.Sequential()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv0(x)

        xc3c = self.conv3c(x1)
        xc5c = self.conv5c(x1)
        xc7c = self.conv7c(x1)

        xm3c = self.maxpool3c(x1)
        xm5c = self.maxpool5c(x1)
        xm7c = self.maxpool7c(x1)

        xa3c = self.avgpool3c(x1)
        xa5c = self.avgpool5c(x1)
        xa7c = self.avgpool7c(x1)

        x_skip = x
        x = torch.cat((xc3c,xc5c,xc7c,xm3c,xm5c,xm7c,xa3c,xa5c,xa7c), 1)
        x = self.pointwise_layer(x)
        out = x_skip + x
        return out / 1.414




class ResBlock3Dup(nn.Module):
    def __init__(self, in_channels,out_channels,scale_factor=2,input_size=[32,32,32]):
        super().__init__()
        H, W, Z = input_size
        H, W, Z = H*scale_factor, W*scale_factor, Z
        C = out_channels
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(scale_factor,scale_factor,1),
                               stride=(scale_factor,scale_factor,1),padding=0,bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,1), stride=1,padding=(1,1,0),bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.skip_layer = nn.Conv3d(in_channels, out_channels, 1, 1)
        self.downsample = nn.Sequential()
        if scale_factor != 1:
            self.upsample = nn.Upsample(scale_factor=(scale_factor,scale_factor,1), mode='nearest')

    def forward(self, x, x_cat):
        x = torch.cat((x, x_cat), 1)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)


        if self.same_channels:

            x_skip = self.upsample(x)
            out = x_skip + x2
            return out / 1.414
        else:
            x_skip = self.skip_layer(x)
            x_skip = self.upsample(x_skip)
            out = x_skip + x2
            return out / 1.414

class ResBlock3Dup0(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, input_size=[32, 32, 32]):
        super().__init__()
        H, W, Z = input_size
        H, W, Z = H * scale_factor, W * scale_factor, Z
        C = out_channels
        '''
        standard ResNet style convolutional block
        '''
        self.same_channels = in_channels == out_channels

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(scale_factor, scale_factor, 1),
                               stride=(scale_factor, scale_factor, 1), padding=0, bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=(3,3,1), stride=1, padding=(1, 1, 0), bias=False),
            nn.LayerNorm([C, H, W, Z]),
            nn.GELU(),
        )
        self.skip_layer = nn.Conv3d(in_channels, out_channels, 1, 1)
        self.downsample = nn.Sequential()
        if scale_factor != 1:
            self.upsample = nn.Upsample(scale_factor=(scale_factor,scale_factor,1), mode='nearest')

    def forward(self, x):
        #print(x.shape,1)
        x1 = self.conv1(x)
        #print(x.shape, 2)
        x2 = self.conv2(x1)

        if self.same_channels:

            x_skip = self.upsample(x)
            out = x_skip + x2
            return out / 1.414
        else:
            x_skip = self.skip_layer(x)
            x_skip = self.upsample(x_skip)
            out = x_skip + x2
            return out / 1.414


if __name__ == '__main__':
    w1 = nn.Upsample(scale_factor=2, mode='nearest')

    TestLayer1 = UXNetBlock(32,kernel_size=7,input_size=[64,64,12])
    TestLayer2 = ResBlock3D(in_channels=32,out_channels=64,stride=2,input_size=[64,64,12])
    TestLayer3 = ResBlock3Dup0(in_channels=64,out_channels=32,scale_factor=2,input_size=[32,32,12])
    TestData = torch.randn((16,32,64,64,12))

    print(w1(TestData).shape)
    print(TestData.shape)
    TestData = TestLayer1(TestData)
    print(TestData.shape)
    TestData = TestLayer2(TestData)
    print(TestData.shape)
    TestData = TestLayer3(TestData)
    print(TestData.shape)
    TestLayer4 = ResBlock3DLink(in_channels=32, inv_factor=2, input_size=[64, 64, 12])
    TestData = TestLayer4(TestData)
    print(TestData.shape)
