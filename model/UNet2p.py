import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
    # inplanes代表输入通道数，planes代表输出通道数。
        super(ResBlock, self).__init__()
        # Conv
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        self.relu = nn.ReLU(inplace=True)

        self.skip_layer = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):

        out = self.conv(x)
        residual = self.skip_layer(x)
        out = (residual + out) / 1.414
        out = self.relu(out)

        return out


class ResBlockUp(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, upsample=2):
    # inplanes代表输入通道数，planes代表输出通道数。
        super(ResBlockUp, self).__init__()
        # Conv
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(planes, planes, kernel_size=upsample, stride=upsample, bias=False),
            nn.BatchNorm2d(planes)
        )

        self.relu = nn.ReLU(inplace=True)

        self.skip_layer = nn.Sequential(
            nn.Upsample(scale_factor=upsample, mode='nearest'),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):

        out = self.conv(x)
        residual = self.skip_layer(x)
        #print(residual.shape,out.shape)
        out = (residual + out) / 1.414
        out = self.relu(out)

        return out

class UNetppL3(nn.Module):
    def __init__(self,h_dim=64):
        super(UNetppL3, self).__init__()
        self.convx0_0 = ResBlock(1, h_dim, stride=2) #/2
        self.convx0_0_Down = ResBlock(h_dim, h_dim, stride=2)  # output: /4

        self.convx1_0 = ResBlock(h_dim * 1, h_dim * 1) #/4
        self.convx1_0_Down = ResBlock(h_dim, h_dim * 2, stride=2)  # output: /8
        self.convx1_0_Up = ResBlockUp(h_dim * 1, h_dim * 1, upsample=2)  # output: /4

        self.convx2_0 = ResBlock(h_dim * 2, h_dim * 2) # input: downsample 10, output: /8
        self.convx2_0_Down = ResBlock(h_dim * 2, h_dim * 4, stride=2)  # output: /16
        self.convx2_0_Up = ResBlockUp(h_dim * 2, h_dim * 1, upsample=2)  # output: /4

        self.convx3_0 = ResBlock(h_dim * 4, h_dim * 4) # input: downsample 20, output: /16
        self.convx3_0_Up = ResBlockUp(h_dim * 4, h_dim * 2, upsample=2)  # output: /8

        self.convx2_1 = ResBlock(h_dim * (2 + 2), h_dim * 4) # input: 20 and upsample 30, output: /8
        self.convx2_1_Up = ResBlockUp(h_dim * 4, h_dim * 2, upsample=2) # output: /4

        self.convx1_1 = ResBlock(h_dim * (1 + 1), h_dim * 2)  # input: 10 and upsample 20, output: /4
        self.convx1_1_Up = ResBlockUp(h_dim * 2, h_dim * 1, upsample=2)  # output: /2

        self.convx1_2 = ResBlock(h_dim * (2 + 2), h_dim * 4)  # input: 11 and upsample 21, output: /4
        self.convx1_2_Up = ResBlockUp(h_dim * 4, h_dim * 2, upsample=2)  # output: /2

        self.convx0_1 = ResBlock(h_dim * (1 + 1), h_dim)  # input: 00 and upsample 10, output: /2
        self.convx0_2 = ResBlock(h_dim * (1 + 1), h_dim)  # input: 10 and upsample 11, output: /2
        self.convx0_3 = ResBlock(h_dim * (2 + 1), h_dim)  # input: 20 and upsample 12, output: /2

        self.convx0_1_final = nn.Sequential(
            ResBlockUp(h_dim, h_dim, upsample=2),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.Softmax2d(),
        ) # UNetppL1 final output
        self.convx0_2_final = nn.Sequential(
            ResBlockUp(h_dim, h_dim, upsample=2),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.Softmax2d(),
        ) # UNetppL2 final output
        self.convx0_3_final =  nn.Sequential(
            ResBlockUp(h_dim, h_dim, upsample=2),
            nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(h_dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
            #nn.Softmax2d(),
        ) # UNetppL3 final output

    def forward(self, x):
        x0_0 = self.convx0_0(x)
        x0_0_Down = self.convx0_0_Down(x0_0)

        x1_0 = self.convx1_0(x0_0_Down)
        x1_0_Down = self.convx1_0_Down(x1_0)
        x1_0_Up = self.convx1_0_Up(x1_0)

        x2_0 = self.convx2_0(x1_0_Down)
        x2_0_Down = self.convx2_0_Down(x2_0)
        x2_0_Up = self.convx2_0_Up(x2_0)

        x3_0 = self.convx3_0(x2_0_Down)
        x3_0_Up = self.convx3_0_Up(x3_0)

        x2_1 = self.convx2_1(torch.cat([x3_0_Up,x2_0],dim=1)) # input: 20 and upsample 30, output: /8
        x2_1_Up = self.convx2_1_Up(x2_1)

        x1_1 = self.convx1_1(torch.cat([x2_0_Up,x1_0],dim=1))  # input: 10 and upsample 20, output: /4
        x1_1_Up = self.convx1_1_Up(x1_1)

        x1_2 = self.convx1_2(torch.cat([x2_1_Up,x1_1],dim=1))  # input: 11 and upsample 21, output: /4
        x1_2_Up = self.convx1_2_Up(x1_2)



        x0_1 = self.convx0_1(torch.cat([x1_0_Up,x0_0],dim=1))  # input: 00 and upsample 10, output: /2
        #print(x1_1_Up.shape, x0_1.shape)
        x0_2 = self.convx0_2(torch.cat([x1_1_Up,x0_1],dim=1))  # input: 01 and upsample 11, output: /2
        #print(x1_2_Up.shape, x0_2.shape)
        x0_3 = self.convx0_3(torch.cat([x1_2_Up,x0_2],dim=1))  # input: 02 and upsample 12, output: /2

        return [self.convx0_1_final(x0_1),self.convx0_2_final(x0_2),self.convx0_3_final(x0_3)]

if __name__ == '__main__':
    a = torch.randn((16,1,256,256)).to(device)

    model = UNetppL3(h_dim=32).to(device)
    print(model(a)[-1].shape)