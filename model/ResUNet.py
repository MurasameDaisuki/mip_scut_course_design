import torch
import torch.nn as nn
from model.ACC_UNetBlock import UXNetBlock,ResBlock3D,ResBlock3Dup,ResBlock3Dup0,ResBlock3DLink

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResUNet(nn.Module):
    def __init__(self,h_dim=48,input_size=(64,64,64)):
        super(ResUNet, self).__init__()
        H, W, Z = input_size
        scale_factor = 1
        H, W = int(H/scale_factor), int(W/scale_factor)

        self.UNetDown0 = nn.Sequential(
            nn.Conv3d(1, h_dim, kernel_size=(27,27,1),
                      stride=(2*scale_factor,2*scale_factor,1),
                      padding=(13,13,0)),
        )

        self.UNetDown1 = nn.Sequential(
            ResBlock3D(h_dim, h_dim, stride=1, input_size=[int(H / 2), int(W / 2), int(Z)]),
            ResBlock3D(h_dim, h_dim*2, stride=2, input_size=[int(H / 2), int(W / 2), int(Z)]),
        )
        self.UNetDown2 = nn.Sequential(
            ResBlock3D(h_dim*2, h_dim*2, stride=1, input_size=[int(H / 4), int(W / 4), int(Z)]),
            ResBlock3D(h_dim*2, h_dim*4, stride=2, input_size=[int(H / 4), int(W / 4), int(Z)]),
        )
        self.UNetDown3 = nn.Sequential(
            ResBlock3D(h_dim*4, h_dim*4, stride=1, input_size=[int(H / 8), int(W / 8), int(Z)]),
            ResBlock3D(h_dim*4, h_dim*8, stride=2, input_size=[int(H / 8), int(W / 8), int(Z)]),
        )
        self.UNetDown4 = nn.Sequential(
            ResBlock3D(h_dim*8, h_dim*8, stride=1, input_size=[int(H / 16), int(W / 16), int(Z)]),
            ResBlock3D(h_dim*8, h_dim*16, stride=2, input_size=[int(H / 16), int(W / 16), int(Z)]),
        )
        #(in_channels=32, inv_factor=2, input_size=[64, 64, 12])
        self.UNetLink0 = nn.Sequential(
            ResBlock3D(h_dim, h_dim, stride=1, input_size=[int(H / 2), int(W / 2), int(Z)]),
        )
        self.UNetLink1 = nn.Sequential(
            ResBlock3D(h_dim*2, h_dim*2, stride=1, input_size=[int(H / 4), int(W / 4), int(Z)]),
        )
        self.UNetLink2 = nn.Sequential(
            ResBlock3D(h_dim*4, h_dim*4, stride=1, input_size=[int(H / 8), int(W / 8), int(Z)]),
        )
        self.UNetLink3 = nn.Sequential(
            ResBlock3D(h_dim*8, h_dim*8, stride=1, input_size=[int(H/16), int(W/16), int(Z)]),
        )
        self.UNetLink4 = nn.Sequential(
            ResBlock3D(h_dim*16, h_dim*16, stride=1, input_size=[int(H/32), int(W/32), int(Z)]),
        )

        self.UNetUp4 = ResBlock3Dup0(h_dim*16, h_dim*8, scale_factor=2, input_size=[int(H/32), int(W/32), int(Z)])

        self.UNetUp3 = ResBlock3Dup(h_dim*16, h_dim*4, scale_factor=2, input_size=[int(H/16), int(W/16), int(Z)])

        self.UNetUp2 = ResBlock3Dup(h_dim*8, h_dim*2, scale_factor=2, input_size=[int(H/8), int(W/8), int(Z)])

        self.UNetUp1 = ResBlock3Dup(h_dim*4, h_dim*1, scale_factor=2, input_size=[int(H/4), int(W/4), int(Z)])

        self.UNetUp0 = ResBlock3Dup(h_dim*2, h_dim, scale_factor=2*scale_factor, input_size=[int(H/2), int(W/2), int(Z)])

        self.UNetOutput = nn.Sequential(
            ResBlock3D(h_dim, h_dim, stride=1, input_size=[int(H*scale_factor), int(W*scale_factor), int(Z)]),
            ResBlock3D(h_dim, 1, stride=1, input_size=[int(H*scale_factor), int(W*scale_factor), int(Z)]),
            #nn.Sigmoid()
        )


    def forward(self, x):
        #print(x.shape)
        x0 = self.UNetDown0(x)
        #print(x0.shape)
        x1 = self.UNetDown1(x0)
        x2 = self.UNetDown2(x1)
        x3 = self.UNetDown3(x2)
        x4 = self.UNetDown4(x3)
        #print(x0.shape,x1.shape,x2.shape,x3.shape)

        x0 = self.UNetLink0(x0)
        x1 = self.UNetLink1(x1)
        x2 = self.UNetLink2(x2)
        x3 = self.UNetLink3(x3)
        x4 = self.UNetLink4(x4)

        x_up4 = self.UNetUp4(x4)
        #del x4
        #print(x_up4.shape, x3.shape)
        x_up3 = self.UNetUp3(x_up4, x3)
        #del x_up4, x3
        x_up2 = self.UNetUp2(x_up3, x2)
        #del x_up3, x2
        x_up1 = self.UNetUp1(x_up2, x1)
        #del x_up2, x1
        #print(x_up1.shape, x0.shape,x1.shape)
        x_up0 = self.UNetUp0(x_up1, x0)
        #print(x_up0.shape)
        #del x_up1, x0

        output = self.UNetOutput(x_up0)


        return output

if __name__ == '__main__':
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(meminfo.total / 1024 ** 2)  # 总的显存大小
    print(meminfo.used / 1024 ** 2)  # 已用显存大小
    print(meminfo.free / 1024 ** 2)  # 剩余显存大小


    TestLayer1 = ACC_UXNet(h_dim=12,input_size=(256,256,12)).to(device)
    print(type(TestLayer1.state_dict()))  # 查看state_dict所返回的类型，是一个“顺序字典OrderedDict”

    for param_tensor in TestLayer1.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        print(param_tensor, '\t', TestLayer1.state_dict()[param_tensor].size())
    TestData = torch.randn((1,1,256,256,12)).to(device)

    TestData = TestLayer1(TestData)
    print(TestData.shape)
