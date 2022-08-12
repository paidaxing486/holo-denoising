import torch
from torch import nn
import numpy as np

#定义残差块
class Resblock(nn.Module):
    def __init__(self,in_channel, out_channel):
        super(Resblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.relu = nn.ReLU(inplace=True)
        self.sample = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
    def forward(self, x):
        out = self.block(x)
        #x = self.sample(x)
        out = self.relu(out+x)
        return out


#定义前置的cnn模块
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(
            Resblock(32,32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            Resblock(64,64),
            Resblock(64,64)
                                        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True)
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.res_blocks(x)
        out = self.conv2(x)
        return out

#定义下采样模块
class block_down(nn.Module):

    def __init__(self, inp_channel, out_channel):
        super(block_down, self).__init__()
        self.conv1 = nn.Conv2d(inp_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#定义上采样模块
class block_up(nn.Module):

    def __init__(self, inp_channel, out_channel, y):
        super(block_up, self).__init__()
        self.up = nn.ConvTranspose2d(inp_channel, out_channel, 2, stride=2)
        self.conv1 = nn.Conv2d(inp_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.y = y

    def forward(self, x):
        x = self.up(x)
        x = torch.cat([x, self.y], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#搭建整个U-Net网络
class U_Net(nn.Module):

    def __init__(self):
        super(U_Net, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.CNN = CNN()
    def forward(self, x):
        out1 = self.CNN(x)
        block1 = block_down(1, 64).cuda()
        x1_use = block1(out1)
        x1 = self.maxpool(x1_use)
        block2 = block_down(64, 128).cuda()
        x2_use = block2(x1)
        x2 = self.maxpool(x2_use)
        block3 = block_down(128, 256).cuda()
        x3_use = block3(x2)
        x3 = self.maxpool(x3_use)
        block4 = block_down(256, 512).cuda()
        x4_use = block4(x3)
        x4 = self.maxpool(x4_use)
        block5 = block_down(512, 1024).cuda()
        x5 = block5(x4)
        block6 = block_up(1024, 512, x4_use).cuda()
        x6 = block6(x5)
        block7 = block_up(512, 256, x3_use).cuda()
        x7 = block7(x6)
        block8 = block_up(256, 128, x2_use).cuda()
        x8 = block8(x7)
        block9 = block_up(128, 64, x1_use).cuda()
        x9 = block9(x8)
        out2 = self.conv3(x9)
        return out2


if __name__=="__main__":
    test_input=torch.rand(4, 1, 512, 512)
    print("input_size:",test_input.size())
    model=U_Net()
    ouput=model(test_input)
    print("output_size:",ouput.size())
