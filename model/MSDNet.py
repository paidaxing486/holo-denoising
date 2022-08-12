import math

import torch
from torch import nn

class Residual_Block(nn.Module):
    def __init__(self, dam, d):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(dam * 2 ** d, int(dam * 2 ** (d-1)), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(3 * dam * 2 ** (d - 1)), int(dam * 2 ** (d - 1)), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(4 * dam * 2 ** (d - 1)), dam * 2 ** d, kernel_size=3, stride=1, padding=1)
        self.PReLU = nn.PReLU()
        self.BN1 = nn.BatchNorm2d(int(dam * 2 ** (d-1)))
        self.BN2 = nn.BatchNorm2d(dam * 2 ** d)
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.BN1(x1)
        x1 = self.PReLU(x1)
        x2 = torch.cat([x, x1], dim=1)
        x3 = self.conv2(x2)
        x3 = self.BN1(x3)
        x3 = self.PReLU(x3)
        x4 = torch.cat([x2, x3], dim=1)
        x5 = self.conv3(x4)
        x5 = self.BN2(x5)
        x5 = self.PReLU(x5)
        out = x+x5
        return out
#dim代表传入特征图的层数，d代表第几个残差块，dropout为dropout的隐藏系数大小
class Dropout_Residual_Block(nn.Module):
    def __init__(self, dam, d, dropout):
        super(Dropout_Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(dam * 2 ** (d-1), int(dam * 2 ** (d-2)), kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(int(3 * dam * 2 ** (d - 2)), int(dam * 2 ** (d - 2)), kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(int(4 * dam * 2 ** (d - 2)), dam * 2 ** (d-1), kernel_size=3, stride=1, padding=1)
        self.PReLU = nn.PReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.BN1 = nn.BatchNorm2d(int(dam * 2 ** (d - 2)))
        self.BN2 = nn.BatchNorm2d(dam * 2 ** (d-1))

    def forward(self, x):
        x = self.dropout(x)
        x1 = self.conv1(x)
        x1 = self.BN1(x1)
        x1 = self.PReLU(x1)
        x2 = torch.cat([x, x1], dim=1)
        x2 = self.dropout(x2)
        x3 = self.conv2(x2)
        x3 = self.BN1(x3)
        x3 = self.PReLU(x3)
        x4 = torch.cat([x2, x3], dim=1)
        x4 = self.dropout(x4)
        x5 = self.conv3(x4)
        x5 = self.BN2(x5)
        x5 = self.PReLU(x5)
        out = x+x5
        return out

#图片传入后第一块预处理模块

class Process_block(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(Process_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel),
            nn.PReLU(),
            nn.Conv2d(output_channel, output_channel * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channel*2),
            nn.PReLU(),
        )
        self.Resblock = Residual_Block(dam=output_channel*2, d=0)

    def forward(self, x):
        x1 = self.layer(x)
        out = self.Resblock(x1)
        return out

#定义编码器模块，dim一般为128，变化d，从1到3
class Encoder_block(nn.Module):
    def __init__(self, dam, d):
        super(Encoder_block, self).__init__()
        self.pooling = nn.Sequential(
            nn.Conv2d(dam * 2 ** (d-1), dam * 2 ** d, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(dam * 2 ** d),
            nn.PReLU()
        )
        self.Resblock = Residual_Block(dam=dam, d=d)

    def forward(self, x):
        x1 = self.pooling(x)
        out = self.Resblock(x1)
        return out

#dim代表其传入的基础特征图层数，一般为128，d代表第几个解码器模块,首先是
class Deconder_block(nn.Module):
    def __init__(self, input_channel, dam, d, dropout):
        super(Deconder_block, self).__init__()
        self.TConv = nn.Sequential(
            nn.ConvTranspose2d(input_channel, dam * 2 ** (d - 1), kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(dam * 2 ** (d - 1)),
            nn.PReLU()
        )
        self.Resblock = Dropout_Residual_Block(dam=dam, d=d, dropout=dropout)

    def forward(self, x):
        x1 = self.TConv(x)
        out = self.Resblock(x1)
        return out


class Generate_block(nn.Module):
    def __init__(self, input_channel, middle_channel, output_channel, dropout):
        super(Generate_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(input_channel, middle_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(middle_channel),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(middle_channel, int(middle_channel/2), kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(int(middle_channel/2)),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(int(middle_channel/2), output_channel, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class MSDNet(nn.Module):
    def __init__(self,choose='train'):
        super(MSDNet, self).__init__()
        self.Process = Process_block(1, 64)
        self.Enconder1 = Encoder_block(128, 1)
        self.Enconder2 = Encoder_block(128, 2)
        self.Enconder3 = Encoder_block(128, 3)
        if choose == 'train':
            self.Deconder3 = Deconder_block(1024, 128, 3, 0.3)
            self.Deconder2 = Deconder_block(1024, 128, 2, 0.4)
            self.Deconder1 = Deconder_block(512, 128, 1, 0.5)
            self.Generate = Generate_block(256, 128, 1, 0.5)
        elif choose == 'test':
            self.Deconder3 = Deconder_block(1024, 128, 3, 0)
            self.Deconder2 = Deconder_block(1024, 128, 2, 0)
            self.Deconder1 = Deconder_block(512, 128, 1, 0)
            self.Generate = Generate_block(256, 128, 1, 0)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            n //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        #先预处理
        x1 = self.Process(x)
        #再传入编码器中
        x2 = self.Enconder1(x1)
        x3 = self.Enconder2(x2)
        x4 = self.Enconder3(x3)
        #再传入解码器中
        y4 = self.Deconder3(x4)
        concate3 = torch.cat([y4, x3], dim=1)
        y3 = self.Deconder2(concate3)
        concate2 = torch.cat([y3, x2], dim=1)
        y2 = self.Deconder1(concate2)
        concate1 = torch.cat([y2, x1], dim=1)
        out = self.Generate(concate1)
        return out

