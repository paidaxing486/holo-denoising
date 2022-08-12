import cv2
from model.model2 import U_Net, CNN
from model.MSDNet import MSDNet
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#from skimage.metrics import structural_similarity
import os
from data.dataset_holo2 import ALLDataset2
import torch.optim as optim
from datetime import datetime
from utils import NPCC,transimg,transnp
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import argparse


#读取基础参数
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoches', type=int, default=100, help='训练的轮次')
parser.add_argument('--train_batchsizes', type=int, default=4, help='训练集的batchsize大小')
parser.add_argument('--test_batchsizes', type=int, default=1, help='测试集的batchsize大小')
parser.add_argument('--lr', type=float, default=0.0001, help='学习率大小')
parser.add_argument('--cut_number', type=int, default=300, help='图像切割数量')
parser.add_argument('--cut_h', type=int, default=15, help='图像切割纵向数量')
parser.add_argument('--cut_w', type=int, default=20, help='图像切割横向数量')
parser.add_argument('--cut_size', type=int, default=512, help='图像切割后的大小')
parser.add_argument('--choose', type=str, default='train', help='选择训练还是测试')
parser.add_argument('--choose_number', type=int, default=0, help='选择训练的那张图像')
args = parser.parse_args()
#决定是否使用CUDA
use_CUDA = torch.cuda.is_available()

#加载数据集
trn_dataset = ALLDataset2(data_choose='train', cut_number=args.choose_number)
val_dataset = ALLDataset2(data_choose='test', cut_number=args.choose_number)
trn_dloader = torch.utils.data.DataLoader(dataset=trn_dataset, batch_size=args.train_batchsizes, shuffle=True)
val_dloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.test_batchsizes, shuffle=True)

#定义模型，优化器
if use_CUDA:
    #net = MSDNet(choose=args.choose).cuda()
    net = CNN().cuda()
else:
    #net = MSDNet(choose=args.choose)
    net = CNN()
#使用Adam优化器进行训练，看其收敛情况
#optimizer = optim.Adam(net.parameters(),lr=args.lr,betas=(0.9, 0.999),eps=1e-08,weight_decay=0,amsgrad=False)
#运用动量法作为优化器
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
criterion = nn.MSELoss()

#创建参数保存目录,d代表对应的第几张切片图像
output_dir = './outputs_{:%Y-%m-%d-%H-%M-%S}/'.format(datetime.now())
middle_dir = os.path.join(output_dir, 'result')
checkp_dir = os.path.join(middle_dir, '_checkpoints')
#logtxt_dir = os.path.join(output_dir, str(d),'log.txt')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(middle_dir, exist_ok=True)
os.makedirs(checkp_dir, exist_ok=True)
#开始训练
num_epoches = args.num_epoches
losses = []
losses1 = []
losses2 = []
losses3 = []


for epoch_idx in range(num_epoches):
    for batch_idx, (inputs,target) in enumerate(trn_dloader,start=1):
        #若选用的为gray格式图像，则该处数据集的大小为 train_batchsizes×N×M
        inputs = inputs.float()
        target = target.float()
        if use_CUDA:
            inputs, target = inputs.cuda(), target.cuda()
            #此处需要用unsqueeze在表示图像数量维度后面，即1的位置添加一个单独的维度
            inputs,target = inputs.unsqueeze(1),target.unsqueeze(1)
        else:
            inputs, target = inputs.unsqueeze(1),target.unsqueeze(1)
        #注意此处outputs1是前置的CNN的输出，outputs是U-Net的最终输出
        outputs1 = net(inputs)
        loss1 = criterion(outputs1,target)
        #loss = NPCC(outputs,target,args.train_batchsizes)#计算损失值
        optimizer.zero_grad()#清空梯度值
        loss1.backward()#反向传播
        optimizer.step()#随机梯度下降
        losses.append(loss1.data)#将损失值加入到列表中
        losses1.append(loss1.data)
        losses2.append(loss1.data)

        #每隔10个batch——size打印一次损失值
        if batch_idx % 10 == 0:
            _loss = sum(losses) / len(losses)
            log_str = ''
            log_str += '[%3d/%3d]' % (epoch_idx, num_epoches)
            log_str += '[%5d/%5d]' % (batch_idx, len(trn_dloader))
            log_str += '\t%.4f' % (_loss)
            print(log_str)
            losses = []

    #每个循环保存一次测试结果
    for batch_idx, (inputs, target) in enumerate(val_dloader, start=1):
        if batch_idx == 1:
            inputs = inputs.float()
            target = target.float()
            if use_CUDA:
                inputs, target = inputs.cuda(), target.cuda()
                inputs, target = inputs.unsqueeze(1), target.unsqueeze(1)
            else:
                inputs, target = inputs.unsqueeze(1), target.unsqueeze(1)
            with torch.no_grad():
                outputs1 = net(inputs)
            input_0 = transnp(inputs)
            target_0 = transnp(target)
            #outputs_0 = transnp(outputs1)
            outputs_1 = transnp(outputs1)
            psnr = compare_psnr(target_0, outputs_1)
            ssim = compare_ssim(target_0, outputs_1, multichannel=True)
            _dir = os.path.join(middle_dir, '%03d' % epoch_idx)
            os.makedirs(_dir, exist_ok=True)
            #cv2.imwrite(os.path.join(_dir, '%d_CNNpred.bmp' % batch_idx),outputs_0)
            cv2.imwrite(os.path.join(_dir, '%d_U-Netpred.bmp' % batch_idx), outputs_1)
            cv2.imwrite(os.path.join(_dir, '%d_input.bmp' % batch_idx), input_0)
            cv2.imwrite(os.path.join(_dir, '%d_target.bmp' % batch_idx), target_0)


    _loss2 = sum(losses2) / len(losses2)
    losses2 = []
    losses3.append(_loss2)
    # 每隔10次训练生产一次损失值走向图像
    if (epoch_idx+1) % 10 == 0:
        losses4 = np.array(losses3)
        number = np.arange(1, epoch_idx + 2, 1)
        plt.title('loss')
        plt.plot(number, losses3, color='y', linestyle='-', alpha=0.5)
        save_path = os.path.join(middle_dir, '%03d' % epoch_idx, 'loss.jpg')
        plt.savefig(save_path)
        if (epoch_idx+1) == num_epoches:
            plt.clf()
        dict_save_path = os.path.join(checkp_dir, '%03d_psnr%.2f.pth' % (epoch_idx, psnr))
        torch.save(net.state_dict(), dict_save_path)


