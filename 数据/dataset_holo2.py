import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torch
import torch.utils.data as data

root_path = 'H:/SHUJU/denoising/data/7'
class ALLDataset2(data.Dataset):
    def __init__(self, data_choose='train', cut_number=0):
        self.root = root_path
        self.data_choose = data_choose
        self.img_ids = list()
        self.cut_number = cut_number
        #根据data_choose选择是制作训练集还是数据集
        if self.data_choose == 'train':
            img_list = os.path.join(self.root, 'train_picture', 'train.txt')
            with open(img_list, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    fname = int(line.strip()[:-4])
                    self.img_ids.append(
                        fname,
                    )
        elif self.data_choose == 'test':
            self.img_ids.append(self.cut_number)
        else:
            print('Forget to choose data_choose')
            exit()
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        face_idx = self.img_ids[index]
        #输入数据和标签数据路径拼接
        if self.data_choose == 'train':
            input_path = os.path.join(self.root, 'train_picture', str(self.cut_number), 'input', str(face_idx) + '.bmp')
            label_path = os.path.join(self.root, 'train_picture', str(self.cut_number), 'label', str(face_idx) + '.bmp')
        elif self.data_choose == 'test':
            input_path = os.path.join(self.root, 'cut_picture', str(self.cut_number)+'.bmp')
            label_path = os.path.join(self.root, 'cut_picture', str(self.cut_number+1) + '.bmp')
        #读取全息图
        input = cv2.imread(input_path, 0)
        label = cv2.imread(label_path, 0)
        return np.array(input), np.array(label)

