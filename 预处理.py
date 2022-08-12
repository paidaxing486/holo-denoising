import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy.io as scio
import argparse

def image_cut_save(path, left, upper, right, lower, save_path,size):
    img = cv2.imread(path)
    juzhen = np.zeros((size,size,3))
    juzhen[:,:,:] = img[upper:lower,left:right,:]
    #调用matlab程序
    cv2.imwrite(save_path, juzhen)
def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差：其数值为0-1
    返回:
        gaussian_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 将图片灰度标准化
    #img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 255)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out


if __name__ == '__main__':
    #设置基础参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--pci_path', type=str, default='H:/SHUJU/denoising/1.bmp', help='分割图像的源地址')
    parser.add_argument('--pic_save_dir_path', type=str, default='H:/SHUJU/denoising/data/1/cut_picture/', help='分割完后的图像保存的地址')
    parser.add_argument('--cut_save_dir_path', type=str, default='H:/SHUJU/denoising/data/1/train_picture/', help='加噪后图像保存的地址')
    parser.add_argument('--size', type=int, default=128, help='分割后图像的大小')
    parser.add_argument('-choose_number', type=int, default=10, help='选择训练的那张图像')
    args = parser.parse_args()
    #第一步切片图像
    pic_path = args.pci_path
    pic_save_dir_path = args.pic_save_dir_path
    cut_save_dir_path = args.cut_save_dir_path
    os.makedirs(pic_save_dir_path, exist_ok=True)
    os.makedirs(cut_save_dir_path, exist_ok=True)
    img = cv2.imread(pic_path)
    size = args.size
    W, H, D = img.shape
    w = W // args.size#高
    print(w)
    h = H // args.size#宽
    print(h)
    d = 0
    for i in range(1, h+1):
        for j in range(1, w+1):
            d2 = str(d)
            path = os.path.join(pic_save_dir_path, d2 + '.bmp')
            left, upper, right, lower = args.size*(i-1), args.size*(j-1), args.size*i, args.size*j
            image_cut_save(pic_path, left, upper, right, lower, path, size)
            d += 1
    #第二步，添加噪声
    #先提均值和均方差
    list = os.listdir(pic_save_dir_path)
    # 读取.mat文件中存的噪声均值和均方差
    data_path = os.path.join(pic_save_dir_path, '0.mat')
    if os.path.exists(data_path) == True:
        dict_data = scio.loadmat(data_path)
        array_data = dict_data['data']
        mean, sigma = array_data[0]
        #读取要加噪的切片图
        img_path = os.path.join(pic_save_dir_path, str(args.choose_number) + '.bmp')
        cut_img = cv2.imread(img_path, 0)
        for j2 in range(0, 500):
            #首先创建目录
            mkdir_path = os.path.join(cut_save_dir_path, str(args.choose_number))
            os.makedirs(mkdir_path, exist_ok=True)
            input_path = os.path.join(mkdir_path, 'input')
            label_path = os.path.join(mkdir_path, 'label')
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(label_path, exist_ok=True)
            #创建完后开始加噪
            noise_input_img = gaussian_noise(cut_img, 0, sigma)
            noise_label_img = gaussian_noise(cut_img, 0, sigma)
            #保存图像
            input_save_path = os.path.join(input_path, str(j2)+'.bmp')
            label_save_path = os.path.join(label_path, str(j2)+'.bmp')
            cv2.imwrite(input_save_path, noise_input_img)
            cv2.imwrite(label_save_path, noise_label_img)

