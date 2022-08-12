# holo-denoising
自监督去噪
该方法训练无需数据集，只需要一张带噪图像自身即可完成整个训练过程。

对于普通图像的测试效果存在img文件夹中。

第一步：先对去噪目标进行BM3D.py去噪生成BM3D.BMP。

第二步：再设定好预处理.py中args的默认去噪图像地址参数和默认的保存地址位置即可生成训练的数据集。

第三步：指定好data_holo2.py中的root地址即可。

第四步：model可以选择CNN,U-Net,和MSDNET。注：MSDNET中存在dropout模块，需要额外的test文件。

第五步：train.py
