from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image

import random
import h5py
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from scipy import ndimage
from PIL import Image


class NPY_datasets(Dataset): #处理常规图像格式的2D医学图像数据集，支持训练/验证划分
    def __init__(self, path_Data, config, train=True):
        super(NPY_datasets, self)
        # 根据train标志选择加载训练集或验证集
        if train:
            images_list = sorted(os.listdir(path_Data+'train/images/')) #图像路径
            masks_list = sorted(os.listdir(path_Data+'train/masks/')) #掩码路径
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'train/images/' + images_list[i]
                mask_path = path_Data+'train/masks/' + masks_list[i]
                self.data.append([img_path, mask_path])
            self.transformer = config.train_transformer  #训练时数据增强
        else:
            images_list = sorted(os.listdir(path_Data+'val/images/'))
            masks_list = sorted(os.listdir(path_Data+'val/masks/'))
            self.data = []
            for i in range(len(images_list)):
                img_path = path_Data+'val/images/' + images_list[i]
                mask_path = path_Data+'val/masks/' + masks_list[i]
                self.data.append([img_path, mask_path]) #构建配对序列
            self.transformer = config.test_transformer

    #关键方法：加载单对图像和掩码，应用预处理
    def __getitem__(self, indx):
        img_path, msk_path = self.data[indx]
        img = np.array(Image.open(img_path).convert('RGB')) #加载RGB图像
        msk = np.expand_dims(np.array(Image.open(msk_path).convert('L')), axis=2) / 255 #加载灰度掩码并归一化
        img, msk = self.transformer((img, msk)) #应用变换
        return img, msk

    def __len__(self):
        return len(self.data)



def random_rot_flip(image, label): #随机旋转和翻译
    k = np.random.randint(0, 4) # 随机旋转0/90/180/270度
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2) # 随机水平/垂直翻转
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label): #随机旋转
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):  #综合数据增强器
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class Synapse_dataset(Dataset): # 数据集加载特点：训练集：从.npz中加载2D切片 验证集：从HDF5中加载3D体积数据
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, slice_name+'.npz')
            data = np.load(data_path) #加载NPZ文件
            image, label = data['image'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            data = h5py.File(filepath)  #加载HDF5文件
            image, label = data['image'][:], data['label'][:]

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)  #应用变换
        sample['case_name'] = self.sample_list[idx].strip('\n') #保留一本标识
        return sample
        
    