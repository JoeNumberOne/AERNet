# -*- coding: utf-8 -*-
# time: 2023/11/27 20:13
# file: MyDataset.py
# author: Tommy Joe +
# Project: AERNet
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import os


class MyDataSet(Dataset):

    # 初始化为整个class提供全局变量，为后续方法提供一些量
    def __init__(self, root_dir, children_dir):
        # self
        self.root_dir = root_dir
        self.children_dir = children_dir
        self.path = os.path.join(self.root_dir, self.children_dir)
        self.name_list = os.path.join(self.path, 'A')
        # listdir方法会将路径下的所有文件名（包括后缀名）组成一个列表
        self.img_path_list = os.listdir(self.name_list)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, idx):
        img_name = self.img_path_list[idx]  # 只获取了文件名
        img_item_A_path = os.path.join(self.path, 'A', img_name)  # 每个图片A的位置
        img_item_B_path = os.path.join(self.path, 'B', img_name)  # 每个图片B的位置
        img_item_label_path = os.path.join(self.path, 'label', img_name)  # 每个图片B的位置
        # 读取图片
        img_A = self.transform(Image.open(img_item_A_path))
        img_B = self.transform(Image.open(img_item_B_path))
        img_label = self.transform(Image.open(img_item_label_path))
        # 返回图片
        return img_A, img_B, img_label

    def __len__(self):
        return len(self.img_path_list)


if __name__ == '__main__':
    # root_dir = "../HRCUS-CD/train"
    # children_dir = "A"
    root_dir = "../HRCUS-CD"
    children_dir = "train"
    my_dataset = MyDataSet(root_dir, children_dir)
    img_A, img_B, img_label = my_dataset[0]  # 返回一个元组，返回值就是__getitem__的返回值
    print(img_A.shape, img_B.shape, img_label.shape)
    # print(img_A, img_B, img_label)

    print(len(my_dataset))
