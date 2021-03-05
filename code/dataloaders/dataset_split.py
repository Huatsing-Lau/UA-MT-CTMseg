# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 16:00:14 2020

@author: liuhuaqing
"""

import os
from glob import glob
# import numpy as np
import random

def data_split(full_list, ratio, shuffle=True, seed=42):
    """
    数据集拆分: 将列表full_list按比例ratio（随机）划分为2个子列表sublist_1与sublist_2
    :param full_list: 数据列表
    :param ratio:     子列表1
    :param shuffle:   子列表2
    :return:
    """
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.seed(seed)
        random.shuffle(full_list,)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1,sublist_2

def save_list2txt(list_variable,fn):
    f = open(fn,'w')
    for item in list_variable:
        f.write(item+'\n')
    f.close()


def dataset_split(path,save_dir,ratio=0.8):
    """读取数据集文件夹，该函数将病例分为训练集和测试集，生成两个txt文件"""
    re = os.path.join(path,'*/mri_norm2.h5')
    listt = glob(re)
    names = []
    names += [ fn.split('/')[-2] for fn in listt ]#win系统改为'\\'

    names_train,names_test = data_split(names, ratio, shuffle=True, seed=42)
    
    fn_train = os.path.join(save_dir,'train.list')
    fn_test = os.path.join(save_dir,'test.list')
    save_list2txt(names_train, fn_train)
    save_list2txt(names_test, fn_test)
    return names_train,names_test

def make_dataset_list(path,save_dir):
    """该函数读取数据集文件夹，列出所有病例，生成一个*.list文件"""
    re = os.path.join(path,'*/mri_norm2.h5')
    listt = glob(re)
    names = []
    names += [ fn.split('/')[-2] for fn in listt ]#win系统改为'\\'
    
    fn = os.path.join(save_dir,'train_unseg.list')
    save_list2txt(names, fn)

def remove_files(re):
    """读取所有匹配re规则的文件并删除"""
    listt = glob(re)
    for fn in listt:
        os.remove(fn)
    return listt

if __name__ == '__main__':
    # 有标签数据
    dataset_dir = 'E:/TsingHua-PearlRiverDelta-Phase1/CTM/CTM/CTM_data/Segmented'
    save_dir = 'E:/TsingHua-PearlRiverDelta-Phase1/CTM/CTM/CTM_data'
    dataset_split(path=dataset_dir,save_dir=save_dir)
    # 无标签数据
    dataset_dir = '../../data/CTM_dataset/Segmented'
    make_dataset_list(path=dataset_dir,save_dir=save_dir)
