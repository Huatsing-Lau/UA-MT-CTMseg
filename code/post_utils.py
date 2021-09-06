# -*- coding: utf-8 -*-
# ########## 测试结果后处理所用到的函数 #############

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import SimpleITK as sitk
import torch
import torchvision
import torchvision.transforms as transforms

# 二分类:
def get_proc_df_binary(results_raw,metric):
    # 重新整理数据表,以便适应sns.violinplot
    proc_df = pd.DataFrame(columns=[metric,'category','patient'])
    proc_df[metric] = pd.to_numeric(results_raw[metric])
    proc_df[metric] = proc_df[metric].astype('float')
    proc_df.category = np.array([metric]*len(results_raw)).flatten().tolist()
    proc_df.category = proc_df.category.astype('category')
    proc_df.patient = results_raw['Unnamed: 0'].values.tolist()
    return proc_df

def get_voilinplot_binary(proc_df,metric,filename):
    # # https://cloud.tencent.com/developer/article/1486970
    plt.figure(dpi=100)
    ax = sns.violinplot( data=proc_df, x='category', y=metric, scale="width",palette = 'RdBu' )
    ax = sns.swarmplot(data=proc_df, x='category', y=metric, color=".25", size=4)
    # 添加图形标题
    plt.title(metric)
    # x
    plt.xticks([1],[''])
    plt.xlabel('')
    # 保存图片
    plt.savefig(filename)
    # 显示图形
    plt.show()

def get_boxplot_binary(proc_df,metric,filename):
    # # https://cloud.tencent.com/developer/article/1486970
    plt.figure(dpi=100)
    ax = sns.boxplot( data=proc_df, x='category', y=metric, palette = 'RdBu' )
    ax = sns.swarmplot(data=proc_df, x='category', y=metric, color=".25", size=4)
    # 添加图形标题
    plt.title(metric)
    # x
    plt.xticks([1],[''])
    plt.xlabel('')
    # 保存图片
    plt.savefig(filename)
    # 显示图形
    plt.show()



# 多分类:
def get_proc_df(results_raw,metric,name_classes):
    # 重新整理数据表,以便适应sns.violinplot
    proc_df = pd.DataFrame(columns=[metric,'category','patient'])
    proc_df[metric] = pd.to_numeric(results_raw[metric][name_classes].values.flatten())
    proc_df[metric] = proc_df[metric].astype('float')
    proc_df.category = np.array([ [name]*len(results_raw) for name in name_classes ]).flatten().tolist()
    proc_df.category = proc_df.category.astype('category')
    proc_df.patient = results_raw[('Unnamed: 0_level_0','Unnamed: 0_level_1')].values.tolist()*len(name_classes)
    return proc_df

def get_voilinplot(proc_df,metric,name_classes,filename):
    # # https://cloud.tencent.com/developer/article/1486970
    plt.figure(dpi=100)
    ax = sns.violinplot( data=proc_df, x='category', y=metric, order=name_classes,scale="width",palette = 'RdBu' )
    ax = sns.swarmplot(data=proc_df, x='category', y=metric, order=name_classes, color=".25", size=4)
    # 添加图形标题
    plt.title(metric)
    # 保存图片
    plt.savefig(filename)
    # 显示图形
    plt.show()

def get_boxplot(proc_df,metric,name_classes,filename):
    # # https://cloud.tencent.com/developer/article/1486970
    plt.figure(dpi=100)
    ax = sns.boxplot( data=proc_df, x='category', y=metric, order=name_classes, palette = 'RdBu' )
    ax = sns.swarmplot(data=proc_df, x='category', y=metric, order=name_classes, color=".25", size=4)
    # 添加图形标题
    plt.title(metric)
    # 保存图片
    plt.savefig(filename)
    # 显示图形
    plt.show()

    

    


# gird 图:

def get_images(results_dir,patients,H=140, W=140):
    """
    return: 
        images: tensor
    """
    images = tuple()
    for patient in patients:
        image_sitk = sitk.ReadImage(os.path.join(results_dir, patient+'_img.nii.gz'))
        label_gt_sitk = sitk.ReadImage(os.path.join(results_dir, patient+'_gt.nii.gz'))
        label_pred_sitk = sitk.ReadImage(os.path.join(results_dir, patient+'_pred.nii.gz'))

        image = sitk.GetArrayFromImage(image_sitk).transpose([2,1,0])
        image = (image-image.min())/(image.max()-image.min())*255
        label_gt = sitk.GetArrayFromImage(label_gt_sitk).transpose([2,1,0])/2*255
        label_pred = sitk.GetArrayFromImage(label_pred_sitk).transpose([2,1,0])/2*255

        # 图像必须大小相同
        hmin,hmax = int((image.shape[0]-H)/2), int((image.shape[0]-H)/2+H)
        wmin,wmax = int((image.shape[1]-W)/2), int((image.shape[1]-W)/2+W)
        image = image[hmin:hmax,wmin:wmax,:]
        label_gt = label_gt[hmin:hmax,wmin:wmax,:]

        label_pred = label_pred[hmin:hmax,wmin:wmax,:]
        # case 拼接
        numd = []
        for d in range(label_gt.shape[2]):
            numd.append( len(np.where(label_gt[:,:,d].flatten()==255)[0]) )
        numd = np.array(numd)
        slice = np.where(numd==numd.max())[0][0]
    #     print('slice',slice)
        images = images+(image[:,:,slice][np.newaxis,np.newaxis,:,:],
                         label_gt[:,:,slice][np.newaxis,np.newaxis,:,:], 
                         label_pred[:,:,slice][np.newaxis,np.newaxis,:,:])
    images = np.concatenate(images, axis=0)
    
    print('images.shape:\n',images.shape)
    
    images = torch.tensor(images)
    return images

# functions to show an image
def imshow(img,n_case,filename=None):
    fig = plt.figure( dpi=300, frameon=False)
    npimg = img.numpy()
    npimg = np.transpose(npimg, (2, 1, 0))
    plt.imshow(npimg.astype(int),cmap='plasma')#一定要转为int
    xticks = [npimg.shape[1]/n_case*(i+0.5) for i in range(n_case)]
    yticks = [npimg.shape[0]/3*0.5,npimg.shape[0]/3*1.5,npimg.shape[0]/3*2.5]
    xlabels = ['case '+str(int(i)) for i in range(n_case)]
    plt.xticks(xticks, xlabels,rotation=45) 
    plt.yticks(yticks, ['image', 'gt', 'pred'],rotation=45) 
    if filename:
        plt.savefig(filename)
    plt.show()
