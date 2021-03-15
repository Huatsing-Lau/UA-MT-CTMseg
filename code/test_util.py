# -*- coding: utf-8 -*-
# +
import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os


def predict_and_center_cut_all_case(net, image_list, num_classes, 
                        patch_size=(112, 112, 80), stride_xy=18, stride_z=4, 
                        save_result=True, test_save_path=None, preproc_fn=None,
                        device='cpu'):
    for image_path in tqdm(image_list):
        id = image_path.split('/')[-2]
        print(id,':')
        out_dir = test_save_path+id
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)
            
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        label_pred, score_map = test_single_case(
            net, image, 
            stride_xy, stride_z, patch_size, 
            num_classes=num_classes, 
            device=device)
        
        import pdb
        pdb.set_trace()
        filter_mask = filter_connected_domain(label_pred,num_keep_region=None,ratio_keep=0.001)
        filter_mask = (filter_mask>0).astype(float)
        import pdb
        pdb.set_trace()
        label_pred = label_pred*filter_mask

        # 发现圆形视场的边界处经常出现错误分割(轮廓线),因此需要手动过滤
        r = label_pred.shape[0]/2
        xc,yc = label_pred.shape[0]/2,label_pred.shape[0]/2
#         filter_mask = np.ones(label_pred.shape)
#         for x in range(label_pred.shape[0]):
#             for y in range(label_pred.shape[1]):
#                 filter_mask[x,y,:] = 0 if r*0.5<np.sqrt((x-xc)**2+(y-yc)**2)<r*2 else 1
#         label_pred = filter_mask*label_pred
        
        import pdb
        pdb.set_trace()
        # onehot
        label_onehot_pred = tf.keras.utils.to_categorical(label_pred)
        if not label_onehot_pred.shape[-1]==3:
            print(id+' onehot shape error: miss one or more pixel class')
            continue
            
        # center cut
        tempL = np.nonzero(label_pred)
        minx, maxx = np.min(tempL[0]).astype(int), np.max(tempL[0]).astype(int)
        miny, maxy = np.min(tempL[1]).astype(int), np.max(tempL[1]).astype(int)
        minz, maxz = np.min(tempL[2]).astype(int), np.max(tempL[2]).astype(int)
        image = image[minx:maxx+1, miny:maxy+1, minz:maxz+1]
        label_pred = label_pred[minx:maxx+1, miny:maxy+1, minz:maxz+1]
        label_onehot_pred = label_onehot_pred[minx:maxx+1, miny:maxy+1, minz:maxz+1, :]
            
        # case 拼接
        numd = []
        for d in range(label_pred.shape[2]):
            numd.append( len(np.where(label_pred[:,:,d].flatten()==2)[0]) )
        numd = np.array(numd)
        slice = int(np.where(numd==numd.max())[0][0])
        fig = plt.figure( frameon=False)#dpi=100, 
        image_unstd = (image-image.min())/(image.max()-image.min())*255
        npimg = np.append( image_unstd[:,:,slice],label_pred[:,:,slice]/2*255,axis=1 )
        plt.imshow(npimg.astype(int),cmap='plasma')#一定要转为int
        plt.savefig( test_save_path + id + str(slice) + "_pred.png" )
        plt.show()
        
        import pdb
        pdb.set_trace()
        
        if save_result:
            # save files
            filename = os.path.join(os.path.dirname(image_path),'center_cut.h5')
            f = h5py.File(filename, 'w')
            f.create_dataset('image', data=image.astype(np.float32), compression="gzip")
#             f.create_dataset('label', data=label_onehot_pred.astype(np.int), compression="gzip")
            f.close()
#             nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), 
#                      out_dir+ '/' + id +'_minx%d_maxx%d_miny%d_maxy%d_minz%d_maxz%d'%(minx,maxx,miny,maxy,minz,maxz)+ "_img.nii.gz")
#             nib.save(nib.Nifti1Image(label_pred.astype(np.float32), np.eye(4)), 
#                      out_dir+ '/' + id +'_minx%d_maxx%d_miny%d_maxy%d_minz%d_maxz%d'%(minx,maxx,miny,maxy,minz,maxz)+ "_pred.nii.gz")
#             nib.save(nib.Nifti1Image(label_onehot_pred[:].astype(np.float32), np.eye(4)), 
#                      out_dir+ '/' + id +'_minx%d_maxx%d_miny%d_maxy%d_minz%d_maxz%d'%(minx,maxx,miny,maxy,minz,maxz)+ "_label_onehot_pred.nii.gz")
    print('All finished')


# -

from skimage import measure
def filter_connected_domain(image,num_keep_region=100,ratio_keep=None):
    """
    原文链接：https://blog.csdn.net/a563562675/article/details/107066836
    return label of filter 
    """
    # 标记输入的3D图像
    label, num = measure.label(image, connectivity=1, background=0, return_num=True)
    if num < 1:
        return image

    # 获取对应的region对象
    region = measure.regionprops(label)
    # 获取每一块区域面积并排序
    num_list = [i for i in range(0, num)]
    area_list = [region[i].area for i in num_list]
    
    # 去除面积较小的连通域
    if ratio_keep:
        max_region_area = np.array(area_list).max()
        import pdb
        pdb.set_trace()
        drop_list = np.where(area_list<max_region_area*ratio_keep)[0]
        for i in drop_list:
            label[region[i].slice][region[i].image] = 0 
    
    else:
        if len(num_list) > num_keep_region:
            num_list_sorted = sorted(num_list, key=lambda x: area_list[x])[::-1]# 面积由大到小排序
            for i in num_list_sorted[num_keep_region:]:
                # label[label==i] = 0
                label[region[i].slice][region[i].image] = 0
#             num_list_sorted = num_list_sorted[:num_keep_region]
    import pdb
    pdb.set_trace()
    return label


def test_all_case(
    net, image_list, 
    num_classes, 
    name_classes,
    patch_size=(112, 112, 80), stride_xy=18, stride_z=4, 
    save_result=True, test_save_path=None, preproc_fn=None,
    device="cuda"
):
    if num_classes==2:
        cols = ['dice','jc','hd','asd']
    else:
        cols = [['dice']*len(name_classes)+['jc']*len(name_classes)+['hd']*len(name_classes)+['asd']*len(name_classes), name_classes*4]
    metrics = pd.DataFrame(columns=cols) 

    for image_path in tqdm(image_list):
        id = image_path.split('/')[-2]
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = np.argmax(h5f['label'][:],axis=-1)
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=num_classes, device="cuda")

        if np.sum(prediction)==0:
            single_metric = (0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:], num_classes)
        
        print(id,':')
        print("single_metric:",single_metric)

        metrics.loc[id] = np.array(single_metric).flatten().tolist()
        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), test_save_path + id + "_pred.nii.gz")
            nib.save(nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)), test_save_path + id + "_img.nii.gz")
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), test_save_path + id + "_gt.nii.gz")
    mean_metrics = metrics.mean()
    print('mean metric is:\n')
    print(mean_metrics)

    return metrics


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, device="cuda"):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                #test_patch = torch.from_numpy(test_patch).cuda()# gpu
                test_patch = torch.from_numpy(test_patch).to(device)# cpu
                y1 = net(test_patch)
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,:,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] += 1
    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = np.argmax(score_map, axis = 0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction==i)
        label_tmp = (label==i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice

def calculate_metric_percase(pred, gt, num_classes):
    "二分类、多分类的指标统计"
    if num_classes is None:
        num_classes = len(np.unique(gt))#注意：gt不是onehot编码
    print('np.unique(gt):',np.unique(gt))
    if num_classes==2:
        dice = metric.binary.dc(pred, gt)
        jc = metric.binary.jc(pred, gt)
        hd = metric.binary.hd95(pred, gt)
        asd = metric.binary.asd(pred, gt)
    elif num_classes>2:
        from keras.utils import to_categorical
        gt_onehot = to_categorical(gt, num_classes)
        pred_onehot = to_categorical(pred, num_classes)
        dice = []
        jc = []
        hd = []
        asd = []
        for k in range(num_classes):
            pred_k = pred_onehot[...,k]
            gt_k = gt_onehot[...,k]
            dice +=  [metric.dc(result=pred_k, reference=gt_k)]
            jc += [metric.jc(result=pred_k, reference=gt_k)]
            hd += [metric.hd95(result=pred_k, reference=gt_k)]
            asd += [metric.asd(result=pred_k, reference=gt_k)]
    else:
        raise ValueError("pred和gt不能是onehot编码")
    return dice, jc, hd, asd


