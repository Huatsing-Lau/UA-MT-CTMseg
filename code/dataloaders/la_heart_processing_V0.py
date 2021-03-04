import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import nrrd

output_size =[112, 112, 32]

def covert_h5():
    """
    备注：不要骨头，骨头合并到背景类别中
    """
    listt = glob('/home/cyagen/tyler/CTM/UA-MT/data/CTM_dataset/Segmented/*/CTM.nrrd')
    #listt = glob('../../data/2018LA_Seg_Training Set/*/CTM.nrrd')
    #listt = glob('../../data/2018LA_Seg_Training Set/*/lgemri.nrrd')
    for item in tqdm(listt):
        print(item.split('/')[-2],':')
        
        image, img_header = nrrd.read(item)
        label, label_header = nrrd.read(item.replace('CTM.nrrd', 'Segmentation-label.nrrd'))
        seg, seg_header = nrrd.read(item.replace('CTM.nrrd', 'Segmentation.seg.nrrd'))
        
        offset=[]
        for k in seg_header['Segmentation_ReferenceImageExtentOffset'].split():
            offset += [int(k)]
        sizes = seg_header['sizes'][1::]
        
        image = image[offset[0]:offset[0]+sizes[0],
                      offset[1]:offset[1]+sizes[1],
                      offset[2]:offset[2]+sizes[2]]
        label = label[offset[0]:offset[0]+sizes[0],
                      offset[1]:offset[1]+sizes[1],
                      offset[2]:offset[2]+sizes[2]].astype(np.uint8)
        
        # 类别名称和顺序
        target_name = ['dura','bone','SC']#目标类别顺序
        label_name = [
            seg_header['Segment0_Name'],
            seg_header['Segment1_Name'],
            seg_header['Segment2_Name'] 
            ]#人工标注的类别顺序
        ## 调整顺序，注意：seg是onehot编码
        idx = [label_name.index(name) for name in target_name]
        seg = seg[idx]
        ## 补充背景类别
        bg = np.ones(seg.shape[1:], dtype=np.uint8)-seg.sum(axis=0, dtype=np.uint8)
        bg = bg[np.newaxis,:]
        seg = np.concatenate((bg,seg),axis=0)
        ## 转化为非onehot编码
        label = np.argmax(seg, axis=0)
        
        # 合并骨头到背景中
        label[label==2] = 0
        label[label==3] = 2
        
        # 缩小图像
        image = image[0:-1:2,0:-1:2,:]
        label = label[0:-1:2,0:-1:2,:]

        
        print(label.shape)
        print(np.unique(label)) 

        tempL = np.nonzero(label)
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])
        
        w, h, d = label.shape
        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)
        image = (image - np.mean(image)) / np.std(image)
        image = image.astype(np.float32)
        image = image[minx:maxx, miny:maxy]
        label = label[minx:maxx, miny:maxy]
        print(label.shape)
        
        f = h5py.File(item.replace('CTM.nrrd', 'mri_norm2.h5'), 'w')
        f.create_dataset('image', data=image, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        f.close()
        

if __name__ == '__main__':
    covert_h5()
