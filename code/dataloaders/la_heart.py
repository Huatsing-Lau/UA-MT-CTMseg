import os
import torch
import numpy as np
import random
# from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
# import cv2
from skimage import transform

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split=='train':
            with open(self._base_dir+'/../train.list', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'test':
            with open(self._base_dir+'/../test.list', 'r') as f:
                self.image_list = f.readlines()
        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #print(self.image_list)
        #print("check: ",idx,len(self.image_list))
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        #print("np.unique(label):",np.unique(label))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class LAHeart_unseg(Dataset):
    """ LA Dataset """
    def __init__(self, base_dir=None, num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        with open(self._base_dir+'/../train_unseg.list', 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n','') for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        #print(self.image_list)
        #print("check: ",idx,len(self.image_list))
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir+"/"+image_name+"/mri_norm2.h5", 'r')
        image = h5f['image'][:]
        sample = {'image': image,'label':None}
        if self.transform:
            sample = self.transform(sample)

        return sample

class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if label is not None:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if label:
            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]

        return {'image': image, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if image.shape[0] <= self.output_size[0] or image.shape[1] <= self.output_size[1] or image.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - image.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - image.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - image.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if label is not None:
                label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if label is not None:
            label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}

# import SimpleITK as sitk
# def resample_image3D(
#     image3D,
#     spacing=[0.3,0.3,3],
#     ratio=1.0,
#     method='Linear',):
#     """做插值"""
#     resample = sitk.ResampleImageFilter()
#     if method is 'Linear':
#         resample.SetInterpolator(sitk.sitkLinear)
#     elif method is 'Nearest':
#         resample.SetInterpolator(sitk.sitkNearestNeighbor)
#     resample.SetOutputDirection( (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) )
#     resample.SetOutputOrigin((0,0,0))
#     newspacing = (np.array(spacing)*ratio).tolist()
#     resample.SetOutputSpacing(newspacing)
    
#     newsize = np.round(np.array(image3D.shape)*ratio).astype('int').tolist() 
#     resample.SetSize(newsize)
#     # resample.SetDefaultPixelValue(0)
#     print("image3D.shape:",image3D.shape)
#     image3D = sitk.GetImageFromArray(image3D)
#     image3D.SetSpacing(spacing)
#     image3D.SetDirection( (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) )
#     image3D.SetOrigin((0,0,0))
    
#     newimage = resample.Execute(image3D)
#     newimage = sitk.GetArrayFromImage(newimage)
#     print("newimage.shape:",newimage.shape)
#     return newimage

class RandomScale(object):
    """
    Scale randomly the image within the scaling ratio of 0.8-1.2
    Args:
    ratio_low, ratio_high (float): Desired ratio range of random scale 
    """

    def __init__(self, ratio_low, ratio_high):
        self.ratio_low = ratio_low
        self.ratio_high = ratio_high

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        # rescale
        ratio = np.random.uniform(self.ratio_low, self.ratio_high)
        image = transform.rescale(image,ratio,order=1,anti_aliasing=True,preserve_range=True,multichannel=False) 
        
        assert np.unique(label).tolist() == [0,1,2], "np.unique(label):"+str(np.unique(label).tolist())
        if label is not None:
            label = transform.rescale(label,ratio,order=0,anti_aliasing=True,preserve_range=True,multichannel=False)
            label = np.round(label)

#         assert np.unique(label).tolist() == [0,1,2], "np.unique(rescaled label):"+str(np.unique(label).tolist())
#         print("image.shape",image.shape,
#               "label.shape",label.shape,
#               "ratio,dsize:",ratio,dsize,
#               "np.unique(label):",np.unique(label),
#              )
        return {'image': image, 'label': label}
    
class TransformConsistantOperator():
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """
    def __init__(self, k=None, axis=None):
        if k is not None:
            self.k = k
        else:
            self.k = np.random.randint(0, 4)
        if axis is not None:
            self.axis = axis
        else:
            self.axis = np.random.randint(0, 2)
            
    def transform(self, image):
        """image could be image or mask"""
        image = image.permute(2,3,4,0,1)
        image = torch.rot90(image, self.k)#np.rot90(image, self.k)
        image = torch.flip(image, dims=[self.axis])#np.flip(image, axis=self.axis).copy()
        image = image.permute(3,4,0,1,2)

#         image = image.permute(2,3,4,0,1).cpu()
#         image = np.rot90(image, self.k)
#         image = np.flip(image, axis=self.axis).copy()
#         image = torch.from_numpy( image.transpose((3,4,0,1,2)).copy() )
        return image
    
    def inv_transform(self, image):
        """image could be image or mask"""
        image = image.permute(2,3,4,0,1)
        image = torch.flip(image, dims=[self.axis])
        image = torch.rot90(image, -self.k)
        image = image.permute(3,4,0,1,2)
        
#         image = image.permute(2,3,4,0,1).cpu()
#         image = np.flip(image, axis=self.axis).copy()
#         image = np.rot90(image, -self.k)
#         image = torch.from_numpy( image.transpose((3,4,0,1,2)).copy() )
        
        return image
    
class RandomRot(object):
    """
    Randomly rotate the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        if label is not None:
            label = np.rot90(label, k)

        return {'image': image, 'label': label}

class RandomFlip(object):
    """
    Randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        flip = random.sample([True,False], 1)
        if flip:
            axis = np.random.randint(0, 2)
            image = np.flip(image, axis=axis).copy()
        if label is not None:
            if flip:
                label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        label = sample['label']
        
        if label is not None:
            if 'onehot_label' in sample:
                return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long(),
                        'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
            else:
                return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}
        else:
            if 'onehot_label' in sample:
                return {'image': torch.from_numpy(image),
                        'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
            else:
                return {'image': torch.from_numpy(image)}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
