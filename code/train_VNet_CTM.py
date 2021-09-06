# -*- coding: utf-8 -*-
# +
# python train_VNet_CTM.py --gpu '1,2,3,4'

# +
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import cv2

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
# from utils.losses import dice_loss
from utils import ramps, losses
from utils.util import EarlyStopping
from dataloaders.CTMSpine_sitk import CTMSpine, RandomScale, RandomNoise, RandomCrop, CenterCrop, RandomRot, RandomFlip, ToTensor, TwoStreamBatchSampler
# -

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CTM_dataset/Segmented', help='data path')
parser.add_argument('--exp', type=str,  default='VNet_CTM', help='Name of Experiment, ie. model_name')
# parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum iteration number to train')
parser.add_argument('--max_epoch', type=int,  default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--patience', type=int, default=40, help='maximum epoch number to keep patient')
parser.add_argument('--patience_lr', type=int, default=20, help='maximum epoch number to keep patient for learning reduce')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='7', help='GPU to use')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"
best_filepath = os.path.join(snapshot_path, 'best_model.pth')

# +
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
    
# batch_size = args.batch_size * len(args.gpu.split(','))
batch_size = args.batch_size

max_epoch = args.max_epoch
base_lr = args.base_lr
patience = args.patience
patience_lr = args.patience_lr
# -

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 3
patch_size = (128, 128, 64)
cls_weights = [1,4,10]

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
    net = net.cuda()

    db_train = CTMSpine(
        base_dir=train_data_path,
        split='train',
        filename='preprocessed_CTM.h5',
        num=150,#150,#100,#50,
        transform = transforms.Compose([
            RandomScale(ratio_low=0.9, ratio_high=1.1),
            RandomRot(min_angle=0,max_angle=360),
            RandomFlip(),
            RandomCrop(patch_size),
            RandomNoise(mu=0, sigma=0.05),
            ToTensor(),
        ]))
    db_valid = CTMSpine(
        base_dir=train_data_path,
        split='test',
        filename='preprocessed_CTM.h5',
        transform = transforms.Compose([
            CenterCrop(patch_size),
            ToTensor()
        ]))
    
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
        
    trainloader = DataLoader(
        db_train, 
        batch_size=batch_size, 
        shuffle=True,  
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )
    
    validloader = DataLoader(
        db_valid, 
        batch_size=batch_size, 
        shuffle=True,  
        num_workers=4, 
        pin_memory=True, 
        worker_init_fn=worker_init_fn
    )

    net.train()
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    lr_ = base_lr
    net.train()
    early_stopping = EarlyStopping(patience, verbose=True, best_filepath=best_filepath)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_lr, factor=0.1)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            if epoch_num==0 and i_batch==0:
                print("batch size: ",volume_batch.shape[0])
                
            outputs = net(volume_batch)
            outputs_soft = F.softmax(outputs, dim=1)

            loss_seg = F.cross_entropy( outputs, label_batch, weight=torch.tensor(cls_weights,dtype=torch.float32).cuda() )
            loss_seg_dice = 0
            for i in range(num_classes):
                loss_mid = losses.dice_loss(outputs_soft[:, i, :, :, :], label_batch == i )
                loss_seg_dice += loss_mid
                print('dice score (1-dice_loss): {:.3f}'.format(1-loss_mid))
            print('dicetotal:{:.3f}'.format(loss_seg_dice))
            loss = 0.5*(loss_seg+loss_seg_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            logging.info('epoch %d, iteration %d : loss : %f, loss_seg : %f, loss_seg_dice : %f' % 
                         (epoch_num,
                          iter_num, 
                          loss.item(),
                          loss_seg.item(),
                          loss_seg_dice.item())
                        )
        
        # validation
        net.eval()
        with torch.no_grad():
            loss_seg_dice_valid = 0
            for i_batch, sampled_batch in enumerate(validloader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                outputs = net(volume_batch)
                outputs_soft = F.softmax(outputs, dim=1)
                loss_seg_dice = 0
                for i in range(num_classes):
                    loss_mid = losses.dice_loss(outputs_soft[:, i, :, :, :], label_batch == i )
                    loss_seg_dice += loss_mid
                loss_seg_dice_valid += loss_seg_dice.cpu().numpy()
            loss_seg_dice_valid = loss_seg_dice_valid/(i_batch+1)
            logging.info( 'Epoch %d : sum of dice_loss of validation-set = %f' % (epoch_num, loss_seg_dice_valid) )       
        net.train()
        
        ## save grid image
        gt_label = (255.0/(num_classes-1.0))*label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        proba_label = (255.0/(num_classes-1.0))*torch.argmax(outputs_soft[0, :, :, :, 20:61:10],axis=0).unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        image = volume_batch[0, :, :, :, 20:61:10].permute(3,0,1,2).repeat(1,3,1,1)
        image = (image-image.min())*255.0/(image.max()-image.min())
        image_label = torch.cat( (image,gt_label,proba_label), axis=0 )
        grid_image = make_grid(image_label,nrow=image.shape[0], normalize=False).int()
#         writer.add_image('train/Image_Groundtruth_Predicted_label', grid_image, iter_num)
        filepath = os.path.join(snapshot_path,"grid_image","epoch_%d.png"%epoch_num)
        if not os.path.isdir(os.path.split(filepath)[0]):
            os.mkdir(os.path.split(filepath)[0])
        cv2.imwrite(filepath, grid_image.cpu().numpy().transpose((1,2,0)))
        
        ## save model
        save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
        torch.save(net.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
                
        ## change lr
        scheduler.step(loss_seg_dice_valid) # 根据度量指标调整学习率
        print("learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])
        
        time2 = time.time()
        logging.info( 'epoch %d finished, time cost: %.3f s'%(epoch_num,time2-time1) )
            
        ## early stop
        early_stopping(loss_seg_dice_valid, net)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            logging.info("Early stopping")
            # 结束模型训练
            break
        
    save_mode_path = os.path.join(snapshot_path, 'final.pth')
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
