# -*- coding: utf-8 -*-
# +
# python train_UAMT_CTM_V3 --gpu "0,1,2,3"
# -

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

from vnet import VNet
# from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from utils.util import EarlyStopping
from dataloaders.CTMSpine_sitk import TwoStreamBatchSampler, CTMSpine_semi, CTMSpine, RandomScale, RandomNoise, RandomCrop, CenterCrop, RandomRot, RandomFlip, ToTensor, TransformConsistantOperator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path_labeled', type=str, default='../data/CTM_dataset/Segmented')
parser.add_argument('--root_path_unlabeled', type=str, default='../model/prediction/unSegmented_center_cut/VNet_CTM_post')#VNet_Binary_CTM_post 
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='Name of Experiment, ie. model_name')
parser.add_argument('--max_epoch', type=int,  default=1000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--patience', type=int, default=20, help='maximum epoch number to keep patient')
parser.add_argument('--patience_lr', type=int, default=10, help='maximum epoch number to keep patient for learning reduce')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=10.0, help='consistency')# default=1.0
parser.add_argument('--consistency_rampup', type=float,  default=1.0, help='consistency_rampup')
args = parser.parse_args()

labeled_train_data_path = args.root_path_labeled
unlabeled_train_data_path = args.root_path_unlabeled
snapshot_path = "../model/" + args.exp + "/"
best_filepath = os.path.join(snapshot_path, 'best_model.pth')


# +
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
    
# batch_size = args.batch_size * len(args.gpu.split(','))
# labeled_bs = args.labeled_bs * len(args.gpu.split(','))
batch_size = args.batch_size
labeled_bs = args.labeled_bs

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
patch_size = (128, 128, 96)#(128, 128, 64)
cls_weights = [1,4,10]

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1-alpha)

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

    def create_model(ema=False):
        # Network definition
#         net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
#         model = net.cuda()
        
        model_path = os.path.join(snapshot_path, './best_model_U3.pth')
        model = torch.load(model_path).cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    
    ema_model = create_model(ema=True)
#     if torch.cuda.device_count() > 1:
#         model = torch.nn.DataParallel(model)
#         ema_model = torch.nn.DataParallel(ema_model)
    
    #　pytorch 的数据加载到模型的操作顺序（三板斧）：
    #    ① 创建一个 Dataset 对象
    #    ② 创建一个 DataLoader 对象
    #    ③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
    db_train = CTMSpine_semi(
        labeled_base_dir=labeled_train_data_path,
        unlabeled_base_dir=unlabeled_train_data_path,
        split='train',
        labeled_filename='preprocessed_CTM.h5',
        unlabeled_filename='center_cut.h5',
        num=150,#50#100#150
        transform = transforms.Compose([
            RandomScale(ratio_low=0.9, ratio_high=1.1),
            RandomNoise(mu=0, sigma=0.05),
            RandomRot(),
            RandomFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ]))
    
    db_valid = CTMSpine(
        base_dir=labeled_train_data_path,
        split='test',
        filename='preprocessed_CTM.h5',
        transform = transforms.Compose([
            CenterCrop(patch_size),
            ToTensor()
        ]))
    print("db_train.__len_labeled__():", db_train.__len_labeled__())
    print("db_train.__len_unlabeled__():",db_train.__len_unlabeled__())
    print("db_valid.__len__():",db_valid.__len__())
    
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    # 在linux系统中可以使用多个子进程加载数据，而在windows系统中不能。所以在windows中要将DataLoader中的num_workers设置为0或者采用默认为0的设置。
    labeled_idxs = list( range(0, db_train.__len_labeled__()) )
    unlabeled_idxs = list( range(db_train.__len_labeled__(), db_train.__len_labeled__()+db_train.__len_unlabeled__()) )
#     batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
#     trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, batch_size, batch_size-labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    
    validloader = DataLoader(db_valid, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    
    
    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')

    iter_num = 0
    print("max_epoch:",max_epoch)
    lr_ = base_lr
    early_stopping = EarlyStopping(patience, verbose=True, best_filepath=best_filepath)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_lr, factor=0.1)
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()          
                
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
#             volume_batch, label_batch = volume_batch.cuda(), label_batch[:labeled_bs].cuda()
            volume_batch, label_batch = volume_batch.cuda(), label_batch[-labeled_bs:].cuda()
            
            if epoch_num==0 and i_batch==0:
                print("batch size: ",volume_batch.shape[0])

            ema_inputs = volume_batch
            
            outputs = model(volume_batch)
#             labeled_outputs = outputs[:labeled_bs]
            labeled_outputs = outputs[-labeled_bs:]
            labeled_outputs_soft = F.softmax(labeled_outputs, dim=1)
            
            with torch.no_grad():
                ema_outputs = ema_model(ema_inputs)
            T = 8
            volume_batch_r = volume_batch.repeat(1, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 1
            preds = torch.zeros([stride * T, num_classes, patch_size[0], patch_size[1], patch_size[2]]).cuda()
            for i in range(T//1):
                TCO = TransformConsistantOperator(k=i, axis=np.random.randint(0, 2))
                noise = torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                ema_inputs = volume_batch_r + noise
                with torch.no_grad():
                    ema_inputs = TCO.transform(ema_inputs).cuda()
                    pred = ema_model(ema_inputs)# note: forward with random dropout
                    pred = TCO.inv_transform(pred).cuda()
                    preds[1 * stride * i:1 * stride * (i + 1)] = pred
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, num_classes, patch_size[0], patch_size[1], patch_size[2])
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            uncertainty = -1.0*torch.sum(preds[:,1:,:,:,:]*torch.log(preds[:,1:,:,:,:] + 1e-6), dim=1, keepdim=True)

            
            ## calculate the loss ( only for labeled samples )
            loss_seg = F.cross_entropy( labeled_outputs, label_batch, weight=torch.tensor(cls_weights,dtype=torch.float32).cuda() )
            loss_seg_dice = 0
            for i in range(num_classes):
                loss_mid = losses.dice_loss(labeled_outputs_soft[:, i, :, :, :], label_batch == i )
                loss_seg_dice += loss_mid
                print('dice score (1-dice_loss): {:.3f}'.format(1-loss_mid))
            supervised_loss = (loss_seg+loss_seg_dice)/2.0
#             supervised_loss = loss_seg_dice
#             if epoch_num==20 and i_batch==5:
#                 import pdb
#                 pdb.set_trace()
        
            ## calculate the loss ( only for unlabeled samples )
            consistency_weight = args.consistency# get_current_consistency_weight(epoch_num/20)
            consistency_dist = consistency_criterion(outputs[:,1:,:,:,:], ema_outputs[:,1:,:,:,:]) #(batch, num_classes, 112,112,80)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(epoch_num, 20))*np.log(num_classes)#N分类问题的最大不确定度是log(N)
            mask = (uncertainty<threshold).int()
            consistency_dist = torch.sum(mask*consistency_dist)/(torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * consistency_dist
            loss = supervised_loss + consistency_loss
            
            
            # pytorch模型训练的三板斧
            # 一般训练神经网络，总是逃不开optimizer.zero_grad之后是loss（后面有的时候还会写forward，看你网络怎么写了）之后是是net.backward之后是optimizer.step的这个过程
            optimizer.zero_grad()#把模型中参数的梯度设为0
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, epoch_num)

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('Epoch %d, iteration %d : loss : %f, loss_seg : %f, loss_seg_dice : %f, consistency_loss : %f, cons_dist: %f, consistency_weight: %f' %
                         (epoch_num,
                          iter_num, 
                          loss.item(), 
                          loss_seg.item(),
                          loss_seg_dice.item(),
                          consistency_loss.item(),
                          consistency_dist.item(),
                          consistency_weight))

        
        # validation
        model.eval()
        with torch.no_grad():
            loss_seg_dice_valid = 0
            for i_batch, sampled_batch in enumerate(validloader):
                volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                outputs = model(volume_batch)
                outputs_soft = F.softmax(outputs, dim=1)
                loss_seg_dice = 0
                for i in range(num_classes):
                    loss_mid = losses.dice_loss(outputs_soft[:, i, :, :, :], label_batch == i )
                    loss_seg_dice += loss_mid
                loss_seg_dice_valid += loss_seg_dice.cpu().numpy()
            loss_seg_dice_valid = loss_seg_dice_valid/(i_batch+1)
            logging.info( 'Epoch %d : sum of dice_loss of validation-set = %f' % (epoch_num, loss_seg_dice_valid) )       
        model.train()
        
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
        torch.save(model, save_mode_path)# torch.save(model.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
                
        ## change lr
        scheduler.step(loss_seg_dice_valid) # 根据度量指标调整学习率
        print("learning rate: ", optimizer.state_dict()['param_groups'][0]['lr'])
        
        time2 = time.time()
        logging.info( 'epoch %d finished, time cost: %.3f s'%(epoch_num,time2-time1) )
            
        ## early stop
        early_stopping(loss_seg_dice_valid, model)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
            logging.info("Early stopping")
            # 结束模型训练
            break
            
    save_mode_path = os.path.join(snapshot_path, 'final.pth')
    torch.save(model, save_mode_path)# torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
