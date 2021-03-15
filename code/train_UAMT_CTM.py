# -*- coding: utf-8 -*-
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
from dataloaders.CTMSpine_sitk import CTMSpine, CTMSpine_unseg, RandomScale, RandomNoise, RandomCrop, CenterCrop, RandomRot, RandomFlip, ToTensor, TransformConsistantOperator

parser = argparse.ArgumentParser()
parser.add_argument('--root_path_labeled', type=str, default='../data/CTM_dataset/Segmented')
parser.add_argument('--root_path_unlabeled', type=str, default='../data/CTM_dataset/unSegmented')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='2', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

labeled_train_data_path = args.root_path_labeled
unlabeled_train_data_path = args.root_path_unlabeled
snapshot_path = "../model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
torch.cuda.empty_cache()
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs * len(args.gpu.split(','))

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
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# +
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
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)
    
    #　pytorch 的数据加载到模型的操作顺序（三板斧）：
    #    ① 创建一个 Dataset 对象
    #    ② 创建一个 DataLoader 对象
    #    ③ 循环这个 DataLoader 对象，将img, label加载到模型中进行训练
    db_train_labeled = CTMSpine(
        base_dir=labeled_train_data_path,
        split='train',
        transform = transforms.Compose([
            RandomScale(ratio_low=0.8, ratio_high=1.2),
            RandomNoise(mu=0, sigma=0.05),
            RandomRot(),
            RandomFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ]))
    db_train_unlabeled = CTMSpine_unseg(
        base_dir=unlabeled_train_data_path,
        transform = transforms.Compose([
            RandomScale(ratio_low=0.8, ratio_high=1.2),
            RandomNoise(mu=0, sigma=0.05),
            RandomRot(),
            RandomFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ]))#因为计算一致性损失时增加了噪声，所以不在此处加噪声
#     db_test = LAHeart(base_dir=labeled_train_data_path,
#                        split='test',
#                        transform = transforms.Compose([
#                            CenterCrop(patch_size),
#                            ToTensor()
#                        ]))
    

    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    # 在linux系统中可以使用多个子进程加载数据，而在windows系统中不能。所以在windows中要将DataLoader中的num_workers设置为0或者采用默认为0的设置。
    #trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    #trainloader = DataLoader(db_train_labeled, batch_sampler=batch_sampler, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    
    labeled_trainloader = DataLoader(db_train_labeled, batch_size=labeled_bs, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)
    unlabeled_trainloader = DataLoader(db_train_unlabeled, batch_size=batch_size-labeled_bs, shuffle=True, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

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
    logging.info("{} itertations per epoch".format(len(unlabeled_trainloader)))

    iter_num = 0
    max_epoch = max_iterations//len(labeled_trainloader)+1
    print("max_epoch:",max_epoch)
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, (sampled_batch_labeled, sampled_batch_unlabeled) in enumerate( zip(labeled_trainloader, unlabeled_trainloader) ):
            time2 = time.time()
            # print('fetch data cost {}'.format(time2-time1))
            labeled_volume_batch, label_batch = sampled_batch_labeled['image'], sampled_batch_labeled['label']
            unlabeled_volume_batch = sampled_batch_unlabeled['image']
            unlabeled_volume_batch = torch.cat((labeled_volume_batch,unlabeled_volume_batch),dim=0)
            labeled_volume_batch, label_batch = labeled_volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = unlabeled_volume_batch.cuda()

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.05, -0.05, 0.05)
            ema_inputs = unlabeled_volume_batch + noise
            
            outputs = model(labeled_volume_batch)
            unlabeled_outputs = model(unlabeled_volume_batch)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, num_classes, patch_size[0], patch_size[1], patch_size[2]]).cuda()
            for i in range(T//2):
                TCO = TransformConsistantOperator(k=i, axis=np.random.randint(0, 2))
                ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    ema_inputs = TCO.transform(ema_inputs).cuda()
                    pred = ema_model(ema_inputs)
                    pred = TCO.inv_transform(pred).cuda()
                    preds[2 * stride * i:2 * stride * (i + 1)] = pred
            preds = F.softmax(preds, dim=1)
            #preds = preds.reshape(T, stride, 2, 112, 112, 80)
            preds = preds.reshape(T, stride, num_classes, patch_size[0], patch_size[1], patch_size[2])
            preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True)

            ## calculate the loss(only for labeled samples)
            loss_seg = F.cross_entropy( outputs, label_batch, weight=torch.tensor(cls_weights,dtype=torch.float32).cuda() )
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = 0
            print('\n')
            for i in range(num_classes):
                loss_mid = losses.dice_loss(outputs_soft[:, i, :, :, :], label_batch == i )
                loss_seg_dice += loss_mid
                print('dice score (1-dice_loss): {:.3f}'.format(1-loss_mid))

#             print('dicetotal:{:.3f}'.format( loss_seg_dice))
            #loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)
            
            # only for unlabeled samples
            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(unlabeled_outputs, ema_output) #(batch, num_classes, 112,112,80)
            threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.sqrt(3)#N分类问题的最大不确定度是sqrt(N)
            mask = (uncertainty<threshold).float()
#             print("consistency_dist:",consistency_dist.item())
            asd = np.prod( list(mask.shape) )
            #print("mask:",np.sum(mask.item())/asd )
            consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_loss = consistency_weight * consistency_dist
            loss = supervised_loss + consistency_loss
            
            
            # pytorch模型训练的三板斧
            # 一般训练神经网络，总是逃不开optimizer.zero_grad之后是loss（后面有的时候还会写forward，看你网络怎么写了）之后是是net.backward之后是optimizer.step的这个过程
            optimizer.zero_grad()#把模型中参数的梯度设为0
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

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

            logging.info('iteration %d : loss : %f, loss_seg : %f, loss_seg_dice : %f, consistency_loss : %f, cons_dist: %f, loss_weight: %f' %
                         (iter_num, 
                          loss.item(), 
                          loss_seg.item(),
                          loss_seg_dice.item(),
                          consistency_loss.item(),
                          consistency_dist.item(),
                          consistency_weight))
            if iter_num % 50 == 0:
                image = labeled_volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)#repeat 3是为了模拟图像的RGB三个通道
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/uncertainty', grid_image, iter_num)

                mask2 = (uncertainty > threshold).float()
                image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/mask', grid_image, iter_num)
                #####
                image = labeled_volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
