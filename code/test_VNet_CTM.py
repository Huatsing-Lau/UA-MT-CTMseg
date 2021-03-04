import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/home/cyagen/tyler/CTM/UA-MT/data/CTM_dataset/Segmented', help='Folder of Test Set')
parser.add_argument('--model', type=str,  default='vnet_supervisedonly_dp', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 3

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [os.path.join(FLAGS.root_path,item.replace('\n', ''),"mri_norm2.h5") for item in image_list]

def test_calculate_metric(epoch_num, patch_size=(128, 128, 64), stride_xy=64, stride_z=32):
    net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False)#.cuda()
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric,metrics = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
                               save_result=True, test_save_path=test_save_path)

    return avg_metric, metrics


if __name__ == '__main__':
    avg_metric, metrics = test_calculate_metric(6001, patch_size=(128, 128, 64), stride_xy=64, stride_z=32)
    print(avg_metric)
    metrics.to_csv(os.path.join(test_save_path,'metrics_test_set.csv'))