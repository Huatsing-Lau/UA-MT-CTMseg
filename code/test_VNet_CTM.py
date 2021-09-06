# +
import os
import argparse
import torch
from networks.vnet import VNet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/CTM_dataset/Segmented', help='Folder of Test Set')
parser.add_argument('--model', type=str,  default='VNet_CTM', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

name_classes = ['bg','dura','SC']
num_classes = len(name_classes)


# with open(FLAGS.root_path + '/../test.list', 'r') as f:
with open('../data/CTM_dataset/test.list', 'r') as f:
    image_list = f.readlines()
image_list = [os.path.join(FLAGS.root_path,item.replace('\n', ''),"preprocessed_CTM.h5") for item in image_list]

def test_calculate_metric(
    model_path, patch_size=(128, 128, 64), 
    stride_xy=64, stride_z=32, 
    device='cuda'):
    
    #net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).to(device)
    #net.load_state_dict(torch.load(model_path))
    net = torch.load(model_path)
    net.eval()

    metrics = test_all_case(
        net, image_list, 
        num_classes=num_classes,
        name_classes=name_classes,
        patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z,
        save_result=True, test_save_path=test_save_path)

    return metrics

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(snapshot_path, 'best_model.pth')# 'final.pth'
    metrics = test_calculate_metric(model_path, patch_size=(128, 128, 64), stride_xy=64, stride_z=32, device=device)
    metrics.to_csv(os.path.join(test_save_path,'metrics_test_set.csv'))
