import torch
import sys
from cmath import inf
from torch.autograd import Variable
import numpy as np
import time
import torch.nn as nn
from tqdm import tqdm
import models.model as module_arch
#from tqdm import tqdm_notebook as tqdm
from typing import List
import sys
from base import BaseTrainer
from utils import inf_loop, get_logger, Timer, load_from_state_dict, set_seed
from collections import OrderedDict
import argparse
from parse_config import ConfigParser
import data_loader.data_loaders as module_data
import torch.nn.functional as F
import os
import pdb
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
        
def fgsm(gradz, step_size):
    return step_size*torch.sign(gradz)


def validate_pgd(test_loader, model, config,fc_features):
    
    # K = configs['pgd_attack']['K']
    # step = configs['pgd_attack']['step']
    # eps = configs['trainer']['adv_clip_eps']
    K = 10
    step = 2
    eps=8
    l2eps = eps*32*32*3/256

    # Attack amount
    color_value = 255.0
    step /= color_value
    eps /= color_value
    l2eps /= color_value
    print(f"PGD attack with each step {step} and a total of {K} steps, total eps {eps}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.linear.register_forward_pre_hook(fc_features)
    mu_c_dict={}
    before_class_dict={}
    pre_dict={}
    
    #################  center  #######################
    # pdb.set_trace()
    for batch_idx, (inputs, targets) in enumerate(test_loader):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        # pdb.set_trace()
        features = fc_features.outputs[0][0]
        features = F.normalize(features, dim=1)
        fc_features.clear()

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
                before_class_dict[y] = [features[b, :].detach().cpu().numpy()]
                pre_dict[y] = [outputs[b, :].detach().cpu().numpy()]

            else:
                mu_c_dict[y] += features[b, :]
                before_class_dict[y].append(features[b, :].detach().cpu().numpy())
                pre_dict[y].append(outputs[b, :].detach().cpu().numpy())

    
    return before_class_dict, pre_dict


def main(args, config: ConfigParser):

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=100,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build model architecture, then print to console
    model = getattr(module_arch, config["arch"]["type"])(
        num_classes = config["arch"]["args"]["num_classes"],
        norm_layer_type = config["arch"]["args"]["norm_layer_type"],
        conv_layer_type = config["arch"]["args"]["conv_layer_type"],
        linear_layer_type = config["arch"]["args"]["linear_layer_type"],
        activation_layer_type = config["arch"]["args"]["activation_layer_type"],
        etf_fc = config["arch"]["args"]["etf_fc"]
    ).to(device)

    checkpoint = torch.load(args.checkpoint_path)
    #model.load_state_dict(checkpoint["state_dict"])
    load_from_state_dict(model, checkpoint["state_dict"])
    model.eval()

    validate_pgd(test_data_loader, model, config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', type=str, default="./config_robust_train.json",
                        help='config file path (default: None)')
    parser.add_argument('-s', '--checkpoint_path', type=str, default="/data/xinshiduo/code/NC_good_or_bad-main/res_saved/models/Robust_experimentCIFAR10DataLoaderresnet18-num_classes-10-norm_layer_type-bn-conv_layer_type-conv-linear_layer_type-linear-activation_layer_type-relu-etf_fc-False-Seed=8/0525_000049/model_epoch_200.pth",
                        help='path to find model checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='0', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('--result_dir', type=str, default='saved', 
                        help='directory for saving results')
    parser.add_argument('--seed', type=int, default=6, 
                        help='Random seed')
    parser.add_argument('--name', type=str, default='', 
                        help='name for this model')

    config = ConfigParser.get_instance(parser)
    args = parser.parse_args()
    print("Start")
    set_seed(manualSeed = args.seed)
    main(args, config)