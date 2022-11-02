from dis import dis
import torch
import sys
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
import os
import matplotlib.pyplot as plt
import pickle

os.environ['CUDA_VISIBLE_DEVICES']='7'
import setproctitle
setproctitle.setproctitle('NC@xinshiduo')

from validate_pgd_escape_l2 import validate_pgd_new

class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []

def main(args, config: ConfigParser):

    test_data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=100,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2
    ).split_validation()
    fc_features = FCFeatures()
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
    top1list=[]
    top5list=[]
    xlist=[]
    distlist=[]
    dist2targetlist=[]
    dist2targetchangelist=[]
    distchangelist=[]
    realL2list=[]
    attack_steplist=[]

    for step in [300]:
        checkpoint = torch.load(args.checkpoint_path+"model_epoch_"+str(step)+".pth")
        #model.load_state_dict(checkpoint["state_dict"])
        load_from_state_dict(model, checkpoint["state_dict"])
        model.eval()
        top1,top5,dist,distchange,dist2target,dist2targetchange,realL2,attack_step=validate_pgd_new(test_data_loader, model, config,fc_features)
        top1list.append(top1)
        top5list.append(top5)
        xlist.append(step)
        distlist.append(dist)
        distchangelist.append(distchange)
        dist2targetlist.append(dist2target)
        dist2targetchangelist.append(dist2targetchange)
        realL2list.append(realL2)
        attack_steplist.append(attack_step)

    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    plt.plot(xlist,top1list, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7 ,label='top1')
    plt.plot(xlist,top5list, 'r', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7 ,label='top5')
    plt.legend()
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('robust_acc', fontsize=40)
    fig.savefig(str(args.checkpoint_path + "roubust-nearest-escl2.pdf"), bbox_inches='tight')
    with open((args.checkpoint_path + 'info-roubust-nearest-escl2.pkl'), 'wb') as f: 
        pickle.dump({'xlist':xlist,'acc':top1list,'acc5':top5list,'dist2center':distlist,'distchange':distchangelist,'dist2target':dist2targetlist,'dist2targetchange':dist2targetchangelist,'realL2':realL2list,'attackstep':attack_steplist}, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', type=str, default="./config_cifar10_standard.json",
                        help='config file path (default: None)')
    parser.add_argument('-s', '--checkpoint_path', type=str, default="/data1/xinshiduo/code-new/natural350-4cls-pgd-tanh/res_saved/models/ExperimentCIFAR10DataLoaderresnet18-num_classes-4-norm_layer_type-bn-conv_layer_type-conv-linear_layer_type-linear-activation_layer_type-relu-etf_fc-False-Seed=12/1012_201932/",
                        help='path to find model checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=0, type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epoch', default=61, type=int,
                        help='---')
    parser.add_argument('-r', '--resume', default=None, type=str,
    help='path to latest checkpoint (default: None)')
    parser.add_argument('--cycle', default=5, type=int,
                        help='---')
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