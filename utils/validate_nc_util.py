import sys
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import scipy.linalg as scilin

import copy
import os
import numpy as np
from utils import load_from_state_dict
import matplotlib.pyplot as plt

MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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

        
class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def compute_info(device, model, fc_features, dataloader, isTrain=True):
    mu_G = 0
    mu_c_dict = dict()
    before_class_dict = dict()
    after_class_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        features = fc_features.outputs[0][0]
        fc_features.clear()

        mu_G += torch.sum(features, dim=0)

        for b in range(len(targets)):
            y = targets[b].item()
            if y not in mu_c_dict:
                mu_c_dict[y] = features[b, :]
                before_class_dict[y] = [features[b, :].detach().cpu().numpy()]
                after_class_dict[y] = [outputs[b, :].detach().cpu().numpy()]
            else:
                mu_c_dict[y] += features[b, :]
                before_class_dict[y].append(features[b, :].detach().cpu().numpy())
                after_class_dict[y].append(outputs[b, :].detach().cpu().numpy())

        prec1, prec5 = compute_accuracy(outputs.data, targets.data, topk=(1, 5))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    if isTrain:
        mu_G /= sum(CIFAR10_TRAIN_SAMPLES)
        for i in range(len(CIFAR10_TRAIN_SAMPLES)):
            mu_c_dict[i] /= CIFAR10_TRAIN_SAMPLES[i]
    else:
        mu_G /= sum(CIFAR10_TEST_SAMPLES)
        for i in range(len(CIFAR10_TEST_SAMPLES)):
            mu_c_dict[i] /= CIFAR10_TEST_SAMPLES[i]

    return mu_G, mu_c_dict, before_class_dict, after_class_dict, top1.avg, top5.avg


def compute_Sigma_W(device, model, fc_features, mu_c_dict, dataloader, isTrain=True):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            
        features = fc_features.outputs[0][0]
        fc_features.clear()

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if isTrain:
        Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
    else:
        Sigma_W /= sum(CIFAR10_TEST_SAMPLES)

    return Sigma_W.detach().cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.detach().cpu().numpy()


def compute_ETF(W):
    K = W.shape[0]
    WWT = torch.mm(W, W.T)
    WWT /= torch.norm(WWT, p='fro')

    sub = (torch.eye(K) - 1 / K * torch.ones((K, K))).cuda() / pow(K - 1, 0.5)
    ETF_metric = torch.norm(WWT - sub, p='fro')
    return ETF_metric.detach().cpu().numpy().item()


def compute_W_H_relation(W, mu_c_dict, mu_G):
    K = len(mu_c_dict)
    M = torch.empty(mu_c_dict[0].shape[0], K)
    for i in range(K):
        M[:, i] = mu_c_dict[i] - mu_G
    sub = 1 / np.sqrt(K-1) * (torch.eye(K) - torch.ones(K, K) / K)

    WH = W.cpu() @ M ####Uncomment this if delete above
    
    res = torch.norm(WH / torch.norm(WH, p='fro') - sub, p='fro')

    return res.detach().cpu().numpy()


def validate_nc_epoch(checkpoint_dir, epoch, orig_model, trainloader, testloader, info_dict):
    print(f"Processing the NC information for epoch {epoch}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(orig_model)
    
    fc_features = FCFeatures()
    model.linear.register_forward_pre_hook(fc_features)

#     model.load_state_dict(torch.load(str(checkpoint_dir / f'model_epoch_{epoch}.pth'), map_location=device)["state_dict"])
    model.eval()
    
    have_bias = False
    for n, p in model.named_parameters():
        if 'linear.weight' in n:
            W = p.clone()
        if 'linear.bias' in n:
            b = p.clone()
            have_bias = True

    mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, train_acc1, train_acc5 = compute_info(device, model, fc_features, trainloader, isTrain=True)
    mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, test_acc1, test_acc5 = compute_info(device, model, fc_features, testloader, isTrain=False)
    
    Sigma_W = compute_Sigma_W(device, model, fc_features, mu_c_dict_train, trainloader, isTrain=True)
    
    Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

    collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
    ETF_metric = compute_ETF(W)
    WH_relation_metric = compute_W_H_relation(W, mu_c_dict_train, mu_G_train) # Added back

    info_dict['collapse_metric'].append(collapse_metric)
    info_dict['ETF_metric'].append(ETF_metric)
    info_dict['WH_relation_metric'].append(WH_relation_metric) # Added back
    info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
    info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())
    for key in mu_c_dict_train:
        mu_c_dict_train[key] = mu_c_dict_train[key].detach().cpu().numpy()
    for key in mu_c_dict_test:
        mu_c_dict_test[key] = mu_c_dict_test[key].detach().cpu().numpy()
    info_dict['mu_c_dict_train'] = mu_c_dict_train
    info_dict['mu_c_dict_test'] = mu_c_dict_test
    info_dict['before_class_dict_train'] = before_class_dict_train
    info_dict['after_class_dict_train'] = after_class_dict_train
    info_dict['before_class_dict_test'] = before_class_dict_test
    info_dict['after_class_dict_test'] = after_class_dict_test
    info_dict['W'].append((W.detach().cpu().numpy()))
    if have_bias:
        info_dict['b'].append(b.detach().cpu().numpy())

    info_dict['train_acc1'].append(train_acc1)
    info_dict['train_acc5'].append(train_acc5)
    info_dict['test_acc1'].append(test_acc1)
    info_dict['test_acc5'].append(test_acc5)

    print(f"Epoch {epoch} is processed")
        

def plot_nc(info_dict, epochs):
    XTICKS = [30 * i for i in range(8) if i < epochs / 30]
    
    fig_collapse = plot_collapse(info_dict, epochs, XTICKS)
    fig_etf = plot_ETF(info_dict, epochs, XTICKS)
    fig_wh = plot_WH_relation(info_dict, epochs, XTICKS)
    fig_train_acc = plot_train_acc(info_dict, epochs, XTICKS)
    fig_test_acc = plot_test_acc(info_dict, epochs, XTICKS)
    
    return fig_collapse, fig_etf, fig_wh, fig_train_acc, fig_test_acc
    

############ Below are support methods for plot_nc ############
###############################################################
def plot_collapse(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    plt.plot(info['collapse_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_1$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 2, 0.1), fontsize=30) 

    plt.axis([0, epochs, 0, 1.0]) 
    
    return fig

def plot_ETF(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['ETF_metric'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_2$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(-0.2, 1.21, .2), fontsize=30) 
    
    plt.axis([0, epochs, -0.02, 1.2]) 

    return fig


def plot_WH_relation(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['WH_relation_metric'], 'c', marker='v', ms=16,  markevery=25, linewidth=5, alpha=0.7)
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel(r'$\mathcal{NC}_3$', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(0, 1.21, 0.2), fontsize=30)

    plt.axis([0, epochs, 0, 1.2]) 

    return fig

def plot_train_acc(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)

    plt.plot(info['train_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Training accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(20, 110, 20), fontsize=30) 

    plt.axis([0, epochs, 20, 102])

    return fig

def plot_test_acc(info, epochs, XTICKS):
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    plt.plot(info['test_acc1'], 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7)

    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('Testing accuracy', fontsize=40)
    plt.xticks(XTICKS, fontsize=30)

    plt.yticks(np.arange(20, 100.1, 10), fontsize=30)

    plt.axis([0, epochs, 20, 100])

    return fig