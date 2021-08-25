import sys
import pickle
import torchvision
import torchvision.transforms as transforms
import torch
import scipy.linalg as scilin

import models
from models.resnet import ResNet18
import argparse
import os
import numpy as np
from torchvision.datasets import CIFAR10, MNIST
from utils import load_from_state_dict


MNIST_TRAIN_SAMPLES = (5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949)
MNIST_TEST_SAMPLES = (980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009)
CIFAR10_TRAIN_SAMPLES = 10 * (5000,)
CIFAR10_TEST_SAMPLES = 10 * (1000,)

class CIFAR10_subs(CIFAR10):
    def __init__(self, **kwargs):
        super(CIFAR10_subs, self).__init__(**kwargs)
        if self.train:
            with open(self.root+'/cifar10_uniform_128/train_label.pkl', 'rb') as f:
                train_all = pickle.load(f)
                self.targets = train_all["label"]
        else:
            with open(self.root+'/cifar10_uniform_128/test_label.pkl', 'rb') as f:
                test_all = pickle.load(f)
                self.targets = test_all["label"]

# def compute_accuracy(output, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     maxk = max(topk)
#     batch_size = target.size(0)
    
#     pred = output.view(batch_size, 128, 10)
#     pred = torch.linalg.norm(pred, dim=1)
    
#     _, pred = pred.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res

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

def parse_eval_args():
    parser = argparse.ArgumentParser()

    # parameters
    # Model Selection
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--no-bias', dest='bias', action='store_false')
    parser.add_argument('--ETF_fc', dest='ETF_fc', action='store_true')
    parser.add_argument('--fixdim', dest='fixdim', type=int, default=0)
    parser.add_argument('--SOTA', dest='SOTA', action='store_true')
    
    # MLP settings (only when using mlp and res_adapt(in which case only width has effect))
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=6)

    # Hardware Setting
    parser.add_argument('--gpu_id', type=int, default=0)

    # Directory Setting
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'cifar10_random'], default='cifar10')
    parser.add_argument('--data_dir', type=str, default='/scratch/xl998/DL/data')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--p_name', type=str, default=None)

    # Learning Options
    parser.add_argument('--epochs', type=int, default=150, help='Max Epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--sample_size', type=int, default=None, help='sample size PER CLASS')

    args = parser.parse_args()

    return args

class FCFeatures:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in):
        self.outputs.append(module_in)

    def clear(self):
        self.outputs = []


def compute_info(args, model, fc_features, dataloader, isTrain=True):
    mu_G = 0
    mu_c_dict = dict()
    before_class_dict = dict()
    after_class_dict = dict()
    top1 = AverageMeter()
    top5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
            #fea, outputs = model(inputs)

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

    if args.dataset == 'mnist':
        if isTrain:
            mu_G /= sum(MNIST_TRAIN_SAMPLES)
            for i in range(len(MNIST_TRAIN_SAMPLES)):
                mu_c_dict[i] /= MNIST_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(MNIST_TEST_SAMPLES)
            for i in range(len(MNIST_TEST_SAMPLES)):
                mu_c_dict[i] /= MNIST_TEST_SAMPLES[i]
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            mu_G /= sum(CIFAR10_TRAIN_SAMPLES)
            for i in range(len(CIFAR10_TRAIN_SAMPLES)):
                mu_c_dict[i] /= CIFAR10_TRAIN_SAMPLES[i]
        else:
            mu_G /= sum(CIFAR10_TEST_SAMPLES)
            for i in range(len(CIFAR10_TEST_SAMPLES)):
                mu_c_dict[i] /= CIFAR10_TEST_SAMPLES[i]

    return mu_G, mu_c_dict, before_class_dict, after_class_dict, top1.avg, top5.avg


def compute_Sigma_W(args, model, fc_features, mu_c_dict, dataloader, isTrain=True):

    Sigma_W = 0
    for batch_idx, (inputs, targets) in enumerate(dataloader):

        inputs, targets = inputs.to(args.device), targets.to(args.device)

        with torch.no_grad():
            outputs = model(inputs)
            #fea, outputs = model(inputs)
            
        features = fc_features.outputs[0][0]
        fc_features.clear()

        for b in range(len(targets)):
            y = targets[b].item()
            Sigma_W += (features[b, :] - mu_c_dict[y]).unsqueeze(1) @ (features[b, :] - mu_c_dict[y]).unsqueeze(0)

    if args.dataset == 'mnist':
        if isTrain:
            Sigma_W /= sum(MNIST_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(MNIST_TEST_SAMPLES)
    elif args.dataset == 'cifar10' or args.dataset == 'cifar10_random':
        if isTrain:
            Sigma_W /= sum(CIFAR10_TRAIN_SAMPLES)
        else:
            Sigma_W /= sum(CIFAR10_TEST_SAMPLES)

    return Sigma_W.cpu().numpy()


def compute_Sigma_B(mu_c_dict, mu_G):
    Sigma_B = 0
    K = len(mu_c_dict)
    for i in range(K):
        Sigma_B += (mu_c_dict[i] - mu_G).unsqueeze(1) @ (mu_c_dict[i] - mu_G).unsqueeze(0)

    Sigma_B /= K

    return Sigma_B.cpu().numpy()

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
    
    ################
    #### Added  ####
#     if W.shape[0] == 1280:
#         W_clone = W.clone()
#         W_clone = W_clone.view(128,10,512)
#         W_clone = torch.mean(W_clone, 0)
#     WH = W_clone.cpu() @ M
    ################
    WH = W.cpu() @ M ####Uncomment this if delete above
    
    res = torch.norm(WH / torch.norm(WH, p='fro') - sub, p='fro')

    return res.detach().cpu().numpy()


def main():
    args = parse_eval_args()

    if args.load_path is None:
        sys.exit('Need to input the path to a pre-trained model!')

    #device = torch.device("cuda:"+str(args.gpu_id) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Dataset part
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform_train)
#     trainset = CIFAR10_subs(
#         root=args.data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform_test)
#     testset = CIFAR10_subs(
#         root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    
    # Model part
    if args.ETF_fc:
        model = ResNet18(num_classes=10,
                     norm_layer_type="bn",
                     conv_layer_type="conv",
                     linear_layer_type="linear",
                     activation_layer_type="relu",
                     etf_fc = True).to(device)
    else:
        model = ResNet18(num_classes=10,
                     norm_layer_type="bn",
                     conv_layer_type="conv",
                     linear_layer_type="linear",
                     activation_layer_type="relu",
                     etf_fc = False).to(device)

    fc_features = FCFeatures()
    model.linear.register_forward_pre_hook(fc_features)

    info_dict = {
                 'collapse_metric': [],
                 'ETF_metric': [],
                 'WH_relation_metric': [],
                 'W': [],
                 'b': [],
                 'mu_G_train': [],
                 'mu_G_test': [],
                 'mu_c_dict_train': [],
                 'mu_c_dict_test': [],
                 'before_class_dict_train': {},
                 'after_class_dict_train': {},
                 'before_class_dict_test': {},
                 'after_class_dict_test': {},
                 'train_acc1': [],
                 'train_acc5': [],
                 'test_acc1': [],
                 'test_acc5': []
                 }

    for i in range(args.epochs):
#         if i == 0:
#             state_d = torch.load(args.load_path + 'model_epoch_' + str(i + 1) + '.pth', map_location=device)["state_dict"]
#             load_from_state_dict(model, state_d)
#         else:
#             model.load_state_dict(torch.load(args.load_path + 'model_epoch_' + str(i + 1) + '.pth', map_location=device)["state_dict"])
        model.load_state_dict(torch.load(args.load_path + 'model_epoch_' + str(i + 1) + '.pth', map_location=device)["state_dict"])
    #### Indentation
        #model.load_state_dict(torch.load(args.load_path, map_location=device)["state_dict"])
        model.eval()

        for n, p in model.named_parameters():
            if 'linear.weight' in n:
                W = p.clone()
            if 'linear.bias' in n:
                b = p.clone()

        mu_G_train, mu_c_dict_train, before_class_dict_train, after_class_dict_train, train_acc1, train_acc5 = compute_info(args, model, fc_features, trainloader, isTrain=True)
        mu_G_test, mu_c_dict_test, before_class_dict_test, after_class_dict_test, test_acc1, test_acc5 = compute_info(args, model, fc_features, testloader, isTrain=False)

        Sigma_W = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, trainloader, isTrain=True)
        # Sigma_W_test_norm = compute_Sigma_W(args, model, fc_features, mu_c_dict_train, testloader, isTrain=False)
        Sigma_B = compute_Sigma_B(mu_c_dict_train, mu_G_train)

        collapse_metric = np.trace(Sigma_W @ scilin.pinv(Sigma_B)) / len(mu_c_dict_train)
        ETF_metric = compute_ETF(W)
        WH_relation_metric = compute_W_H_relation(W, mu_c_dict_train, mu_G_train) # Added back

        info_dict['collapse_metric'].append(collapse_metric)
        info_dict['ETF_metric'].append(ETF_metric)
        info_dict['WH_relation_metric'].append(WH_relation_metric) # Added back
        info_dict['mu_G_train'].append(mu_G_train.detach().cpu().numpy())
        info_dict['mu_G_test'].append(mu_G_test.detach().cpu().numpy())
        info_dict['mu_c_dict_train'] = mu_c_dict_train
        info_dict['mu_c_dict_test'] = mu_c_dict_test
        info_dict['before_class_dict_train'] = before_class_dict_train
        info_dict['after_class_dict_train'] = after_class_dict_train
        info_dict['before_class_dict_test'] = before_class_dict_test
        info_dict['after_class_dict_test'] = after_class_dict_test
        info_dict['W'].append((W.detach().cpu().numpy()))
        if args.bias:
            info_dict['b'].append(b.detach().cpu().numpy())

        info_dict['train_acc1'].append(train_acc1)
        info_dict['train_acc5'].append(train_acc5)
        info_dict['test_acc1'].append(test_acc1)
        info_dict['test_acc5'].append(test_acc5)
        
        print(f"Epoch {i} is processed")
        
    with open(args.load_path + args.p_name, 'wb') as f: #'info_normal.pkl'
        pickle.dump(info_dict, f)


if __name__ == "__main__":
    main()