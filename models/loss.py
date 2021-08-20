import torch.nn.functional as F
import torch
import numpy as np
from parse_config import ConfigParser
import torch.nn as nn
from torch.autograd import Variable
import math
from utils import sigmoid_rampup, sigmoid_rampdown, cosine_rampup, cosine_rampdown, linear_rampup

def cross_entropy(output, target):
    return F.cross_entropy(output, target)


class elr_loss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3):
        super(elr_loss, self).__init__()
        self.num_classes = num_classes
        self.config = ConfigParser.get_instance()
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp, self.num_classes)
        self.beta = beta
        

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))
        ce_loss = F.cross_entropy(output, label)
        elr_reg = ((1-(self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss +  self.config['train_loss']['args']['lambda']*elr_reg
        return  final_loss

    
class elr_plus_loss(nn.Module):
    def __init__(self, num_examp, config, device, num_classes=10, beta=0.3):
        super(elr_plus_loss, self).__init__()
        self.config = config
        self.pred_hist = (torch.zeros(num_examp, num_classes)).to(device)
        self.q = 0
        self.beta = beta
        self.num_classes = num_classes

    def forward(self, iteration, output, y_labeled):
        y_pred = F.softmax(output,dim=1)

        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)

        if self.num_classes == 100:
            y_labeled = y_labeled*self.q
            y_labeled = y_labeled/(y_labeled).sum(dim=1,keepdim=True)

        ce_loss = torch.mean(-torch.sum(y_labeled * F.log_softmax(output, dim=1), dim = -1))
        reg = ((1-(self.q * y_pred).sum(dim=1)).log()).mean()
        final_loss = ce_loss + sigmoid_rampup(iteration, self.config['coef_step'])*(self.config['train_loss']['args']['lambda']*reg)
      
        return  final_loss, y_pred.cpu().detach()

    def update_hist(self, epoch, out, index= None, mix_index = ..., mixup_l = 1):
        y_pred_ = F.softmax(out,dim=1)
        self.pred_hist[index] = self.beta * self.pred_hist[index] +  (1-self.beta) *  y_pred_/(y_pred_).sum(dim=1,keepdim=True)
        self.q = mixup_l * self.pred_hist[index]  + (1-mixup_l) * self.pred_hist[index][mix_index]
        

def cross_entropy_iden(output, target):
    features, output = output
    # Add a nuclear norm maximization term for the correct class (features)
    #return F.cross_entropy(output, target) - 0.01 * torch.linalg.norm(features, ord="nuc")
    identity_constraint = torch.linalg.norm(torch.eye(output.shape[0]).to(output.device) - features @ features.T)
    return F.cross_entropy(output, target) + 0.0001 * identity_constraint


def squaredLoss(feature, target, model):
    # changed from https://github.com/MTandHJ/roboc/blob/master/src/loss_zoo.py
    weight = model.linear.weight.data.clone()
    targets_fea = weight[target]
    return F.mse_loss(feature, targets_fea, reduction="mean")