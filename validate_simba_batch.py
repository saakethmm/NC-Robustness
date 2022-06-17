import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from parse_config import ConfigParser
import simba_utils
import math
import random
import argparse
import os
import sys
import pdb
sys.path.append('pytorch-cifar')
import models.model as module_arch
from simba import SimBA
from utils import inf_loop, get_logger, Timer, load_from_state_dict, set_seed
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES']='3'
import setproctitle
setproctitle.setproctitle('pgdAttack@xinshiduo')


def main(args, config: ConfigParser):

    if not os.path.exists(args.sampled_image_dir):
        os.mkdir(args.sampled_image_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model and dataset
    model = getattr(module_arch, config["arch"]["type"])(
        num_classes = config["arch"]["args"]["num_classes"],
        norm_layer_type = config["arch"]["args"]["norm_layer_type"],
        conv_layer_type = config["arch"]["args"]["conv_layer_type"],
        linear_layer_type = config["arch"]["args"]["linear_layer_type"],
        activation_layer_type = config["arch"]["args"]["activation_layer_type"],
        etf_fc = config["arch"]["args"]["etf_fc"]
    ).to(device)
    
    image_size = 32
    

    advlist=[]
    xlist=[]
    probslist=[]
    succslist=[]
    querieslist=[]
    l2_normslist=[]
    linf_normslist=[]
    # load sampled images or sample new ones
    # this is to ensure all attacks are run on the same set of correctly classified images
    for step in range(args.cycle,args.epoch+args.cycle,args.cycle):
        checkpoint = torch.load(args.checkpoint_path+"model_epoch_"+str(step)+".pth")
        load_from_state_dict(model, checkpoint["state_dict"])
        model.eval()
        xlist.append(step)
        testset = dset.CIFAR100(root=config['data_loader']['args']['data_dir'], train=False, download=False, transform=simba_utils.CIFAR100_TRANSFORM)
        attacker = SimBA(model, args.dataset, image_size)   
        # batchfile = '%s/images_%s_%d.pth' % (args.sampled_image_dir, args.dataset, args.num_runs)
        # if os.path.isfile(batchfile):
        #     checkpoint = torch.load(batchfile)
        #     images = checkpoint['images']
        #     labels = checkpoint['labels']
        # else:
        images = torch.zeros(args.num_runs, 3, image_size, image_size)
        labels = torch.zeros(args.num_runs).long()
        preds = labels + 1
        while preds.ne(labels).sum() > 0:
            idx = torch.arange(0, images.size(0)).long()[preds.ne(labels)]
            for i in list(idx):
                images[i], labels[i] = testset[random.randint(0, len(testset) - 1)]
            preds[idx], _ = simba_utils.get_preds(model, images[idx], args.dataset, batch_size=args.batch_size)
            # torch.save({'images': images, 'labels': labels}, batchfile)

        if args.order == 'rand':
            n_dims = 3 * args.freq_dims * args.freq_dims
        else:
            n_dims = 3 * image_size * image_size
        if args.num_iters > 0:
            max_iters = int(min(n_dims, args.num_iters))
        else:
            max_iters = int(n_dims)
        N = int(math.floor(float(args.num_runs) / float(args.batch_size)))
        for i in range(N):
            upper = min((i + 1) * args.batch_size, args.num_runs)
            images_batch = images[(i * args.batch_size):upper]
            labels_batch = labels[(i * args.batch_size):upper]
            # replace true label with random target labels in case of targeted attack
            if args.targeted:
                labels_targeted = labels_batch.clone()
                while labels_targeted.eq(labels_batch).sum() > 0:
                    labels_targeted = torch.floor(10 * torch.rand(labels_batch.size())).long()
                labels_batch = labels_targeted
            adv, probs, succs, queries, l2_norms, linf_norms = attacker.simba_batch(
                images_batch, labels_batch, max_iters, args.freq_dims, args.stride, args.epsilon, linf_bound=args.linf_bound,
                order=args.order, targeted=args.targeted, pixel_attack=args.pixel_attack, log_every=args.log_every)
            if i == 0:
                all_adv = adv
                all_probs = probs
                all_succs = succs
                all_queries = queries
                all_l2_norms = l2_norms
                all_linf_norms = linf_norms
            else:
                all_adv = torch.cat([all_adv, adv], dim=0)
                all_probs = torch.cat([all_probs, probs], dim=0)
                all_succs = torch.cat([all_succs, succs], dim=0)
                all_queries = torch.cat([all_queries, queries], dim=0)
                all_l2_norms = torch.cat([all_l2_norms, l2_norms], dim=0)
                all_linf_norms = torch.cat([all_linf_norms, linf_norms], dim=0)
            # if args.pixel_attack:
            #     prefix = 'pixel'
            # else:
            #     prefix = 'dct'
            # if args.targeted:
            #     prefix += '_targeted'
        probslist.append(all_probs.detach().cpu().numpy().mean())
    fig = plt.figure(figsize=(10, 8))
    plt.grid(True)
    
    plt.plot(xlist,probslist, 'c', marker='v', ms=16, markevery=25, linewidth=5, alpha=0.7 ,label='top1')
    plt.legend()
    plt.xlabel('Epoch', fontsize=40)
    plt.ylabel('probs_BA', fontsize=40)
    fig.savefig(str(args.checkpoint_path + "BA.pdf"), bbox_inches='tight')

            

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Runs SimBA on a set of images')
    # parser.add_argument('--data_root', type=str, required=True, help='root directory of imagenet data')
    parser.add_argument('--result_dir', type=str, default='./res_saved/save_cifar100', help='directory for saving results')
    parser.add_argument('--sampled_image_dir', type=str, default='./res_saved/save_cifar100', help='directory to cache sampled images')
    # parser.add_argument('--model', type=str, default='resnet18', help='type of base model to use')
    # parser.add_argument('--model_ckpt', type=str, required=True, help='model checkpoint location')
    parser.add_argument('--num_runs', type=int, default=1000, help='number of image samples')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for parallel runs')
    parser.add_argument('--num_iters', type=int, default=200, help='maximum number of iterations, 0 for unlimited')
    parser.add_argument('--log_every', type=int, default=10, help='log every n iterations')
    parser.add_argument('--epsilon', type=float, default=0.2, help='step size per iteration')
    parser.add_argument('--linf_bound', type=float, default=0.0, help='L_inf bound for frequency space attack')
    parser.add_argument('--freq_dims', type=int, default=32, help='dimensionality of 2D frequency space')
    parser.add_argument('--order', type=str, default='rand', help='(random) order of coordinate selection')
    parser.add_argument('--stride', type=int, default=7, help='stride for block order')
    parser.add_argument('--targeted', action='store_true', help='perform targeted attack')
    parser.add_argument('--pixel_attack', action='store_true', help='attack in pixel space')
    parser.add_argument('--save_suffix', type=str, default='', help='suffix appended to save file')
    parser.add_argument('--dataset', default='cifar100',type=str,  help='root directory of imagenet data')
    parser.add_argument('-c', '--config', type=str, default="./config_robust_train_cifar100.json",
                        help='config file path (default: None)')
    parser.add_argument('-s', '--checkpoint_path', type=str, default="/data/xinshiduo/new_code/copy2/res_saved/models/Robust_experiment_CIFAR100CIFAR100DataLoaderresnet18-num_classes-100-norm_layer_type-bn-conv_layer_type-conv-linear_layer_type-linear-activation_layer_type-relu-etf_fc-False-Seed=8/0604_021417/",
                        help='path to find model checkpoint (default: None)')
    parser.add_argument('-d', '--device', default='3', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-e', '--epoch', default=200, type=int,
                        help='---')
    parser.add_argument('-r', '--resume', default=None, type=str,
    help='path to latest checkpoint (default: None)')
    parser.add_argument('--cycle', default=5, type=int,
                        help='---')
    # parser.add_argument('--result_dir', type=str, default='saved', 
    #                     help='directory for saving results')
    parser.add_argument('--seed', type=int, default=6, 
                        help='Random seed')

    args = parser.parse_args()
    config = ConfigParser.get_instance(parser)
    set_seed(manualSeed = args.seed)
    main(args, config)
