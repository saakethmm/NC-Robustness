from logger import CometWriter
import torch
from tqdm import tqdm
from abc import abstractmethod
from numpy import inf
import pickle
from utils import validate_nc_epoch, plot_nc

class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, metrics, optimizer, config, val_criterion):
        self.config = config
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.writer = CometWriter(
            self.logger,
            project_name = "preconditioning",
            experiment_name = config['exper_name'],
            api_key = config['comet']['api'],
            log_dir = config.log_dir,
            offline = config['comet']['offline'])

        self.writer.log_hyperparams(config.config)
        #self.writer.log_code()

        # setup GPU device if available, move model into configured device
        self.device, device_ids = self._prepare_device(config['n_gpu'])
        self.model = model.to(self.device)
        
        self.bin_gates = [p for p in self.model.parameters() if getattr(p, 'bin_gate', False)]

        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model, device_ids=device_ids)
        
        self.val_criterion = val_criterion
        self.metrics = metrics

        self.optimizer = optimizer

        cfg_trainer = config['trainer']
        self.do_adv = cfg_trainer["do_adv"]
        self.epochs = cfg_trainer['epochs']
        self.save_period = cfg_trainer['save_period']
        self.monitor = cfg_trainer.get('monitor', 'off')
        dataloader_name = config['data_loader']['type']
        if "CIFAR10" in dataloader_name:
            self.dataset = "cifar10"
        elif "MiniImageNet" in dataloader_name:
            self.dataset = "miniimagenet"
        print(f"Using {self.dataset} dataset.")

        # configuration to monitor model performance and save best
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.get('early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = config.save_dir

        # setup visualization writer instance        
        if config.resume is not None:
            self._resume_checkpoint(config.resume)
        
        self.info_dict = {
                 'collapse_metric': [],
                 'nuclear_metric': [], # #of epoch big lists, under each big list: dict with # of class keys, under each key, an array contains
                                       # all singular values of this epoch this class
                 'ETF_metric': [],
                 'WH_relation_metric': [],
                 'Wh_b_relation_metric': [],
                 'prob_margin': [],
                 'cos_margin': [],
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

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current epochs number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0

        for epoch in tqdm(range(self.start_epoch, self.epochs + 1), desc='Total progress: '):
            result = self._train_epoch(epoch)
            print("one done")
            # save logged informations into log dict
            log = {'epoch': epoch}
            for key, value in result.items():
                if key == 'metrics':
                    log.update({mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'val_metrics':
                    log.update({'val_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                elif key == 'test_metrics':
                    log.update({'test_' + mtr.__name__: value[i] for i, mtr in enumerate(self.metrics)})
                else:
                    log[key] = value

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            if (epoch - 1) % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            
            ######### Added #################################################
            self._validate_nc(epoch)
            
        with open(str(self.checkpoint_dir / 'info.pkl'), 'wb') as f: 
            pickle.dump(self.info_dict, f)
        # Plot
        fig_collapse, fig_nuclear_metric, fig_etf, fig_wh, fig_whb,\
        fig_prob_margin, fig_cos_margin, fig_train_acc, fig_test_acc = plot_nc(self.info_dict, self.epochs + 1)

        fig_collapse.savefig(str(self.checkpoint_dir / "NC_1.pdf"), bbox_inches='tight')
        fig_nuclear_metric.savefig(str(self.checkpoint_dir / "NF_metric.pdf"), bbox_inches='tight')
        fig_etf.savefig(str(self.checkpoint_dir / "NC_2.pdf"), bbox_inches='tight')
        fig_wh.savefig(str(self.checkpoint_dir / "NC_3.pdf"), bbox_inches='tight')
        fig_whb.savefig(str(self.checkpoint_dir / "NC_4.pdf"), bbox_inches='tight')
        fig_prob_margin.savefig(str(self.checkpoint_dir / "prob_margin.pdf"), bbox_inches='tight')
        fig_cos_margin.savefig(str(self.checkpoint_dir / "cos_margin.pdf"), bbox_inches='tight')
        fig_train_acc.savefig(str(self.checkpoint_dir / "train_acc.pdf"), bbox_inches='tight')
        fig_test_acc.savefig(str(self.checkpoint_dir / "test_acc.pdf"), bbox_inches='tight')
            ################################################################

        self.writer.finalize()

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.logger.warning("Warning: There\'s no GPU available on this machine,"
                                "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                                "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__

        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best
        }

        path = str(self.checkpoint_dir / f'model_epoch_{epoch}.pth')
        torch.save(state, path)
        self.logger.info("Saving current model: current model save at: {} ...".format(path))
    
    def _validate_nc(self, epoch):
        collapse_metric, nf_metric_epoch, ETF_metric, WH_relation_metric, Wh_b_relation_metric, \
        avg_prob_margin, avg_cos_margin, prob_margin_dist_fig, cos_margin_dist_fig = validate_nc_epoch(
            self.checkpoint_dir, epoch, self.model, self.data_loader, self.test_data_loader, self.info_dict,
            do_adv = self.do_adv
        )
        
        self.writer.add_scalar({'NC_1': collapse_metric}, epoch=epoch)
        self.writer.add_scalar({'NF_metric': nf_metric_epoch}, epoch=epoch)
        self.writer.add_scalar({'NC_2': ETF_metric}, epoch=epoch)
        self.writer.add_scalar({'NC_3': WH_relation_metric}, epoch=epoch)
        self.writer.add_scalar({'NC_4': Wh_b_relation_metric}, epoch=epoch)
        self.writer.add_scalar({'prob_margin': avg_prob_margin}, epoch=epoch)
        self.writer.add_scalar({'cos_margin': avg_cos_margin}, epoch=epoch)
        self.writer.add_plot('prob_margin_distribution', prob_margin_dist_fig, epoch=epoch)
        self.writer.add_plot('cos_margin_distribution', cos_margin_dist_fig, epoch=epoch)





    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints
        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))
