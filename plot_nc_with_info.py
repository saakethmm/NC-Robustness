from utils import validate_nc_epoch, plot_nc
import pickle

f = open(r"/data/xinshiduo/code/NC_good_or_bad-main/res_saved/models/Robust_experimentCIFAR100DataLoaderresnet18-num_classes-100-norm_layer_type-bn-conv_layer_type-conv-linear_layer_type-linear-activation_layer_type-relu-etf_fc-False-Seed=8/0525_151134/info.pkl",'rb')
info_dict = pickle.load(f)
checkpoint_dir = "/data/xinshiduo/code/NC_good_or_bad-main/res_saved/models/Robust_experimentCIFAR100DataLoaderresnet18-num_classes-100-norm_layer_type-bn-conv_layer_type-conv-linear_layer_type-linear-activation_layer_type-relu-etf_fc-False-Seed=8/0525_151134/"
# Plot
fig_collapse, fig_nuclear_metric, fig_etf, fig_wh, fig_whb,\
fig_prob_margin, fig_cos_margin, fig_train_acc, fig_test_acc = plot_nc(info_dict, 200 + 1)

fig_collapse.savefig(str(checkpoint_dir + "NC_1.pdf"), bbox_inches='tight')
fig_nuclear_metric.savefig(str(checkpoint_dir + "NF_metric.pdf"), bbox_inches='tight')
fig_etf.savefig(str(checkpoint_dir + "NC_2.pdf"), bbox_inches='tight')
fig_wh.savefig(str(checkpoint_dir + "NC_3.pdf"), bbox_inches='tight')
fig_whb.savefig(str(checkpoint_dir + "NC_4.pdf"), bbox_inches='tight')
fig_prob_margin.savefig(str(checkpoint_dir + "prob_margin.pdf"), bbox_inches='tight')
fig_cos_margin.savefig(str(checkpoint_dir + "cos_margin.pdf"), bbox_inches='tight')
fig_train_acc.savefig(str(checkpoint_dir + "train_acc.pdf"), bbox_inches='tight')
fig_test_acc.savefig(str(checkpoint_dir + "test_acc.pdf"), bbox_inches='tight')