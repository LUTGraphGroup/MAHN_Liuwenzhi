import warnings  # 提供了一种处理警告信息的机制
import torch
from train import Train
from utils import plot_auc_curves, plot_prc_curves


if __name__ == '__main__':
    warnings.filterwarnings("ignore")  # 在终端中忽略警告消息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fprs, tprs, auc, precisions, recalls, prc = Train(device, directory='..\data',
                                                      epochs=1000,
                                                      attn_size=64,
                                                      attn_heads=6,
                                                      out_dim=64,
                                                      dropout=0.2,
                                                      slope=0.2,
                                                      lr=0.001,
                                                      wd=5e-3,
                                                      random_seed=1234,
                                                      sample_num=50,
                                                      model_type='MAHN')

    plot_auc_curves(fprs, tprs, auc, directory='../result', name='test_auc_1')
    plot_prc_curves(precisions, recalls, prc, directory='../result', name='test_prc_1')