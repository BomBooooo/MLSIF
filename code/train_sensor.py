import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
p = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(p)
import argparse
import logging
import torch
import numpy as np
from pprint import pformat, pprint

from get_dataset import Preprocess
from utils.hparams import HParams
from utils.train_utils import run_epoch, visualization, plot_result
from torch.utils.tensorboard import SummaryWriter
import time
from transformers import Encoder

import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--cfg_file', default='./params.json', type=str)
args = parser.parse_args()
params = HParams(args.cfg_file)
# pprint(params.dict)
np.random.seed(params.seed)
torch.manual_seed(params.seed)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)  # 检测模型中inplace operation报错的具体位置

time = time.strftime('%Y-%m-%d', time.localtime(time.time()))

data_path='/home/yangjinsheng/tcn200/'
data_name = 'sensor_data/'

little_loss_list = [# missing num 600
    # 'gpcjs_DDISP-007_45710417_ddisp_x.csv',
    # 'gpcjs_DDISP-007_45710417_ddisp_y.csv',
    # 'gpcjs_DDISP-008_45710418_ddisp_x.csv',#
    'gpcjs_DDISP-008_45710418_ddisp_y.csv',#
]

middle_loss_list= [# missing num 1000
    'gpcjs_Water_01_55710412_water.csv',#
    # 'gpcjs_Water_02_55710413_water.csv',
    # 'gpcjs_Water_03_55710411_water.csv',#
]

many_loss_list = [# missing num 6000
    # 'gpcjs_DDISP-011_45710421_ddisp_x.csv',#
    # 'gpcjs_DDISP-011_45710421_ddisp_y.csv',#
    # 'gpcjs_DDISP-012_45710422_ddisp_x.csv',
    'gpcjs_DDISP-012_45710422_ddisp_y.csv',
]

freq = '1H'

# alpha_list = [0.96, 0.97, 0.98]
alpha_list = [0.99, 0.95, 0.9, 0.8]

for node_name in many_loss_list:
    for alpha in alpha_list:

        level = 1

        pre = Preprocess(data_path=data_path, data_name=data_name, node_name=node_name, freq=freq)
        data, mean, std, name = pre.read_data()
        data[:, 0] = (data[:, 0] - mean) / std
        data[:, 0] *= data[:, 1]
        data = np.concatenate([data, data[:, 1:]], axis=-1)
        mask_ori = torch.Tensor(data[:, 1]).cuda()

        folder_name = "final_version2_epochs_{epochs}_lr_{lr}_ln_{layer_norm}_layers_{n_layers}_" \
                      "clip_{clip}_sita_{sita}_alpha_{alpha}" .format(
            epochs=params.max_epoch, lr=params.lr, layer_norm=params.layer_norm, n_layers=params.n_layers,
            clip=params.clip, sita=params.sita, alpha=alpha
        )

        exp_dir = './log/' + name + '/' + time + '_' + folder_name
        # creat exp dir
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        if not os.path.exists(os.path.join(exp_dir, 'ckpt')):
            os.mkdir(os.path.join(exp_dir, 'ckpt'))

        ############################################################
        logging.basicConfig(filename=exp_dir + '/train.log',
                            filemode='w',
                            level=logging.INFO,
                            format='%(message)s')
        logging.info(pformat(params.dict))
        ############################################################
        model = Encoder(
            max_time_scale=params.max_time_scale,
            time_enc_dim=params.time_enc_dim,
            time_dim=params.time_dim,
            expand_dim=params.expand_dim,
            mercer=params.mercer,
            n_layers=params.n_layers,
            n_head=params.n_heads,
            d_k=params.att_dims,
            d_v=params.att_dims,
            d_model=params.model_dims,
            d_inner=params.inner_dims,
            d_data=1,
            dropout=params.dropout,
            use_layer_norm=params.layer_norm,
            use_gap_encoding=params.use_gap_encoding,
            adapter=params.adapter,
            use_mask=params.att_mask
        )
        model = nn.DataParallel(model).to(device)
        # model = nn.parallel.DistributedDataParallel(model).to(device)

        # epochs = params.max_epoch
        score_total = 0

        print(node_name)
        print(folder_name)

        while True:
            # load data
            dataset, validity_list, percent_true, impute_num, last_index = pre.split_data(data)
            lens = dataset.shape[1]

            lr = params.lr
            if dataset is None:
                continue
            elif dataset.shape[0] < 10:
                print(node_name, 'has shape: %s, do not have enough data!' % str(data.shape))
                continue

            dataset = torch.Tensor(dataset)
            validity_list = torch.Tensor(validity_list).long()
            train_data = torch.Tensor(dataset)[validity_list]
            print('[level: %d]  [train percent: %2f%%]  [impute num: %s]  [train data shape: %s]' %
                  (level, percent_true*100, impute_num, dataset.numpy().shape))

            # print(model)

            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=params.weight_decay)
            print("Start Training")
            writer = SummaryWriter(exp_dir)

            best_loss = 1e6

            optimizer.param_groups[0]['lr'] = lr

            epochs = int(min(params.max_epoch, impute_num))

            for epoch in range(1, epochs+1):

                # print('\n[*] Start Train')
                mse_loss, loss = run_epoch(model, train_data, params.sita, alpha, params.clip, optimizer=optimizer,
                                           batch_size=params.batch_size)

                plot = 0
                if loss < best_loss:
                    plot += 1
                    best_loss = loss
                    best_index = epoch
                    torch.save(model.state_dict(), exp_dir + '/ckpt/level_%d_best.pt' % (level))
                # plot
                _, mean_change, std_change, skew_change, kurt_change, root_path = visualization(
                    plot, level, model, torch.Tensor(dataset).cuda(), train_data, validity_list, node_name, time,
                    folder_name, loss, lens, last_index, batch_size=params.batch_size)

                epoch_score = mean_change + std_change + skew_change + kurt_change
                output_str = '[%d] [%d]/[%d] mse: %4f, loss: %4f, best loss: %4f' % \
                             (level, epoch, epochs, mse_loss, loss, best_loss)

                print(output_str)
                logging.info(output_str)
                writer.add_scalar('Score/level %s' % level, epoch_score, epoch)

            checkpoint = torch.load(os.path.join(exp_dir, 'ckpt', "level_%d_best.pt" % level))
            model.load_state_dict(checkpoint)

            predict, mean_change, std_change, skew_change, kurt_change, _ = \
                visualization(1, level, model, torch.Tensor(dataset).cuda(), train_data, validity_list,
                              node_name, time, folder_name, best_loss, lens, last_index,
                              batch_size=params.batch_size)

            score = mean_change + std_change + skew_change + kurt_change
            score_total += score
            predict = predict.detach().cpu().numpy()
            data[: predict.shape[0], 0] += predict[:, 0] * (predict[:, 1] * (1 - data[: predict.shape[0], 1]))
            data[: predict.shape[0], 1] = predict[:, 1]

            level += 1

            if percent_true == 1:
                break
        plot_result(data, node_name, time, score_total, folder_name)