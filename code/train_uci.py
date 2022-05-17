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

from get_uci_data import GetUCI
from utils.hparams import HParams
from utils.train_utils import run_epoch, visualization, plot_result
from torch.utils.tensorboard import SummaryWriter
import time
from transformers import Encoder

import copy

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

date = time.strftime('%Y-%m-%d', time.localtime(time.time()))

# columns_list = ['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
#          'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
#          'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',]

# columns_list = ['CO(GT)', 'AH', 'NOx(GT)', 'T', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)',
#                 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)',  'RH', ]

# columns_list = ['AH', 'CO(GT)', 'NOx(GT)']

# columns_list = ['C6H6(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', ]
# columns_list = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', ]
columns_list = ['PT08.S5(O3)', ]

# columns_list = ['C6H6(GT)', ]
# columns_list = ['PT08.S1(CO)', ]
# columns_list = ['PT08.S2(NMHC)', ]
# columns_list = ['PT08.S3(NOx)', ]
# columns_list = ['PT08.S4(NO2)', ]
# columns_list = ['PT08.S5(O3)', ]
# copy_column = 'NOx(GT)'
# copy_column = 'CO(GT)'
copy_column = 'NO2(GT)'
alpha_list = [0, 0.9, 0.97, 0.98, 0.99, 1 ]
# alpha_list = [0.25, 0.5, 0.75, 0.8, 0.95, 0.96]

copy_mask = True

for columns in columns_list:
    for alpha in alpha_list:
        start = time.time()
        level = 1
        pre = GetUCI()
        data, mean, std = pre.read_data(columns)
        data[:, 0] = (data[:, 0] - mean) / std

        if copy_mask:
            copy_data, _, _ = pre.read_data(copy_column)
            ori_data = copy.deepcopy(data[:, 0])
            ori_mask = copy.deepcopy(data[:, 1])
            data[:, 1] *= copy_data[:, 1]

        data[:, 0] *= data[:, 1]

        # plot_ori(data)
        data = np.concatenate([data, data[:, 1:]], axis=-1)
        mask_ori = torch.Tensor(data[:, 1]).cuda()

        if copy_mask:
            folder_name = "1_{compare_name}_epochs_{epochs}_lr_{lr}_ln_{layer_norm}_layers_{n_layers}_" \
                          "clip_{clip}_drop_{drop}_sita_{sita}_alpha_{alpha}".format(
                epochs=params.max_epoch, lr=params.lr, layer_norm=params.layer_norm, n_layers=params.n_layers,
                clip=params.clip, sita=params.sita, alpha=alpha, compare_name=copy_column, drop=params.dropout
            )
        else:
            folder_name = "epochs_{epochs}_lr_{lr}_ln_{layer_norm}_layers_{n_layers}_" \
                          "clip_{clip}_drop_{drop}_sita_{sita}_alpha_{alpha}".format(
                epochs=params.max_epoch, lr=params.lr, layer_norm=params.layer_norm, n_layers=params.n_layers,
                clip=params.clip, sita=params.sita, alpha=alpha, drop=params.dropout
            )

        exp_dir = './log_uci/' + columns + '/' + date + '_' + folder_name
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

        # epochs = params.max_epoch
        score_total = 0

        print(columns)
        print(folder_name)

        while True:
            # load data
            dataset, validity_list, percent_true, impute_num, last_index = pre.split_data(data)
            lens = dataset.shape[1]

            dataset = torch.Tensor(dataset)
            validity_list = torch.Tensor(validity_list).long()
            train_data = dataset[validity_list]
            print('[level: %d]  [train percent: %2f%%]  [impute num: %s]  [train data shape: %s]' %
                  (level, percent_true*100, impute_num, dataset.numpy().shape))

            # print(model)

            optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
            print("Start Training")
            writer = SummaryWriter(exp_dir)

            best_loss = 1e7

            optimizer.param_groups[0]['lr'] = params.lr

            epochs = min(params.max_epoch, impute_num)

            for epoch in range(1, int(epochs)+1):

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
                    plot, level, model, dataset.cuda(), train_data, validity_list, columns, date,
                    folder_name, loss, lens, last_index, batch_size=params.batch_size)

                epoch_score = mean_change + std_change + skew_change + kurt_change
                output_str = '[%d] [%d]/[%d] mse: %4f, loss: %4f, best loss: %4f' % \
                             (level, epoch, epochs, mse_loss, loss, best_loss)

                print(output_str)
                logging.info(output_str)
                writer.add_scalar('Score/level %s'%level, epoch_score, epoch)

            checkpoint = torch.load(os.path.join(exp_dir, 'ckpt', "level_%d_best.pt" % level))
            model.load_state_dict(checkpoint)

            np.save(root_path + " mask.npy", data[:, 1])

            predict, mean_change, std_change, skew_change, kurt_change, _ = \
                visualization(1, level, model, dataset.cuda(), train_data, validity_list,
                              columns, date, folder_name, best_loss, lens, last_index,
                              batch_size=params.batch_size)

            score = mean_change + std_change + skew_change + kurt_change
            score_total += score
            predict = predict.cpu().numpy()
            data[: predict.shape[0], 0] += predict[:, 0] * (predict[:, 1] * (1 - data[: predict.shape[0], 1]))
            data[: predict.shape[0], 1] = predict[:, 1]

            level += 1

            if percent_true == 1:
                break

        if copy_mask:
            plot_result(data, columns, date, score_total, folder_name, ori_data, ori_mask)
        else:
            plot_result(data, columns, date, score_total, folder_name)
        end = time.time()
        print('time: %4f H' % ((end - start) / 3600))