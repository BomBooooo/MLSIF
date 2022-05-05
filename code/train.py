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
import copy

from get_uci_data import GetUCI
from utils.hparams import HParams
from utils.train_utils import run_epoch, visualization
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

columns_list = ['C6H6(GT)', 'PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)', ]

copy_column = 'NOx(GT)'
# copy_column = 'CO(GT)'
# copy_column = 'NO2(GT)'

copy_mask = True

for columns in columns_list:

    pre = GetUCI()
    dataset, mean, std = pre.read_data(columns)
    dataset[:, 0] = (dataset[:, 0] - mean) / std
    if copy_mask:
        copy_data, _, _ = pre.read_data(copy_column)
        ori_data = copy.deepcopy(dataset[:, 0])
        ori_mask = copy.deepcopy(dataset[:, 1])
        dataset[:, 1] *= copy_data[:, 1]
    dataset[:, 0] *= dataset[:, 1]
    data, _, _, _, last_index = pre.split_data(dataset)
    data = torch.Tensor(data).cuda()

    folder_name = "{compare}_epochs_{epochs}_lr_{lr}_layers_{n_layers}_clip_{clip}_alpha_{alpha}".format(
        epochs=params.epochs, lr=params.lr, n_layers=params.n_layers, clip=params.clip, alpha=params.alpha,
        compare=copy_column
    )
    exp_dir = './log/' + columns + '/' + time + '-' + folder_name
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

    test_index = torch.from_numpy(np.random.choice(torch.arange(data.shape[0]), int(data.shape[0]*0.2), replace=False)).long()
    train_index = torch.from_numpy(np.array(list(set(range(data.shape[0])) - set(test_index.numpy())), dtype='float32')).long()
    train_data = data[train_index]
    test_data = data[test_index]
    print('train data shape:', train_data.shape)
    print('test data shape:', test_data.shape)

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
        d_data=train_data[:, :, 0:1].shape[-1],
        dropout=params.dropout,
        use_layer_norm=params.layer_norm,
        use_gap_encoding=params.use_gap_encoding,
        adapter=params.adapter,
        use_mask=params.att_mask,
        confidence=params.confidence
    )
    model = nn.DataParallel(model).to(device)
    # print(model)
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    print("Start Training")
    writer = SummaryWriter(exp_dir)

    best_test_loss = 1e6

    optimizer.param_groups[0]['lr'] = params.lr
    start_epoch = 0

    for epoch in range(start_epoch, params.epochs):
        # print('\n[*] Start Train')
        epoch_mean_train_loss = run_epoch(True, model, train_data, params.alpha, params.clip, optimizer=optimizer)
        # print('\n[*] Start Test')
        epoch_mean_test_loss = run_epoch(False, model, test_data, params.alpha, params.clip, optimizer=optimizer)

        if epoch_mean_test_loss < best_test_loss:
            best_test_loss = epoch_mean_test_loss
            torch.save(model.state_dict(), exp_dir + '/ckpt/best_model.pt')
            if copy_mask:
                visualization(model, data, last_index, columns, time, folder_name, epoch, [ori_data, ori_mask],
                              batch_size=params.batch_size)
            else:
                visualization(model, data, last_index, columns, time, folder_name, epoch, [[], []],
                              batch_size=params.batch_size)
        output_str = '[%d]/[%d] Training Loss: %4f Testing Loss: %4f Best Testing Loss: %4f' % \
                     (epoch+1, params.epochs, epoch_mean_train_loss, epoch_mean_test_loss, best_test_loss)
        print(output_str)
        logging.info(output_str)
        writer.add_scalar('Loss/train', epoch_mean_train_loss, epoch)
        writer.add_scalar('Loss/test', epoch_mean_test_loss, epoch)
