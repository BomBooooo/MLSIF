#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/7/2 13:16
@Author     : YangJinsheng
@Descripion : 
@File       : get_dataset.py
@Software   : PyCharm
"""
import pandas as pd
import numpy as np
import os

class Preprocess():
    def __init__(self, data_path='/home/yangjinsheng/qyf_impution/gphxj/gphxj_ori/',
                 data_name = 'gphxj_ori_split/',
                 root_path = '/home/yangjinsheng/tcn/NRTIS-sensor/',
                 node_name='gphxj_GPS-011_264_gps-total-x.csv',
                 freq = '2H', lens=24, steps=24):
        # 数据路径
        self.data_path = data_path
        self.data_name = data_name
        self.root_path = root_path
        self.tenant_code = self.data_name.split('_')[0]
        self.node_name = node_name
        self.monitor_item = self.node_name.split('_')[-1][:-4]
        self.node_id = self.node_name.split('_')[2]
        self.freq = freq

        # 样本长度和步长
        self.lens = lens
        self.steps = steps

        # 定义路径
        self.save_path = './results/'
        exists = os.path.exists(self.save_path)
        if not exists:
            os.makedirs(self.save_path)

    def data_pre(self, data):
        data = data['monitor_value'].values
        data = data - data[0]
        delet_data = data.copy()
        delet_data = delet_data[~np.isnan(delet_data)]
        mean = np.mean(delet_data)
        std = np.std(delet_data)
        data[np.where(data >  3*std + mean)] = np.nan
        data[np.where(data < -3*std + mean)] = np.nan
        delet_data = data.copy()
        delet_data = delet_data[~np.isnan(delet_data)]
        mean = np.mean(delet_data)
        std = np.std(delet_data)
        return data, mean, std

    def mask_matrix(self, x):
        mask = np.isnan(x)
        mask = 1 - mask
        mask = np.array(mask, dtype='float32')
        return mask

    def set_dataset(self, data, lens, steps):    # 将数据按照长度lens步长steps划分
        for index in range(0, data.shape[0], steps):
            if index == 0:
                dataset = data[np.newaxis, index: index+lens, :]
            else:
                if (index+lens) < data.shape[0]:
                    dataset = np.concatenate([dataset, data[np.newaxis, index: index+lens, :]], axis=0)
                elif index < data.shape[0]:
                    dataset = np.concatenate([dataset, data[np.newaxis, -lens:, :]], axis=0)
                    last_index = data.shape[0] - index
                    return dataset, last_index
        return dataset, False

    def read_data(self):
        df = pd.read_csv(self.data_path + self.data_name + self.node_name)
        if self.freq:
            df = df[df['freq'] == self.freq]
        if len(df) == 0:
            print(self.node_name, 'do not have freq %s data!' % self.freq)
            return None, None
        x, mean, std = self.data_pre(df)
        m = self.mask_matrix(x)
        x = np.nan_to_num(x)
        data = np.concatenate([x[:, np.newaxis], m[:, np.newaxis]], axis=-1)
        name = '{tenant_code}_{node_id}_{monitor_item}_{shape}_{freq}'.format(
            tenant_code=self.tenant_code, node_id=self.node_id, monitor_item=self.monitor_item,
            shape=data.shape,freq=self.freq)
        return data, mean, std, name


    def split_data(self, data):

        lens = 1

        while True:
            dataset, last_index = self.set_dataset(data, self.lens*lens, self.steps*lens)
            validity_list, percent_true = self.validity(dataset)
            data_valid = dataset[validity_list]
            mask = data_valid.reshape([-1, dataset.shape[-1]])[:, 1]
            impute_num = mask.shape[0] - np.sum(mask)
            if impute_num <= data.shape[0] * 1e-3:
                lens += 1
                continue
            else:
                break

        print('[*] Data Split Finish!')

        return dataset, validity_list, percent_true, impute_num, last_index

    def validity(self, dataset):

        validity_list = []
        for i in range(dataset.shape[0]):
            mask = dataset[i, :, 1].reshape(-1)
            obs_list = np.where(mask)[0].tolist()
            if len(obs_list) == 0:
                continue
            if np.sum(mask) / mask.shape[0] >= 0.9:
                validity_list.append(i)
        percent_true = len(validity_list) / dataset.shape[0]

        return validity_list, percent_true

if __name__ == '__main__':
    data_pre = Preprocess()
    data, mean, std, name = data_pre.read_data()
    dataset, validity_list, percent_true, impute_num, last_index = data_pre.split_data(data, previous_lens=0, percent_pre=0)
    print(data.shape)
    print(dataset.shape)
    print(name)
    print(last_index)
    print(percent_true, impute_num)
