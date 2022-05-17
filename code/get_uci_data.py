#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time       : 2021/9/15 9:20
@Author     : YangJinsheng
@Descripion :
@File       : get_dataset.py
@Software   : PyCharm
"""
import copy

import pandas as pd
import numpy as np

class GetUCI():

    def __init__(self):

        self.data_path = '/home/yangjinsheng/UCI_data/'
        self.air_data = pd.read_excel(self.data_path + 'AirQualityUCI.xlsx', engine='openpyxl')[:9357]
        self.lens = 24
        self.steps = 24

        # columns=['CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 'C6H6(GT)',
        #          'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)',
        #          'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH',]

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

    def split_data(self, data):

        lens = 1

        while True:
            dataset, last_index = self.set_dataset(data, self.lens*lens, self.steps*lens)
            validity_list, percent_true = self.validity(dataset)
            data_valid = dataset[validity_list]
            mask = data_valid.reshape([-1, dataset.shape[-1]])[:, 1]
            impute_num = mask.shape[0] - np.sum(mask)
            # if impute_num <= data.shape[0] * 1e-3:
            if impute_num <= 0:
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

    def read_data(self, columns):

        df_columns = self.air_data[columns].values.astype('float64')
        df_columns[df_columns == -200] = np.nan
        mask = self.mask_matrix(df_columns)
        delet_data = copy.deepcopy(df_columns)
        delet_data = delet_data[~np.isnan(delet_data)]
        mean = np.mean(delet_data)
        std = np.std(delet_data)
        df_columns = np.nan_to_num(df_columns)
        data = np.concatenate([df_columns[:, np.newaxis], mask[:, np.newaxis]], axis=-1)

        return data, mean, std
