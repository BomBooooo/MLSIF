import torch 
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
import pandas as pd

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_next_to_impute(data, mask, seq_len):
    target_data = data.clone()
    data_masked = data * torch.Tensor(mask[:, :, :]).cuda()
    seq_length = mask.shape[0]
    bs = data.shape[1]

    num_obs_per_t = mask.sum(1)
    next_list = np.argwhere(num_obs_per_t == np.amin(num_obs_per_t))[:,0].tolist()
    obs_list = list(set([i for i in range(seq_length)]) - set(next_list))
    obs_mask = mask[obs_list, :]
    obs_data = torch.cat([data_masked[obs_list, :, :], torch.Tensor(obs_mask).cuda()], -1)
    next_mask = mask[next_list, :]
    next_data = torch.cat([data_masked[next_list, :, :], torch.Tensor(next_mask).cuda()], -1)
    target_data = target_data[next_list, :, :]
    mask[next_list] = np.ones_like(mask[next_list])

    next_list = torch.Tensor(next_list).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
    obs_list = torch.Tensor(obs_list).unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
    obs_data = obs_data.transpose(0,1)
    next_data = next_data.transpose(0, 1)
    target_data = target_data.transpose(0, 1)

    min_dist_to_obs = np.zeros(seq_len)
    if obs_list.shape[1] == 0:
        return 0, [], 0, 0, 0, 0, 0
    for i in range(seq_len):
        if i not in obs_list:
            min_dist = np.abs((np.array(obs_list) - i)).min()
            min_dist_to_obs[i] = min_dist
    min_dist_to_obs = torch.Tensor(min_dist_to_obs)
    gap = torch.masked_select(min_dist_to_obs, min_dist_to_obs.ge(1)).unsqueeze(0).unsqueeze(-1)

    return obs_data, obs_list, next_data, next_list, target_data, mask, gap

def func(x):   # 开三次根号
    if x < 0:
        return -(pow(abs(x),1/3))
    else:
        return pow(x,1/3)

def tensor_skew(tensor, mean, std):
    n = tensor.shape[0]
    skew = torch.sum(((tensor - mean) / std).pow(3)) / n
    return skew

def tensor_kurt(tensor, mean, std):
    n = tensor.shape[0]
    kurt = torch.sum(((tensor - mean) / std).pow(4)) / n
    return kurt

def run_epoch(train, model, exp_data, alpha, clip, optimizer, batch_size=1):
    if train:
        model.train()
    else:
        model.eval()
    losses = []
    inds = np.random.permutation(exp_data.shape[0])     # 打乱
    i = 0
    d_data = exp_data[:, :, 0:1].shape[-1]
    len_data = exp_data.shape[1]

    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    while i + batch_size <= exp_data.shape[0]:

        ind = torch.from_numpy(inds[i:i + batch_size]).long()
        i += batch_size
        data = exp_data[ind]
        ori = torch.masked_select(data[0, :, 0], data[0, :, 1].bool())
        # score
        mean_ori = Variable(torch.mean(ori))
        std_ori = Variable(torch.std(ori))
        skew_ori = Variable(func(tensor_skew(ori, mean_ori, std_ori)))
        kurt_ori = Variable((tensor_kurt(ori, mean_ori, std_ori)).pow(1 / 4))

        data = data.cuda()
        exp_mask = data[:, :, 1:2]
        data = data[:, :, 0:1]

        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.transpose(0, 1))
        exp_mask = Variable(exp_mask.transpose(0, 1))
        exp_mask = exp_mask.cpu().numpy()

        if np.sum(exp_mask) / data.shape[0] < 0.2:
            continue

        mask_index = np.where(np.reshape(exp_mask, [-1, ]))[0]
        missing_idx = np.random.choice(mask_index, int(len(mask_index) * 0.8), replace=False)
        mask_drop = np.ones(batch_size * d_data * len_data)
        mask_drop[missing_idx] = 0
        mask_drop = mask_drop.reshape(len_data, batch_size, d_data)
        mask = exp_mask * mask_drop
        batch_loss = Variable(torch.tensor(0.0), requires_grad=True).cuda()

        if mask.sum() != mask.shape[0] * mask.shape[1]:
            obs_data, obs_list, next_data, next_list, target_data, mask, gap = get_next_to_impute(data, mask, len_data)
            impute_list = next_list[0, :, 0].long()
            impute_mask = torch.Tensor(exp_mask)[impute_list].float().transpose(0, 1).cuda()
            if len(obs_list) == 0:
                continue
            prediction = model(obs_data, obs_list, next_data, next_list, gap)

            # drop loss
            ori_drop_loss = torch.mean(((prediction - target_data) * impute_mask).pow(2))
            # statistic loss
            regression = torch.cat([obs_data[:, :, 0:1], prediction], axis=1).squeeze(0).squeeze(-1)
            mean_regression = torch.mean(regression)
            std_regression = torch.std(regression)
            skew_regression = func(tensor_skew(regression, mean_regression, std_regression))
            kurt_regression = (tensor_kurt(regression, mean_regression, std_regression)).pow(1 / 4)

            if np.isnan(mean_ori.cpu()) or \
                    np.isnan(std_ori.cpu()) or \
                    np.isnan(skew_ori.cpu()) or \
                    np.isnan(kurt_ori.cpu()):
                continue

            mean_change = torch.square(mean_ori - mean_regression)
            std_change = torch.square(std_ori - std_regression)
            skew_change = torch.square(skew_ori - skew_regression)
            kurt_change = torch.square(kurt_ori - kurt_regression)

            score = mean_change + std_change + skew_change + kurt_change
            # total loss
            level_loss = (1 - alpha) * ori_drop_loss + alpha * score
            batch_loss += level_loss
            losses.append(batch_loss.data.cpu().numpy())
            if train:
                optimizer.zero_grad()
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

    return np.mean(losses)

def visualization(model, exp_data, last_index, name, time, folder_name, epoch,
                  ori, batch_size=64):
    model.eval()
    data_dic = {}
    i = 0
    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    mean_change = 0
    std_change = 0
    skew_change = 0
    kurt_change = 0
    len_data = exp_data.shape[1]
    dataset_dim = exp_data.shape[2]
    while i + batch_size <= exp_data.shape[0]:
        data = exp_data[i]

        data_ori = torch.masked_select(data[:, 0], data[:, 1].bool())
        mean_ori = Variable(torch.mean(data_ori))
        std_ori = Variable(torch.std(data_ori))
        skew_ori = Variable(func(tensor_skew(data_ori, mean_ori, std_ori)))
        kurt_ori = Variable((tensor_kurt(data_ori, mean_ori, std_ori)).pow(1 / 4))

        data = data.cuda().unsqueeze(0)
        exp_mask = data[:, :, 1:]
        data = data[:, :, 0:1]
        if torch.sum(exp_mask) <= 2:
            data_dic[i] = data.squeeze().squeeze()
            i += batch_size
            continue
        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.transpose(0, 1))
        mask = Variable(exp_mask.transpose(0, 1))
        mask = mask.cpu().numpy()

        if mask.sum() < mask.shape[0] * mask.shape[1]:
            regression = torch.ones(len_data).cuda()
            obs_data, obs_list, next_data, next_list, target_data, mask, gap = \
                get_next_to_impute(data, mask, exp_data.shape[1])

            prediction = model(obs_data, obs_list, next_data, next_list, gap)

            regression[obs_list[0, :, 0].long()] = obs_data[0, :, 0]
            regression[next_list[0, :, 0].long()] = prediction[0, :, 0]
            regression = regression.detach()

            exp_data[i, :, 0] = regression

            mean_regression = torch.mean(regression)
            std_regression = torch.std(regression)
            skew_regression = func(tensor_skew(regression, mean_regression, std_regression))
            kurt_regression = (tensor_kurt(regression, mean_regression, std_regression)).pow(1 / 4)

            mean_change += torch.square(mean_ori - mean_regression)
            std_change += torch.square(std_ori - std_regression)
            skew_change += torch.square(skew_ori - skew_regression)
            kurt_change += torch.square(kurt_ori - kurt_regression)

        i += batch_size

    outer = exp_data[-1, :, :]
    exp_data = exp_data[:-1, :, :].reshape([-1, dataset_dim])
    exp_data = torch.cat([exp_data, outer[-last_index:, :]], axis=0).cpu().numpy()
    prediction = exp_data[:, 0]
    exp_mask = exp_data[:, 1]

    scores = mean_change + std_change + skew_change + kurt_change

    plot_sensor(prediction, exp_mask, name, time, folder_name, epoch, scores, ori)

def plot_sensor(prediction, mask, name, time, folder_name, epoch, scores, ori):
    ori_data, ori_mask = ori
    if len(ori_data) == 0:
        pass
    else:
        compare_data = np.argwhere(ori_mask > mask)
        se = np.square((prediction - ori_data)[compare_data[:, 0]])
        mse = np.mean(se)
    fig = plt.figure(figsize=(30, 9))
    x = range(prediction.shape[0])
    mask_index = torch.where(1-torch.Tensor(mask))
    real_color = 'blue'
    fake_color = 'red'
    # 数据处理
    real_color_list = [real_color, ] * (prediction.shape[0])
    for j in mask_index[0]:
        real_color_list[j] = fake_color
    # 画图
    plt.scatter(x, prediction, c=real_color_list, s=4)

    path = './results/%s/%s_%s' % (name, time, folder_name)
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)
    if len(ori_data) == 0:
        plt.title('scores_%4f'%scores)
        fig.savefig(path + "/best_results.png")
    else:
        plt.title('scores_%4f_mse_%4f' % (scores, mse))
        fig.savefig(path + "/best_results.png")
    np.save(path + '/best_results.npy', prediction)
    plt.close()
    # print('\n[*] Plot Finished')

