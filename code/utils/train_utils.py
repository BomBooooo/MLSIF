# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch.autograd import Variable
from torch import nn

import os

def func(x):   # 开三次根号
    if x < 0:
        return -(pow(abs(x),1/3))
    else:
        return pow(x,1/3)

def tensor_skew(tensor, mean, std):
    n = tensor.shape[0]
    skew =torch.sum(((tensor - mean) / std).pow(3)) / n
    return skew

def tensor_kurt(tensor, mean, std):
    n = tensor.shape[0]
    kurt = torch.sum(((tensor - mean) / std).pow(4)) / n
    return kurt

def plot_ori(data):

    prediction = data[:, 0]
    mask = data[:, 1]

    fig = plt.figure(figsize=(30, 9))
    x = range(prediction.shape[0])
    # mask_index = torch.where(1-torch.Tensor(mask))

    real_color = 'blue'
    real_color_list = [real_color, ] * (prediction.shape[0])

    # for j in mask_index[0]:
    #     real_color_list[j] = fake_color
    # 画图
    plt.scatter(x, prediction, c=real_color_list, s=4)
    plt.show()

def plot_result(data, name, time, score_total, folder_name, ori_data=[], ori_mask=[]):

    prediction = data[:, 0]
    mask = data[:, 2]

    fake_color = 'red'
    real_color = 'blue'
    path = './results_uci/%s/%s_%s' % (name, time, folder_name)
    exists = os.path.exists(path)
    if not exists:
        os.makedirs(path)

    x = range(prediction.shape[0])

    if len(ori_data) != 0:
        compare_data = np.argwhere(ori_mask > mask)
        mse = np.mean(np.square((prediction - ori_data)[compare_data[:, 0]]))
        ori_index = np.argwhere(ori_mask > 0)
        ori = torch.Tensor(ori_data[ori_index[:, 0]])
        # score
        mean_ori = torch.mean(ori)
        std_ori = torch.std(ori)
        skew_ori = func(tensor_skew(ori, mean_ori, std_ori))
        kurt_ori = (tensor_kurt(ori, mean_ori, std_ori)).pow(1 / 4)

        regression = torch.Tensor(prediction)
        mean_regression = torch.mean(regression)
        std_regression = torch.std(regression)
        skew_regression = func(tensor_skew(regression, mean_regression, std_regression))
        kurt_regression = (tensor_kurt(regression, mean_regression, std_regression)).pow(1 / 4)

        mean_change = torch.square(mean_ori - mean_regression)
        std_change = torch.square(std_ori - std_regression)
        skew_change = torch.square(skew_ori - skew_regression)
        kurt_change = torch.square(kurt_ori - kurt_regression)
        # statistic loss
        score = mean_change + std_change + skew_change + kurt_change
        score = score.numpy()

        fig_ori = plt.figure(figsize=(30, 9))
        ori_mask_index = torch.where(1 - torch.Tensor(ori_mask))
        real_color_list = [real_color, ] * (prediction.shape[0])
        nan_color = 'white'
        for j in ori_mask_index[0]:
            real_color_list[j] = nan_color
        # 画图
        plt.scatter(x, ori_data, c=real_color_list, s=4)
        fig_ori.savefig(path + "/ori_data.png")
        plt.close()

    fig = plt.figure(figsize=(30, 9))
    mask_index = torch.where(1-torch.Tensor(mask))
    real_color_list = [real_color, ] * (prediction.shape[0])

    for j in mask_index[0]:
        real_color_list[j] = fake_color
    # 画图
    plt.scatter(x, prediction, c=real_color_list, s=4)
    if len(ori_data) == 0:
        plt.title('score_total_%4f' % (score_total))
        fig.savefig(path + "/results_score_total_%4f.png" % (score_total))
    else:
        plt.title('score_total_%4f_score_%4f_mse_%4f' % (score_total, score, mse))
        fig.savefig(path + "/results_score_total_%4f_score_%4f_mse_%4f.png" % (score_total, score, mse))
    np.save(path + "/impute_results.npy", data)
    plt.close()

def get_next_to_impute(train, data, mask_drop, exp_mask, seq_len):
    target_data = data.clone()
    seq_length = exp_mask.shape[0]
    bs = data.shape[1]
    if train:
        mask = mask_drop * exp_mask
        data_masked = data * mask
    else:
        data_masked = data * exp_mask
        mask = exp_mask

    num_obs_per_t = np.array(mask.sum(1).cpu())
    next_list = np.argwhere(num_obs_per_t == np.amin(num_obs_per_t))[:,0].tolist()
    obs_list = list(set([i for i in range(seq_length)]) - set(next_list))
    obs_list = torch.Tensor(obs_list).long().cuda()
    obs_mask = mask[obs_list, :]
    obs_data = torch.cat([data_masked[obs_list, :, :], obs_mask], -1)
    next_list = torch.Tensor(next_list).long().cuda()
    next_mask = mask[next_list, :]
    next_data = torch.cat([data_masked[next_list, :, :], next_mask], -1)
    target_data = target_data[next_list, :, :]
    mask[next_list] = torch.ones_like(mask[next_list])

    next_list = next_list.unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
    obs_list = obs_list.unsqueeze(0).unsqueeze(-1).repeat(bs,1,1)
    obs_data = obs_data.transpose(0,1)
    next_data = next_data.transpose(0, 1)
    target_data = target_data.transpose(0, 1)

    min_dist_to_obs = torch.zeros(seq_len)
    if obs_list.shape[1] == 0:
        return 0, [], 0, 0, 0, 0, 0
    for i in range(seq_len):
        if i not in obs_list:
            min_dist = abs((obs_list - i)).min()
            min_dist_to_obs[i] = min_dist
    gap = torch.masked_select(min_dist_to_obs, min_dist_to_obs.ge(1)).unsqueeze(0).unsqueeze(-1)

    return obs_data, obs_list, next_data, next_list, target_data, gap

def run_epoch(model, exp_data, sita, alpha, clip, optimizer=None, batch_size=64):

    model.train()
    losses = []
    drop_losses = []
    inds = torch.randperm(exp_data.shape[0]).cuda()      # 打乱
    i = 0
    d_data = exp_data[:, :, 0:1].shape[-1]
    len_data = exp_data.shape[1]

    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    while i + batch_size <= exp_data.shape[0]:
        
        ind = inds[i:i+batch_size].long()
        i += batch_size
        data = Variable(exp_data[ind]).cuda()
        ori = torch.masked_select(data[0, :, 0], data[0, :, 1].bool())
        # score
        mean_ori = Variable(torch.mean(ori))
        std_ori = Variable(torch.std(ori))
        skew_ori = Variable(func(tensor_skew(ori, mean_ori, std_ori)))
        kurt_ori = Variable((tensor_kurt(ori, mean_ori, std_ori)).pow(1 / 4))

        # data = data.cuda()
        ori_mask = data[:, :, 2:3]
        exp_mask = data[:, :, 1:2]
        data = data[:, :, 0:1]

        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.transpose(0, 1))
        exp_mask = Variable(exp_mask.transpose(0, 1))
        ori_mask = Variable(ori_mask.transpose(0, 1))

        mask_index = torch.where(torch.reshape(exp_mask, [-1, ]))[0]
        missing_idx = torch.Tensor(np.random.choice(mask_index.cpu(), int(len(mask_index)*0.5),
                                                    replace=False)).long().cuda()
        mask_drop = torch.ones(batch_size*d_data*len_data).cuda()
        mask_drop[missing_idx] = 0
        mask_drop = mask_drop.reshape(len_data, batch_size, d_data)
        batch_loss = Variable(torch.tensor(0.0), requires_grad = True).cuda()

        regression = torch.ones(len_data).cuda()
        obs_data, obs_list, next_data, next_list, target_data, gap = \
            get_next_to_impute(True, data, mask_drop, exp_mask, len_data)
        impute_list = next_list[0, :, 0].long()
        impute_mask = (exp_mask[impute_list] > ori_mask[impute_list]).float().transpose(0, 1)
        prediction = model(obs_data, obs_list, next_data, next_list, gap)

        # drop loss
        ori_drop_loss = torch.mean(
            ((prediction - target_data) * (1 - impute_mask)).pow(2))
        impute_loss = torch.mean(
            ((prediction - target_data) * impute_mask).pow(2))
        drop_loss = sita * impute_loss + (1 - sita) * ori_drop_loss

        regression[obs_list[0, :, 0].long()] = obs_data[0, :, 0]
        regression[next_list[0, :, 0].long()] = prediction[0, :, 0]
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
        # statistic loss
        score = mean_change + std_change + skew_change + kurt_change
        # total loss
        level_loss = (1 - alpha) * drop_loss + alpha * score
        batch_loss += level_loss
        losses.append(batch_loss.data.detach().cpu().numpy())
        drop_losses.append(drop_loss.detach().cpu().numpy())

        # for parameters in model.parameters():
        #     print(parameters)
        #     break
        optimizer.zero_grad()
        batch_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

    return np.mean(drop_losses), np.mean(losses)

def visualization(plot, level, model, dataset, exp_data, validity_list, name, time,
                  folder_name, loss, lens, last_index, batch_size=1):
    model.eval()
    i = 0
    dataset_dim = dataset.shape[-1]
    if exp_data.shape[0] < batch_size:
        batch_size = exp_data.shape[0]
    mean_change = 0
    std_change = 0
    skew_change = 0
    kurt_change = 0
    len_data = exp_data.shape[1]
    exp_data = exp_data.cuda()

    level_mask = torch.zeros(dataset.shape[0]*dataset.shape[1]).cuda()
    while i + batch_size <= exp_data.shape[0]:
        data = exp_data[i:i+1]
        # score
        ori = torch.masked_select(data[0, :, 0], data[0, :, 1].bool())
        mean_ori = Variable(torch.mean(ori))
        std_ori = Variable(torch.std(ori))
        skew_ori = Variable(func(tensor_skew(ori, mean_ori, std_ori)))
        kurt_ori = Variable((tensor_kurt(ori, mean_ori, std_ori)).pow(1 / 4))

        # ori_mask = data[:, :, 2:3]
        exp_mask = data[:, :, 1:2]
        data = data[:, :, 0:1]
        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.transpose(0, 1))
        exp_mask = Variable(exp_mask.transpose(0, 1))

        if exp_mask.sum() != exp_mask.shape[0] * exp_mask.shape[1]:
            regression = torch.ones(len_data).cuda()
            dataset_index = validity_list[i].long().cpu()
            level_mask[dataset_index*exp_data.shape[1]:(dataset_index+1)*exp_data.shape[1]] += 1 - exp_mask[:, 0, 0]
            obs_data, obs_list, next_data, next_list, target_data, gap = \
                get_next_to_impute(False, data, None, exp_mask, exp_data.shape[1])
            prediction = model(obs_data, obs_list, next_data, next_list, gap)

            regression[obs_list[0, :, 0].long()] = obs_data[0, :, 0]
            regression[next_list[0, :, 0].long()] = prediction[0, :, 0]
            regression = regression.detach()

            dataset[dataset_index, :, 0] = regression
            dataset[dataset_index, :, 1] = torch.ones(len_data).cuda()
            # dataset[dataset_index, :, 2] = ori_mask[0, :, 0]

            mean_regression = torch.mean(regression)
            std_regression = torch.std(regression)
            skew_regression = func(tensor_skew(regression, mean_regression, std_regression))
            kurt_regression = (tensor_kurt(regression, mean_regression, std_regression)).pow(1 / 4)

            mean_change += torch.square(mean_ori - mean_regression)
            std_change += torch.square(std_ori - std_regression)
            skew_change += torch.square(skew_ori - skew_regression)
            kurt_change += torch.square(kurt_ori - kurt_regression)

        i += batch_size

    outer = dataset[-1, :, :]
    dataset = dataset[:-1, :, :].reshape([-1, dataset_dim])
    dataset = torch.cat([dataset, outer[-last_index:, :]], axis=0)

    level_mask_outer = level_mask[-lens:, ]
    level_mask = level_mask[:-lens, ]
    level_mask = torch.cat([level_mask, level_mask_outer[-last_index:, ]], axis=0)

    score = mean_change+std_change+skew_change+kurt_change

    root_path = plot_sensor(plot, level, dataset, level_mask, name, time, folder_name, loss, lens,
                            score)

    return dataset, mean_change, std_change, skew_change, kurt_change, root_path

def plot_sensor(plot, level, data, level_mask, name, time, folder_name, loss, lens,
                score):
    root_path = './results_uci/%s/%s_%s' % (name, time, folder_name)
    exists = os.path.exists(root_path)
    if not exists:
        os.makedirs(root_path)
    path = './results_uci/%s/%s_%s/level_%s lens_%s' % (name, time, folder_name, level, lens)
    if plot == 1:
        prediction = data[:, 0]
        mask = data[:, 1]
        mask_ori = data[:, 2]

        fig = plt.figure(figsize=(30, 9))
        x = range(prediction.shape[0])
        mask_index = torch.where(1-mask)
        level_mask_index = torch.where(level_mask)
        ori_index = torch.where(1-mask_ori)

        real_color = 'blue'
        fake_color = 'red'
        nan_color = 'white'
        impute_color = 'green'
        # 数据处理
        real_color_list = [real_color, ] * (prediction.shape[0])

        for j in ori_index[0]:
            if j in mask_index[0]:
                real_color_list[j] = nan_color
            elif j in level_mask_index[0]:
                real_color_list[j] = fake_color
            else:
                real_color_list[j] = impute_color
        # 画图
        plt.scatter(x, prediction.detach().cpu().numpy(), c=real_color_list, s=4)
        plt.title('score_%4f_loss_%4f' % (score, loss))

        fig.savefig(path + " results.png")

        plt.close()
    # print('\n[*] Plot Finished')
    return path

