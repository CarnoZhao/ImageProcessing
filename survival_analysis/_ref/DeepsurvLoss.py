import torch.nn as nn
import torch
import numpy as np

def get_unique(T):
    T = T.numpy()
    unique_num = []
    index_list = []
    for i in range(T.shape[0]):
        if T[i] not in unique_num:
            unique_num.append(T[i])
        index_list.append(unique_num.index(T[i]))
    return unique_num,index_list

def get_segment_max(value,index):
    segment_max = np.zeros([len(list(set(index)))])
    for i in range(value.shape[0]):
        if value[i]>segment_max[index[i]]:
            segment_max[index[i]] = value[i]
    return segment_max

def get_segment_sum(value,index):
    segment_sum = np.zeros([len(list(set(index)))])
    for i in range(value.shape[0]):
        segment_sum[index[i]] += value[i]
    return segment_sum

def get_cussum(yhc):
    return [torch.sum(yhc[:i]).data  for i in range(1,yhc.shape[0]+1,1)]


class SurvLoss(nn.Module):
    def __init__(self):
        super(SurvLoss, self).__init__()
        pass

    def forward(self, outs, T_E, T_T):
        E = torch.squeeze(T_E)
        Y_c = torch.squeeze(T_T)
        Y_hat_c = torch.squeeze(outs).double()

        Y_label_T = torch.abs(Y_c).double()
        E[E > 0] = 1
        Y_label_E = E.double()

        Obs = torch.sum(Y_label_E)
        Y_hat_hr = torch.exp(Y_hat_c)
        ljqh = get_cussum(Y_hat_hr)
        Y_hat_cumsum = torch.log(torch.Tensor(ljqh))
        unique_values, segment_ids = get_unique(Y_label_T)
        loss_s2_v = torch.Tensor(get_segment_max(Y_hat_cumsum, segment_ids))
        loss_s2_count = torch.Tensor(get_segment_sum(Y_label_E, segment_ids))

        loss_s2 = torch.sum(torch.mul(loss_s2_v, loss_s2_count))
        loss_s1 = torch.sum(torch.mul(Y_hat_c, Y_label_E))

        loss = torch.div(torch.sub(loss_s2, loss_s1), Obs)
        return loss
