'''
@Author: Xun Zhao
@Date: 2019-09-26 15:28:48
@LastEditors: Xun Zhao
@LastEditTime: 2019-09-27 23:45:15
@Description: 
'''
import numpy as np
import scipy.io as sio
import os
import sys
import torch
import torchvision
from itertools import product
sys.path.insert(1, "/home/tongxueqing/zhao/MachineLearning/Python_ML/")

import densenet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, matfiles, matpath):
        super(MyDataset).__init__()
        self.matpath = matpath
        self.matfiles = matfiles
        self.length = len(self.matfiles)

    def __getitem__(self, idx):
        x = sio.loadmat(self.matpath + self.matfiles[idx])['data']
        # x = np.concatenate((x, np.zeros((1, *x.shape[-2:]))), axis = 0)
        y = 1 if self.matfiles[idx].startswith('1') else 0
        return (x, y)

    def __len__(self):
        return self.length

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if weight is None:
            self.weight = torch.ones(classes) / classes
        else:
            self.weight = weight

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / self.cls)
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred * self.weight, dim=self.dim))


def load_data(series, ratio = 0.9, batch_size = 64):
    matpath = '/home/tongxueqing/zhao/ImageProcessing/naso_cancer/_data/cut_slice/'

    if series not in ('1', '2', '1c'):
        raise IOError('Please check data series to be in (1, 2, 1c)')
    series = 'data' + series + '.mat'
    matfiles = [filename for filename in os.listdir(matpath) if series in filename]
    indices = list(range(len(matfiles)))
    np.random.shuffle(indices)
    trainIndices = indices[:int(round(ratio * len(matfiles)))]
    testIndices = indices[int(round(ratio * len(matfiles))):]
    trainfiles = [matfiles[i] for i in trainIndices]
    with open(fileidxpath + ".train", 'w') as f:
        f.write('\n'.join(trainfiles) + '\n')
    testfiles = [matfiles[i] for i in testIndices if matfiles[i].endswith("0.rotate")]
    with open(fileidxpath + ".test", 'w') as f:
        f.write('\n'.join(testfiles) + '\n')
    traindataset = MyDataset(trainfiles, matpath)
    testdataset = MyDataset(testfiles, matpath)
    trainLoader = torch.utils.data.DataLoader(traindataset, batch_size = batch_size)
    testLoader = torch.utils.data.DataLoader(testdataset, shuffle = False)
    return trainLoader, testLoader

def accuracy_cost(loader, net, deivce):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(deivce, dtype = torch.float)
            y = y.numpy()
            yhat = net(x).cpu().data.numpy()
            yhat = np.exp(yhat) / np.sum(np.exp(yhat), axis = 1, keepdims = True)
            yhat = np.where(yhat[:, 0] > 0.5, 0, 1)
            correct += sum(yhat == y)
            total += len(y)
    return correct / total

def auc_roc(loader, net, device, filename, bins):
    roc = np.zeros((bins + 1, 4))
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype = torch.float)
            y = y.numpy()
            yhat = net(x).cpu().data.numpy()
            yhat = np.exp(yhat) / np.sum(np.exp(yhat), axis = 1, keepdims = True)
            for K in range(bins + 1):
                k = K / bins
                yhatK = np.where(yhat[:, 0] > k, 0, 1)
                roc[K, 0] += np.sum(np.bitwise_and(yhatK == 1, y == 1)) # tp
                roc[K, 1] += np.sum(np.bitwise_and(yhatK == 1, y == 0)) # fp
                roc[K, 2] += np.sum(np.bitwise_and(yhatK == 0, y == 0)) # tn
                roc[K, 3] += np.sum(np.bitwise_and(yhatK == 0, y == 1)) # fn
    tpr_fpr = np.zeros((bins + 1, 2))
    tpr_fpr[:, 0] = roc[:, 0] / (roc[:, 0] + roc[:, 3])
    tpr_fpr[:, 1] = roc[:, 1] / (roc[:, 1] + roc[:, 2])
    with open(filename, 'w') as f:
        for pair in tpr_fpr:
            f.write('\t'.join([str(i) for i in pair]) + '\n')
    return roc

def dense_net_model(model, loader, lr, numIterations, decay, device):
    net = densenet.densenet121(num_classes = 2)
    net.to('cuda')
    weight = torch.FloatTensor([0.84, 0.16]).to(device)
    loss = LabelSmoothingLoss(classes = 2, smoothing = 0.01, weight = weight)
    optimizer = torch.optim.Adamax(net.parameters(), lr = lr)
    print('start iterating ...')
    for iteration in range(numIterations):
        if decay and iteration % 30 == 0 and iteration != 0:
            lr /= 10
        costs = 0
        for x, y in loader:
            x = x.to(device, dtype = torch.float)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = net(x)
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            optimizer.step()
        if iteration % 10 == 0:
            print("Cost after iteration %d: %.3f" % (iteration, costs))
    return net

def main(series, lr = 0.1, numIterations = 100, ratio = 0.9, decay = True, batch_size = 64, model = '121', device = 'cuda', ifTrain = False, bins = 50):
    print('lr = %.3f\nnumiter = %d\ndecay = %s\nbatch_size = %d\nmodel = %s' %(lr, numIterations, str(decay), batch_size, model))

    trainLoader, testLoader = load_data(series, ratio = ratio, batch_size = batch_size)
    if os.path.exists(modelpath) and not ifTrain:
        net = torch.load(modelpath)
    else:
        net = dense_net_model(model, trainLoader, lr, numIterations, decay, device)
        torch.save(net, modelpath)
    trainAccuracy = accuracy_cost(trainLoader, net, device)
    testAccuracy = accuracy_cost(testLoader, net, device)
    print("Train: accu = %.6f; Test: accu = %.6f" % (trainAccuracy, testAccuracy))
    # trianRoc = auc_roc(trainLoader, net, device, rocpath + '%s.train.csv' % model, bins)
    # testRoc = auc_roc(testLoader, net, device, rocpath + '%s.test.csv' % model, bins)

global fileidxpath
global modelpath
fileidxpath = sys.argv[1]
modelpath = sys.argv[2]
main('1', lr = 0.05, batch_size = 32, numIterations = 100, ifTrain = True)
