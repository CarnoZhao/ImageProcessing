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
import torch
import torchvision
from itertools import product
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_data(series, ratio = 0.9, batch_size = 64):
    print('loading file ...')
    matpath = '/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/cut_slice/'
    if series not in ('1', '2', '1c'):
        raise IOError('Please check data series to be in (1, 2, 1c)')
    series = 'data' + series + '.mat'
    matfiles = [filename for filename in os.listdir(matpath) if series in filename]
    dataset = MyDataset(matfiles, matpath)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    trainIndices = indices[:int(round(ratio * len(dataset)))]
    testIndices = indices[int(round(ratio * len(dataset))):]
    trainSampler = torch.utils.data.sampler.SubsetRandomSampler(trainIndices)
    testSampler = torch.utils.data.sampler.SubsetRandomSampler(testIndices)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size = batch_size , sampler = trainSampler)
    testLoader = torch.utils.data.DataLoader(dataset, shuffle = False, sampler = testSampler)
    return trainLoader, testLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, matfiles, matpath):
        super(MyDataset).__init__()
        self.matpath = matpath
        self.matfiles = matfiles
        self.numDup = sio.loadmat(self.matpath + self.matfiles[0])['data'].shape[0]
        self.length = len(self.matfiles) * self.numDup

    def __getitem__(self, idx):
        fileidx = idx // self.numDup
        layeridx = idx % self.numDup
        x = sio.loadmat(self.matpath + self.matfiles[fileidx])['data'][layeridx, :, :, :]
        x = np.concatenate((x, np.zeros((1, *x.shape[-2:]))), axis = 0)
        y = 1 if self.matfiles[fileidx].startswith('1') else 0
        return (x, y)

    def __len__(self):
        return self.length

def accuracy_cost(loader, net, deivce):
    correct = 0
    total = 0
    loss = torch.nn.CrossEntropyLoss()
    costs = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(deivce, dtype = torch.float)
            y = y.to(deivce)
            yhat = net(x)
            correct += int(torch.sum(torch.argmax(yhat, axis = 1) == y))
            total += len(y)
            cost = float(loss(yhat, y))
            costs += cost
    return correct / total, costs
            

def dense_net_model(model, loader, lr, numIterations, decay, device):
    if model == '121':
        net = torchvision.models.densenet.densenet121(num_classes = 2)
    elif model == '161':
        net = torchvision.models.densenet.densenet161(num_classes = 2)
    elif model == '169':
        net = torchvision.models.densenet.densenet169(num_classes = 2)
    elif model == '201':
        net = torchvision.models.densenet.densenet201(num_classes = 2)
    net.to('cuda')
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr = lr)
    # print('start iterating ...')
    for iteration in range(numIterations):
        if decay and iteration in (30, 60):
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
        # if iteration % 10 == 0:
        #     print("Cost after iteration %d: %.3f" % (iteration, costs))
    return net

def main(series, K = 10, ratio = 0.9, device = 'cuda'):
    lrs = [0.1, 0.01, 0.001]
    numIters = [100, 200, 300, 400, 500]
    decays = [True, False]
    batch_sizes = [32, 64, 128, 256]
    models = ['121', '161', '169', '201']
    prods = product(lrs, numIters, decays, batch_sizes, models)
    for (lr, numIterations, decay, batch_size, model) in prods:
        print('starting using lr = %.3f, numiter = %d, decay = %s, batch_size = %d, model = %s' %(lr, numIterations, str(decay), batch_size, model))
        trainCosts = 0
        testCosts = 0
        trainAccu = 0
        testAccu = 0
        for k in range(K):
            trainLoader, testLoader = load_data(series, ratio = ratio, batch_size = batch_size)
            net = dense_net_model(model, trainLoader, lr, numIterations, decay, device)
            trainAccuracy, trainCost = accuracy_cost(trainLoader, net, device)
            # print("Accuracy in train: %.6f" % trainAccuracy)
            # print("Cost in train: %.6f" % trainCost)
            testAccuracy, testCost = accuracy_cost(testLoader, net, device)
            # print("Accuracy in test: %.6f" % testAccuracy)
            # print("Cost in test: %.6f" % testCost)
            trainCosts += trainCost
            testCosts += testCost
            trainAccu += trainAccuracy
            testAccu += testAccuracy
        print("Train: accu = %.6f, cost = %.6f; Test: accu = %.6f, cost = %.6f" % (trainAccu / K, trainCosts / K, testAccu / K, testCosts / K))


main('1')
