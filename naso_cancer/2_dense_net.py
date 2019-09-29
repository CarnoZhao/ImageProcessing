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

def load_data(series, ratio = 0.85):
    print('loading file ...')
    matpath = '/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/cut_slice/'
    if series not in ('1', '2', '1c'):
        raise IOError('Please check data series to be in (1, 2, 1c)')
    series = 'data' + series + '.mat'
    matfiles = [matpath + filename for filename in os.listdir(matpath) if series in filename]
    dataset = MyDataset(matfiles)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    trainIndices = indices[:int(round(ratio * len(dataset)))]
    testIndices = indices[int(round(ratio * len(dataset))):]
    trainSampler = torch.utils.data.sampler.SubsetRandomSampler(trainIndices)
    testSampler = torch.utils.data.sampler.SubsetRandomSampler(testIndices)
    trainLoader = torch.utils.data.DataLoader(dataset, batch_size = 64, sampler = trainSampler)
    testLoader = torch.utils.data.DataLoader(dataset, shuffle = False, sampler = testSampler)
    return trainLoader, testLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, matfiles):
        super(MyDataset).__init__()
        self.matfiles = matfiles
        self.numDup = sio.loadmat(self.matfiles[0])['data'].shape[0]
        self.length = len(self.matfiles) * self.numDup

    def __getitem__(self, idx):
        fileidx = idx // self.numDup
        layeridx = idx % self.numDup
        x = sio.loadmat(self.matfiles[fileidx])['data'][layeridx, :, :, :]
        y = 1 if self.matfiles[fileidx].startswith('1') else 0
        return (x, y)

    def __len__(self):
        return self.length

def accuracy(loader, net, deivce):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(deivce, dtype = torch.float)
            y = y.to(deivce)
            yhat = net(x)
            correct += int(torch.sum(torch.argmax(yhat, axis = 1) == y))
            total += len(y)
    return correct / total
            

def dense_net_model(loader, lr, numIterations, decay, device):
    net = torchvision.models.densenet.DenseNet(num_classes = 2)
    net.to('cuda')
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr = lr)
    print('start iterating ...')
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
        print("Cost after iteration %d: %.3f" % (iteration, costs))
    return net

def main(series, lr = 0.1, numIterations = 300, decay = False, device = 'cuda'):
    print('starting using lr = %.3f, numiter = %d' %(lr, numIterations))
    trainLoader, testLoader = load_data(series)
    net = dense_net_model(trainLoader, lr, numIterations, decay, device)
    trainAccuracy = accuracy(trainLoader, net, device)
    print("Accuracy in train: %.3f" % trainAccuracy)
    testAccuracy = accuracy(testLoader, net, device)
    print("Accuracy in test: %.3f" % testAccuracy)

main('1', lr = 0.1, numIterations = 90, decay = True)
