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
    matfiles = [filename for filename in os.listdir(matpath) if series in filename]
    X = np.concatenate([sio.loadmat(matfile)['data'] for matfile in matfiles], axis = 3)
    Y = [1 if matfile.startswith('1') else 0 for matfile in matfiles]
    Y = np.repeat(Y, X.shape[-1] // len(Y))
    length = len(Y)
    shuffleIdx = list(range(length))
    np.random.shuffle(shuffleIdx)
    trainIdx = shuffleIdx[:int(round(ratio * length))]
    testIdx = shuffleIdx[int(round(ratio * length)):]
    trainX = X[:, :, :, trainIdx]
    trainY = Y[trainIdx]
    testX = X[:, :, :, testIdx]
    testY = Y[testIdx]
    trainDataset = MyDataset(trainX, trainY)
    testDataset = MyDataset(testX, testY)
    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size = 64, shuffle = True)
    testLoader = torch.utils.data.DataLoader(testDataset, shuffle = False)
    return trainLoader, testLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super(MyDataset).__init__()
        self.length = X.shape[0]
        self.data = [(X[:, :, :, i], Y[i]) for i in range(self.length)]

    def __getitem__(self, idx):
        return self.data[idx]

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