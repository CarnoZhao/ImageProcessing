'''
@Author: Xun Zhao
@Date: 2019-09-26 15:28:48
@LastEditors: Xun Zhao
@LastEditTime: 2019-09-27 17:27:12
@Description: 
'''
import numpy as np
import scipy.io as sio
import os
import torch
import torchvision

def load_data(series, ratio = 0.85):
    matpath = '/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/cut_slice/'
    if series not in ('1', '2', '1c'):
        raise IOError('Please check data series to be in (1, 2, 1c)')
    matfiles = [filename for filename in os.listdir(matpath) if series in filename]
    Y = [1 if matfile.startswith('1') else 0 for matfile in matfiles]
    X = []
    for matfile in matfiles:
        matdata = sio.loadmat(matpath + matfile)
        img = matdata['img']
        roi = matdata['roi']
        X.append([img, roi])
    X = np.array(X)
    Y = np.array(Y)
    shuffleIdx = list(range(len(X)))
    np.random.shuffle(shuffleIdx)
    trainIdx = shuffleIdx[:int(round(ratio * len(X)))]
    testIdx = shuffleIdx[int(round(ratio * len(X))):]
    trainX = X[trainIdx]
    trainY = Y[trainIdx]
    testX = X[testIdx]
    testY = Y[testIdx]
    trainDataset = MyDataset(trainX, trainY)
    testDataset = MyDataset(testX, testY)
    trainLoader = torch.utils.data.DataLoader(trainDataset, shuffle = True)
    testLoader = torch.utils.data.DataLoader(testDataset, shuffle = False)
    return trainLoader, testLoader

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        super(MyDataset).__init__()
        self.length = X.shape[0]
        self.data = [(X[i], Y[i]) for i in range(self.length)]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return self.length

def accuracy(loader, net, deivce):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(deivce)
            y = y.to(deivce)
            yhat = net(x)
            yhat = 1 if yhat > 0.5 else 0
            if yhat == y:
                correct += 1
            total += 1
    return correct / total
            

def dense_net_model(loader, lr, numIterations, device):
    net = torchvision.models.densenet.DenseNet()
    net.to('cuda')
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr = lr)
    for iteration in range(numIterations):
        costs = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            yhat = net(x)
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            optimizer.step()
        if iteration % 100 == 0:
            print("Cost after iteration %d: %.3f" % (iteration, costs))
    return net

def main(lr = 0.0001, numIterations = 1000, device = 'cuda'):
    trainLoader, testLoader = load_data('1')
    net = dense_net_model(trainLoader, lr, numIterations, device)
    trainAccuracy = accuracy(trainLoader, net, device)
    print("Accuracy in train: %.3f" % trainAccuracy)
    testAccuracy = accuracy(testLoader, net, device)
    print("Accuracy in test: %.3f" % testAccuracy)

main()