import numpy as np
import cv2
import os
import sys
import torch
import torchvision
from itertools import product
# sys.path.insert(1, "/home/tongxueqing/zhao/MachineLearning/Python_ML/")
# import densenet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

'''
filename fields:
(No.)_(type)_(typeNo.)_(Hospital)_(sliceNo.).tif
'''

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, path):
        super(MyDataset).__init__()
        self.path = path
        self.filenames = filenames
        self.length = len(self.filenames)
        self.types = {"jizhi": 0,
                      "tumor": 1,
                      "tumorln": 2,
                      "huaisi": 3,}

    def __getitem__(self, i):
        x = cv2.imread(self.path + self.filenames[i])
        x = np.transpose(x, [2, 0, 1])
        y = self.types[self.filenames[i].split('_')[1]]
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
        pred = pred.log_softmax(dim = self.dim)
        with torch.no_grad():
            H = torch.zeros_like(pred)
            H.fill_(self.smoothing / self.cls)
            H.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-H * pred * self.weight, dim = self.dim))

def load_data(ratio = 0.8, batch_size = 64):
    path = "/home/tongxueqing/data/zhaox/stain_classification/cutted/"
    filenames = os.listdir(path)
    indices = list(range(len(filenames)))
    np.random.shuffle(indices)
    trainIndices = indices[:int(round(ratio * len(filenames)))]
    testIndices = indices[int(round(ratio * len(filenames))):]
    trainfiles = [filenames[i] for i in trainIndices]
    with open(fileidxpath + ".train", 'w') as f:
        f.write('\n'.join(trainfiles) + '\n')
    testfiles = [filenames[i] for i in testIndices if filenames[i]]
    with open(fileidxpath + ".test", 'w') as f:
        f.write('\n'.join(testfiles) + '\n')
    traindataset = MyDataset(trainfiles, path)
    testdataset = MyDataset(testfiles, path)
    trainLoader = torch.utils.data.DataLoader(traindataset, batch_size = batch_size)
    testLoader = torch.utils.data.DataLoader(testdataset, shuffle = False)
    return trainLoader, testLoader

def accuracy(loader, net, deivce):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(deivce, dtype = torch.float)
            y = y.numpy()
            yhat = net(x).cpu().data.numpy()
            yhat = np.exp(yhat) / np.sum(np.exp(yhat), axis = 1, keepdims = True)
            yhat = np.argmax(yhat, aixs = 0)
            correct += sum(yhat == y)
            total += len(y)
    return correct / total

def dense_net_model(model, loader, lr, numIterations, decay, device):
    K = 4
    net = torchvision.models.densenet121(num_classes = K)
    net.to('cuda')
    weight = torch.FloatTensor([0.25, 0.25, 0.25, 0.25]).to(device)
    loss = LabelSmoothingLoss(classes = K, smoothing = 0.01, weight = weight)
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

def main(lr = 0.1, numIterations = 100, ratio = 0.9, decay = True, batch_size = 64, model = '121', device = 'cuda', ifTrain = False, bins = 50):
    print('lr = %.3f\nnumiter = %d\ndecay = %s\nbatch_size = %d\nmodel = %s' % (lr, numIterations, str(decay), batch_size, model))
    trainLoader, testLoader = load_data(ratio = ratio, batch_size = batch_size)
    if os.path.exists(modelpath) and not ifTrain:
        net = torch.load(modelpath)
    else:
        net = dense_net_model(model, trainLoader, lr, numIterations, decay, device)
        torch.save(net, modelpath)
    trainAccuracy = accuracy(trainLoader, net, device)
    testAccuracy = accuracy(testLoader, net, device)
    print("Train: accu = %.6f\nTest: accu = %.6f" % (trainAccuracy, testAccuracy))

global fileidxpath
global modelpath
fileidxpath = sys.argv[1]
modelpath = sys.argv[2]
main(batch_size = 16, ifTrain = True)