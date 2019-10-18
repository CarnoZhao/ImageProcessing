import numpy as np
import cv2
import os
import sys
import torch
import torchvision
os.environ["CUDA_VISIBLE_'cuda'S"] = "0"

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
    def __init__(self, classes, smoothing=0.0, dim=-1, weight = None, focal = False, gamma = 2):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if weight is None:
            self.weight = torch.ones(classes) / classes
        else:
            self.weight = weight
        self.focal = focal
        self.gamma = gamma

    def forward(self, pred, target):
        sf = pred.softmax(dim = self.dim)
        pred = pred.log_softmax(dim = self.dim)
        if self.focal:
            pred = (1 - sf) ** self.gamma * pred
        with torch.no_grad():
            H = torch.zeros_like(pred)
            H.fill_(self.smoothing / self.cls)
            H.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-H * pred * self.weight, dim = self.dim))

def load_data(ratio, batch_size):
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

def accuracy(loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to("cuda", dtype = torch.float)
            y = y.numpy()
            yhat = net(x).cpu().data.numpy()
            yhat = np.exp(yhat) / np.sum(np.exp(yhat), axis = 1, keepdims = True)
            yhat = np.argmax(yhat, axis = 1)
            correct += np.sum(np.equal(yhat, y))
            total += len(y)
    return correct / total

def dense_net_model(loader, decay, numIters, lr, preTrain, focal, weighted, smoothing, gamma):
    K = 4
    if preTrain:
        net = torchvision.models.densenet121(pretrained = preTrain)
        for p in net.parameters():
            p.requires_grad = False
        for p in net.features.denseblock4.parameters():
            p.requires_grad = True
        net.classifier = torch.nn.Linear(in_features = 1024, out_features = K, bias = True)
    else:
        net = torchvision.models.densenet121(num_classes = K)
    net.to('cuda')
    if weighted:
        # jizhi: 1850, tumor: 5798, tumorln: 521, huaisi: 337
        t = np.array([1850, 5798, 521, 337])
        t = (1 / t) / np.sum(1 / t)
        weight = torch.FloatTensor(t).to('cuda')
    else:
        weight = None
    loss = LabelSmoothingLoss(classes = K, smoothing = smoothing, weight = weight, focal = focal, gamma = gamma)
    optimizer = torch.optim.Adamax(net.parameters(), lr = lr)
    print('start iterating ...')
    for iteration in range(numIters):
        if decay and iteration % 30 == 0 and iteration != 0:
            lr /= 10
        costs = 0
        for x, y in loader:
            x = x.to('cuda', dtype = torch.float)
            y = y.to('cuda')
            optimizer.zero_grad()
            yhat = net(x)
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            optimizer.step()
        if iteration % 10 == 0:
            print("Cost after iteration %d: %.3f" % (iteration, costs))
    return net

def main(lr, numIters, ratio, decay, batch_size, ifTrain, preTrain, focal, weighted, smoothing, gamma):
    trainLoader, testLoader = load_data(ratio, batch_size)
    if os.path.exists(modelpath) and not ifTrain:
        net = torch.load(modelpath)
    else:
        net = dense_net_model(trainLoader, decay, numIters, lr, preTrain, focal, weighted, smoothing, gamma)
        torch.save(net, modelpath)
    trainAccuracy = accuracy(trainLoader, net)
    testAccuracy = accuracy(testLoader, net)
    print("Train: accu = %.6f\nTest: accu = %.6f" % (trainAccuracy, testAccuracy))

global fileidxpath
global modelpath
fileidxpath = sys.argv[1]
modelpath = sys.argv[2]
params = {
         "numIters":    90,
       "batch_size":    32,
          "ifTrain":    True,
         "preTrain":    True,
            "focal":    True,
         "weighted":    True,
            "decay":    True,
               "lr":    0.001,
            "ratio":    0.7,
        "smoothing":    0.01,
            "gamma":    2
}
for key, value in params.items():
    print("{:<10s}{:>10s}".format(key, str(value)))
main(**params)