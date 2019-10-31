import os
import sys
import cv2
import h5py
import torch
import torchvision
import numpy as np
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from scipy import io, signal
from collections import Counter
from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader, RandomSampler
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def print_to_out(*args):
    with open(outfile, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')


class Loss(torch.nn.Module):
    def __init__(self, K, smoothing=0.0, gamma=0, weights=1):
        super(Loss, self).__init__()
        self.criterion = torch.nn.KLDivLoss()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.K = K
        self.true_dist = None

    def forward(self, Yhat, Y):
        assert Yhat.size(1) == self.K
        Yhat = Yhat.log_softmax(-1)
        true_dist = Yhat.data.clone()
        true_dist.fill_(self.smoothing / (self.K - 1))
        true_dist.scatter_(1, Y.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist
        return self.criterion(Yhat, torch.autograd.Variable(true_dist, requires_grad = False))

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, nameidx, set_pat, pat_fig, data, tps):
        super(ClassifierDataset, self).__init__()
        pat = set_pat == nameidx
        fig = np.sum(pat_fig[pat], axis = 0, dtype = np.bool)
        self.index = np.array(list(range(len(fig))))[fig]
        self.data = data
        self.tps = tps
        self.length = len(self.index)

    def __getitem__(self, i):
        idx = self.index[i]
        return self.data[idx], self.tps[idx]
    
    def __len__(self):
        return self.length

class Data(object):
    def __init__(self, h5path):
        h5 = h5py.File(h5path, 'r')
        self.set_pat = h5['set_pat'][:]
        self.pat_fig = h5['pat_fig'][:]
        self.tps = h5['tps'][:]
        self.label = h5['label'][:]
        self.data = h5['data']
        self.names = ['train', 'val', 'test']

    def load(self, batch_size):
        datasets = {name: ClassifierDataset(i, self.set_pat, self.pat_fig, self.data, self.tps) for i, name in enumerate(self.names)}
        loaders = {name: DataLoader(datasets[name], batch_size = batch_size if name == 'train' else 1, shuffle = name == 'train') for name in self.names}
        return loaders


class Evaluation(object):
    def __init__(self):
        pass

    def predict(self, net, loader, K, subsample):
        Y = np.zeros((0, K))
        Yhat = np.zeros((0, K))
        with torch.no_grad():
            J = len(loader)
            for j, (x, y) in enumerate(loader):
                x = x.to('cuda', dtype=torch.float)
                y = torch.nn.functional.one_hot(y, num_classes=K)
                Y = np.vstack([Y, y.numpy()])
                yhat = torch.softmax(net(x), dim=1).cpu()
                Yhat = np.vstack([Yhat, yhat.numpy()])
                if j == int(J * subsample):
                    break
        accu = np.mean(np.equal(np.argmax(Y, axis=1), np.argmax(Yhat, axis=1)))
        return Y, Yhat, accu

    def plot_roc_auc(self, data):
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'b']
        for i, name in enumerate(data['names']):
            Y = data[name + 'Y']
            Yhat = data[name + 'Yhat']
            fpr, tpr, _ = metrics.roc_curve(Y.ravel(), Yhat.ravel())
            auc = metrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, c=colors[i], lw=1, alpha=0.7,
                    label=u'%sAUC=%.3f' % (name, auc))
        ax.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        ax.set_xlim((-0.01, 1.02))
        ax.set_ylim((-0.01, 1.02))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(b=True, ls=':')
        plt.legend()
        plt.savefig(plotpath)

    def printplot(self, net, loaders, K, subsample):
        data = {'names': list(loaders.keys())}
        accus = {}
        for key, loader in loaders.items():
            Y, Yhat, accu = self.predict(net, loader, K, subsample)
            data[key + 'Y'] = Y
            data[key + 'Yhat'] = Yhat
            accus[key] = accu
        io.savemat(matpath, data)
        self.plot_roc_auc(data)
        for name in data['names']:
            print_to_out('Accuracy in %s = %.6f' % (name, accus[name]))

    def cnter(self, matpath, K):
        cnts = {}
        data = io.loadmat(matpath)
        for name in data['names']:
            name = name.strip()
            print_to_out(name + ":")
            Y = np.argmax(data[name + 'Y'], axis = 1)
            Yhat = np.argmax(data[name + 'Yhat'], axis = 1)
            cnt = np.zeros((K, K))
            for yi, yih in zip(Y, Yhat):
                cnt[yi, yih] += 1
            print_to_out(np.round(cnt / np.sum(cnt, axis = 1, keepdims = True), decimals = 2))
            cnts[name] = cnt
        return cnts


class Train(object):
    def __init__(self, path, iters, K, pretrain, lr, batch_size, gpus, type_weights=None, loss_weights=None, gamma=0, smoothing=0, step=3, ignore_classes=[], subsample=1):
        self.path = path
        self.iters = iters
        self.K = K
        self.pretrain = pretrain
        self.lr = lr
        self.batch_size = batch_size
        self.type_weights = type_weights
        self.loss_weights = loss_weights
        self.gamma = gamma
        self.smoothing = smoothing
        self.step = step
        self.ignore_classes = ignore_classes
        self.subsample = subsample
        self.gpus = gpus

    def _load_net(self):
        if self.pretrain:
            net = torchvision.models.resnet18(pretrained=True)
            net.fc = torch.nn.Linear(in_features=512, out_features=self.K)
        else:
            net = torchvision.models.resnet18(num_classes=self.K)
        net = torch.nn.DataParallel(net, device_ids=self.gpus)
        net = net.cuda()
        return net


    def _call_back(self, i, net, loaders, loss, subsample, trainresult):
        traincosts, trainaccu = trainresult
        if i == 0:
            print_to_out("\t|")
        if i % self.step == 0:
            testcosts = 0
            correct = 0
            total = 0
            with torch.no_grad():
                J = len(loaders['test'])
                for j, (x, y) in enumerate(loaders['test']):
                    net.eval()
                    x = x.cuda(); y = y.cuda()
                    yhat = net(x)
                    total += len(y)
                    correct += int(torch.sum(torch.argmax(yhat, dim = 1) == y))
                    testcosts += float(loss(yhat, y))
                    if j == int(J * subsample):
                        break
            print_to_out("%d\t| train: cost: %.3f  accuracy: %.3f | test: cost: %.3f  accuracy: %.3f" % (i, traincosts, trainaccu, testcosts, correct / total))

    def train(self):
        loaders = Data(self.path, self.batch_size,
                       self.type_weights, self.ignore_classes).load()
        loader = loaders['train']
        net = self._load_net()
        loss = Loss(self.K, self.smoothing, self.gamma)
        opt = torch.optim.Adamax(net.parameters(), lr = self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(opt, 10, 0.1)
        for i in range(1, self.iters + 1):
            # if i == 30:
            #     self.lr /= 10
            #     opt = torch.optim.Adamax(net.parameters(), lr = self.lr)
            costs = 0; total = 0; correct = 0
            J = len(loader)
            net.train()
            for j, (x, y) in enumerate(loader):
                x = x.cuda(); y = y.cuda()
                yhat = net(x)
                opt.zero_grad()
                cost = loss(yhat, y)
                cost.backward()
                costs += float(cost); total += len(y)
                correct += int(torch.sum(torch.argmax(yhat, dim = 1) == y))
                opt.step()# ; scheduler.step()
                if j == int(J * self.subsample):
                    break
            self._call_back(i, net, loaders, loss, self.subsample, (costs, correct / total))
        torch.save(net, modelpath)
        net.eval()
        Evaluation().printplot(net, loaders, self.K, self.subsample)
        Evaluation().cnter(matpath, self.K)
        return net


global modelpath
global plotpath
global matpath
global outfile
modelpath = sys.argv[1]
plotpath = sys.argv[2]
matpath = sys.argv[3]
outfile = sys.argv[4]

params = {
              "path": "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/self_normed",
             "iters":    200,
                 "K":    4,
          "pretrain":    True,
                "lr":    0.000001,
        "batch_size":    64,
      "loss_weights":    None,
             "gamma":    0,
         "smoothing":    0.001,
              "step":    1,
         "subsample":    1,
    "ignore_classes":    [],
              "gpus":    [0, 1]
}

if __name__ == "__main__":
    net = Train(**params).train()
