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
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
root = "/wangshuo/zhaox" if os.path.exists("/wangshuo/zhaox") else "/home/tongxueqing/zhao"


def print_to_out(*args):
    with open(outfile, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')


class Loss(torch.nn.Module):
    def __init__(self, K, smoothing=0.0, gamma=0):
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

    def predict(self, net, loader, K):
        Y = np.zeros((0, K))
        Yhat = np.zeros((0, K))
        with torch.no_grad():
            for j, (x, y) in enumerate(loader):
                x = x.to('cuda', dtype=torch.float)
                y = torch.nn.functional.one_hot(y, num_classes=K)
                Y = np.vstack([Y, y.numpy()])
                yhat = torch.softmax(net(x), dim=1).cpu()
                Yhat = np.vstack([Yhat, yhat.numpy()])
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

    def printplot(self, net, loaders, K):
        data = {'names': list(loaders.keys())}
        accus = {}
        for key, loader in loaders.items():
            Y, Yhat, accu = self.predict(net, loader, K)
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
    def __init__(self, h5path, iters, K, pretrain, lr, batch_size, gpus, gamma = 0, smoothing = 0, step = 3):
        self.h5path = h5path
        self.iters = iters
        self.K = K
        self.pretrain = pretrain
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.smoothing = smoothing
        self.step = step
        self.gpus = gpus

    def _load_net(self):
        if self.pretrain:
            net = torchvision.models.densenet121(pretrained = True)
            net.classifier = torch.nn.Linear(in_features=1024, out_features = self.K, bias = True)
        else:
            net = torchvision.models.resnet18(num_classes = self.K)
        net = torch.nn.DataParallel(net, device_ids = self.gpus)
        net = net.cuda()
        return net


    def _call_back(self, i, net, loaders, loss):
        if i % self.step != 0:
            return 
        net.eval()
        out = "%d" % i
        with torch.no_grad():
            for name, loader in loaders.items():
                a = b = 0
                for x, y in loader:
                    x = x.cuda(); y = y.cuda()
                    yhat = net(x)
                    a += len(y)
                    b += int(torch.sum(torch.argmax(yhat, dim = 1) == y))
                out += " | %s: %.4f" % (name, b / a)
        print_to_out(out)
            

    def train(self):
        loaders = Data(self.h5path).load(self.batch_size)
        loader = loaders['train']
        net = self._load_net()
        loss = Loss(self.K, self.smoothing, self.gamma)
        opt = torch.optim.Adamax(net.parameters(), lr = self.lr)
        for i in range(1, self.iters + 1):
            if i % 30 == 0:
                self.lr /= 10
                opt = torch.optim.Adamax(net.parameters(), lr = self.lr)
            net.train()
            for j, (x, y) in enumerate(loader):
                x = x.cuda(); y = y.cuda()
                yhat = net(x)
                opt.zero_grad()
                cost = loss(yhat, y)
                cost.backward()
                opt.step()
            self._call_back(i, net, loaders, loss)
        torch.save(net, modelpath)
        net.eval()
        Evaluation().printplot(net, loaders, self.K)
        Evaluation().cnter(matpath, self.K)
        return net


global modelpath; global plotpath; global matpath; global outfile
modelpath, plotpath, matpath, outfile = sys.argv[1:5]

params = {
              "h5path": os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"),
             "iters":    60,
                 "K":    4,
          "pretrain":    True,
                "lr":    3e-6,
        "batch_size":    32,
             "gamma":    0,
         "smoothing":    0.001,
              "step":    1,
              "gpus":    [0, 1, 2]
}
for k, v in params.items():
    print_to_out(k, ':', v)
if __name__ == "__main__":
    net = Train(**params).train()
