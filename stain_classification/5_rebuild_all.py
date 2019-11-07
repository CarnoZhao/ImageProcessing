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

root = "/wangshuo/zhaox" if os.path.exists("/wangshuo/zhaox") else "/home/tongxueqing/zhao"


def print_to_out(*args):
    with open(outfile, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

class Loss(torch.nn.Module):
    def __init__(self, K, smoothing = 0, gamma = 0):
        super(Loss, self).__init__()
        self.K = K
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, Yhat, Y):
        Yhat = Yhat.softmax(-1)
        oneHot = torch.zeros_like(Yhat)
        oneHot = oneHot.scatter_(1, Y.data.unsqueeze(1), 1)
        oneHot = torch.clamp(oneHot, self.smoothing / (self.K - 1), 1.0 - self.smoothing)
        pt = (oneHot * Yhat).sum(1)  + 1e-10
        logpt = pt.log()
        loss = -torch.pow(1 - pt, self.gamma) * logpt
        return loss.mean()

class ClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, pat, pat_fig, data, tps):
        super(ClassifierDataset, self).__init__()
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
    def __init__(self, h5path, fold = 4):
        h5 = h5py.File(h5path, 'r')
        self.fold = fold
        self.set_pat = h5['set_pat'][:]
        self.pat_fig = h5['pat_fig'][:]
        self.tps = h5['tps'][:]
        self.label = h5['label'][:]
        self.data = h5['data']
        self.names = ['train', 'val', 'test']
        self.k = 0

    def __datasets_creater(self):
        train_val = np.arange(len(self.set_pat))[self.set_pat == 0]
        inf = int(self.k * len(train_val) / self.fold)
        sup = int((self.k + 1) * len(train_val) / self.fold)
        validx = np.arange(len(train_val))
        validx = np.bitwise_and(np.greater_equal(validx, inf), np.less(validx, sup))
        train = train_val[np.bitwise_not(validx)]
        val = train_val[validx]
        test = np.arange(len(self.set_pat))[self.set_pat == 1]
        datasets = {}
        datasets['train'] = ClassifierDataset(train, self.pat_fig, self.data, self.tps)
        datasets['val'] = ClassifierDataset(val, self.pat_fig, self.data, self.tps)
        datasets['test'] = ClassifierDataset(test, self.pat_fig, self.data, self.tps)
        self.k += 1
        return datasets

    def load(self, batch_size):
        datasets = self.__datasets_creater()
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

    def plot_roc_auc(self, data, k):
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

    def printplot(self, net, loaders, K, k):
        data = {'names': list(loaders.keys())}
        accus = {}
        for key, loader in loaders.items():
            Y, Yhat, accu = self.predict(net, loader, K)
            data[key + 'Y'] = Y
            data[key + 'Yhat'] = Yhat
            accus[key] = accu
        io.savemat(matpath, data)
        self.plot_roc_auc(data, k)
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
    def __init__(self, h5path, iters, K, pretrain, weight_decay, lr, batch_size, gpus, fold = 4, gamma = 0, smoothing = 0, step = 3, nettype = "densenet"):
        self.fold = fold
        self.iters = iters
        self.K = K
        self.lr = lr
        self.lr_bak = lr
        self.batch_size = batch_size
        self.step = step
        self.weight_decay = weight_decay
        self.nettype = nettype
        self.gpus = gpus
        self.pretrain = pretrain
        self.D = Data(h5path, fold)
        self.loss = Loss(self.K, smoothing, gamma)

    def __load_net(self, nettype, gpus, pretrain):
        if pretrain:
            if nettype == "densenet":
                net = torchvision.models.densenet121(pretrained = True)
                net.classifier = torch.nn.Linear(1024, self.K, bias = True)
            elif nettype == "resnext":
                net = torchvision.models.resnext50_32x4d(pretrained = True)
                net.fc = torch.nn.Linear(2048, self.K, bias = True)
            elif nettype == "resnet":
                net = torchvision.models.resnet18(pretrained = True)
                net.fc = torch.nn.Linear(512, self.K, bias = True)
        else:
            net = torchvision.models.resnet18(num_classes = self.K)
        net = torch.nn.DataParallel(net, device_ids = gpus)
        net = net.cuda()
        return net

    def __call_back(self, i, loaders):
        if i % self.step != 0:
            return 
        self.net.eval()
        out = "%d" % i
        with torch.no_grad():
            for name, loader in loaders.items():
                a = b = 0
                for x, y in loader:
                    x = x.cuda(); y = y.cuda()
                    yhat = self.net(x)
                    a += len(y)
                    b += int(torch.sum(torch.argmax(yhat, dim = 1) == y))
                out += " | %s: %.4f" % (name, b / a)
        print_to_out(out)
            
    def __load_opt(self):
        return torch.optim.Adamax(self.net.parameters(), lr = self.lr, weight_decay = self.weight_decay)

    def __lr_step(self, i):
        if i % 30 == 0:
            self.lr /= 10
            self.opt = self.__load_opt()

    def __evalu(self, loaders, k):
        self.net.eval()
        Evaluation().printplot(self.net, loaders, self.K, k)
        Evaluation().cnter(matpath, self.K)

    def train(self):
        for k in range(self.fold):
            self.net = self.__load_net(self.nettype, self.gpus, self.pretrain)
            self.opt = self.__load_opt()
            print_to_out("in fold %d:" % k)
            loaders = self.D.load(self.batch_size)
            for i in range(1, self.iters + 1):
                self.net.train()
                for x, y in loaders['train']:
                    x = x.cuda(); y = y.cuda()
                    yhat = self.net(x)
                    self.opt.zero_grad()
                    cost = self.loss(yhat, y)
                    cost.backward()
                    self.opt.step()
                self.__call_back(i, loaders)
                self.__lr_step(i)
            torch.save(self.net, modelpath)
            self.__evalu(loaders, k)
            for saved in [modelpath, matpath, plotpath]:
                os.system("mv %s %s" % (saved, saved.replace("Nov", "%d.Nov" % k)))
            self.lr = self.lr_bak


global modelpath; global plotpath; global matpath; global outfile
modelpath, plotpath, matpath, outfile = sys.argv[1:5]
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"
params = {
              "h5path": os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"),
             "iters":    50,
                 "K":    4,
          "pretrain":    True,
                "lr":    2.5e-6,
        "batch_size":    32,
             "gamma":    2,
         "smoothing":    0.01,
              "step":    1,
      "weight_decay":    6e-3,
              "gpus":    [0,1,2],
              "fold":    4,
           "nettype":    "densenet"
}
for k, v in params.items():
    print_to_out(k, ':', v)
if __name__ == "__main__":
    Train(**params).train()
