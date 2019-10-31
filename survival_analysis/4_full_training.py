import os
import sys
import cv2
import h5py
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import metrics
from scipy import io, signal
from collections import Counter
from lifelines.utils import concordance_index
from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader, RandomSampler
root = "/wangshuo/zhaox" if os.path.exists("/wangshuo/zhaox") else "/home/tongxueqing/zhao"

def print_to_out(*args):
    with open(outfile, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

class SurvLoss(torch.nn.Module):
    def __init__(self):
        super(SurvLoss, self).__init__()
        pass

    def forward(self, Yhat, Y):
        Yhat = torch.squeeze(Yhat); Y = torch.squeeze(Y)
        order = torch.argsort(Y)
        Y = Y[order]; Yhat = Yhat[order]
        E = (Y > 0).float(); T = torch.abs(Y)
        Yhr = torch.log(torch.cumsum(torch.exp(Yhat), dim = 0))
        obs = torch.sum(E)
        Es = torch.zeros_like(E); Yhrs = torch.zeros_like(Yhr)
        j = 0
        for i in range(1, len(T)):
            if T[i] != T[i - 1]:
                Es[i - 1] = torch.sum(E[j:i])
                Yhrs[i - 1] = torch.max(Yhr[j:i])
                j = i
        Es[-1] = torch.sum(E[j:]); Yhrs[-1] = torch.max(Yhr[j:])
        loss2 = torch.sum(torch.mul(Es, Yhrs))
        loss1 = torch.sum(torch.mul(Yhat, E))
        loss = torch.div(torch.sub(loss2, loss1), obs)
        return loss

class SurvDataset(torch.utils.data.Dataset):
    def __init__(self, nameidx, set_pat):
        super(SurvDataset, self).__init__()
        pat = set_pat == nameidx
        self.index = np.array(list(range(len(pat))))[pat]
        self.label = label
        self.length = len(self.index)

    def __getitem__(self, i):
        return self.index[i], self.label[self.index[i]]

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

    def load(self, batch_size, ratio = [0.8, 0.1, 0.1]):
        datasets = {name: SurvDataset(i, self.set_pat) for i, name in enumerate(self.names)}
        loaders = {name: DataLoader(datasets[name], batch_size = batch_size if name == 'train' else 1, shuffle = name == 'train') for name in self.names}
        mapdic = {i: np.array(list(range(len(self.pat_fig[i]))))[self.pat_fig[i]] for i in range(len(self.pat_fig))}
        return loaders, mapdic, self.data

class SurvNet(torch.nn.Module):
    def __init__(self, savedmodel, savedmodel2):
        super(SurvNet, self).__init__()
        self.prenet = torch.load(savedmodel)
        res = next(self.prenet.children())
        res.fc = torch.nn.Identity()
        self.postnet = torch.load(savedmodel2)

    def forward(self, x):
        x = self.prenet(x)
        x = self.postnet(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, layers, p):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 512, out_features = layers[0], bias = True)
        self.th1 = torch.nn.Tanh()
        self.dr1 = torch.nn.Dropout(p)
        self.fc2 = torch.nn.Linear(in_features = layers[0], out_features = 1, bias = True)
        self.th2 = torch.nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.th1(x)
        x = self.dr1(x)
        x = self.fc2(x)
        x = self.th2(x)
        return x

class Train(object):
    def __init__(self, savedmodel, savedmodel2, h5path, infopath, lr, batch_size, epochs, gpus = [0], lrstep = 100, cbstep = 10, figpath = None, ratio = [0.8, 0.1, 0.1]):
        self.savedmodel = savedmodel
        self.savedmodel2 = savedmodel2
        self.gpus = gpus
        self.net = self.__load_net()
        self.loss = SurvLoss()
        self.loaders, self.mapdic, self.data = Data(h5path, infopath, figpath).load(batch_size, ratio)
        self.lr = lr
        self.epochs = epochs
        self.opt = torch.optim.SGD(self.net.parameters(), lr = self.lr, momentum = 0.9, nesterov = True)
        self.lrstep = lrstep
        self.cbstep = cbstep

    def __lr_step(self, i):
        if i % self.lrstep == 0 and i != 0:
            self.lr /= 10
            self.opt = torch.optim.SGD(self.net.parameters(), lr = self.lr, momentum = 0.9, nesterov = True)

    def __load_net(self):
        net = SurvNet(self.savedmodel, self.savedmodel2)
        net = torch.nn.DataParallel(net, device_ids = self.gpus)
        net = net.cuda()
        return net

    def __get_instance(self, pats, istrain):
        with torch.no_grad():
            self.net.eval()
            x = None
            y = np.zeros(len(pats))
            for j, pat in enumerate(pats):
                patx = torch.FloatTensor(self.__get_x(pat)).cuda()
                patyhat = self.net(patx)
                maxidx = torch.argmax(patyhat)
                if x is None:
                    x = torch.zeros((len(pats), *patx.shape[1:])).cuda()
                x[j] = patx[maxidx]
                y[j] = float(torch.max(patyhat, dim = 0).values)
        return x if istrain else y

    def __get_x(self, pat):
        return self.data[self.mapdic[int(pat)]]

    def __call_back(self, i):
        if i % self.cbstep != 0:
            return
        out = "%d" % i
        for name, loader in self.loaders.items():
            Y = np.zeros(len(loader.dataset))
            Yhat = np.zeros(len(loader.dataset))
            i = 0
            for pats, y in loader:
                Y[i:i + len(y)] = y
                Yhat[i:i + len(y)] = self.__get_instance(pats, False)
                i += len(y)
            ci = concordance_index(np.abs(Y), -Yhat, np.where(Y > 0, 1, 0))
            out += " | %s: %.3f" % (name, ci)
        print_to_out(out)

    def train(self):
        for i in range(self.epochs):
            for pats, y in self.loaders['train']:
                x = self.__get_instance(pats, True)
                y = y.cuda()
                self.net.train()
                self.opt.zero_grad()
                yhat = self.net(x)
                cost = self.loss(yhat, y)
                cost.backward()
                self.opt.step()
            self.__call_back(i)
            self.__lr_step(i)
        torch.save(self.net, modelpath)

if __name__ == "__main__":
    global modelpath; global plotpath; global matpath; global outfile
    modelpath, plotpath, matpath, outfile = sys.argv[1:5]

    params = {
        "savedmodel": os.path.join(root, "ImageProcessing/stain_classification/_models/success.Oct.27_14:40.model"),
        "savedmodel2": os.path.join(root, "ImageProcessing/survival_analysis/_models/success.Oct.30_20:18.model"),
        "h5path": os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"),
        "infopath": os.path.join(root, "ImageProcessing/survival_analysis/_data/merged.csv"),
        "figpath": os.path.join(root, "ImageProcessing/stain_classification/_data/subsets"),
        "lr": 1e-7,
        "batch_size": 64,
        "epochs": 200,
        "gpus": [0],
        "lrstep": 70,
        "cbstep": 1,
        "ratio": [0.8, 0.1, 0.1]
    }
    for key, value in params.items():
        print_to_out(key, ":", value)
    Train(**params).train()