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
from collections import Counter, OrderedDict
from lifelines.utils import concordance_index
from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader, RandomSampler
if os.path.exists("/wangshuo/zhaox"):
    root = "/wangshuo/zhaox" 
else:
    root = "/home/tongxueqing/zhao"
    torch.nn.Module.dump_patches = True

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
    def __init__(self, pat, label):
        super(SurvDataset, self).__init__()
        self.pat = pat
        self.label = label
        self.length = len(self.pat)

    def __getitem__(self, i):
        return self.pat[i], self.label[self.pat[i]]

    def __len__(self):
        return self.length
      
class Data(object):
    def __init__(self, h5path, ifcombine = False):
        h5 = h5py.File(h5path, 'r')
        self.pats = h5["pats"][:]
        self.set_pat = h5['set_pat'][:]
        self.pat_fig = h5['pat_fig'][:]
        self.tps = h5['tps'][:]
        self.label = h5['label'][:]
        self.data = h5['data']
        self.postdata = h5['postdata'][:]
        self.ifcombine = ifcombine

    def __datasets_creater(self):
        datasets = {}
        train_val = np.arange(len(self.set_pat))[self.set_pat == 0]
        test = np.arange(len(self.set_pat))[self.set_pat == 1]
        if self.ifcombine:
            datasets['train'] = SurvDataset(train_val, self.label)
            datasets['test'] = SurvDataset(test, self.label)
        else:
            train_val = np.arange(len(self.set_pat))[self.set_pat == 0]
            train_val_name = self.pats[self.set_pat == 0]
            train_name = [int(x.strip().split()[1]) for x in open(os.path.join(root, "ImageProcessing/combine_model/_data/new_set.txt"))]
            train = np.array([train_val[i] for i in range(len(train_val)) if train_val_name[i] in train_name])
            val = np.array([train_val[i] for i in range(len(train_val)) if train_val_name[i] not in train_name])
            test = np.arange(len(self.set_pat))[self.set_pat == 1]
            datasets['train'] = SurvDataset(train, self.label)
            datasets['val'] = SurvDataset(val, self.label)
            datasets['test'] = SurvDataset(test, self.label)
        return datasets

    def load(self, batch_size):
        datasets = self.__datasets_creater()
        loaders = {name: DataLoader(datasets[name], batch_size = batch_size if name == 'train' else 1, shuffle = name == 'train', drop_last = name == 'train') for name in datasets}
        mapdic = {i: np.arange(len(self.pat_fig[i]))[self.pat_fig[i]] for i in range(len(self.pat_fig))}
        return loaders, mapdic, self.data, self.postdata

class Train(object):
    def __init__(self, savedmodel, savedmodel2 = None, h5path = None, infopath = None, lr = 1e-4, lr2 = None, batch_size = 64, epochs = 20, layer = 100, p = 0, weight_decay1 = 5e-4, weight_decay2 = 5e-4, optim = "SGD", lr_decay = -1, gpus = [0], lrstep = 100, cbstep = 10, figpath = None, mission = 'Surv', ifprint = True, ifcombine = False):
        self.savedmodel = savedmodel
        self.savedmodel2 = savedmodel2
        self.layer = layer
        self.p = p
        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2
        self.gpus = gpus
        self.mission = mission
        self.net = self.__load_net()
        self.loss = SurvLoss()
        self.loaders, self.mapdic, self.data, self.postdata = Data(h5path, ifcombine).load(batch_size)
        self.lr = lr
        self.lr2 = lr2 if lr != None else lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.optim = optim
        self.opt = self.__get_opt()
        self.lrstep = lrstep
        self.cbstep = cbstep
        self.ifprint = ifprint

    def __get_opt(self):
        if self.mission == 'ClassSurv':
            if self.optim == "Ada":
                return torch.optim.Adamax([
                    {'params': self.net.module.prenet.parameters(), 'lr': self.lr, "weight_decay": self.weight_decay1},
                    {"params": self.net.module.postnet.parameters(), 'lr': self.lr2, "weight_decay": self.weight_decay2},
                ])
            elif self.optim == "SGD":
                return torch.optim.SGD([
                    {'params': self.net.module.prenet.parameters(), 'lr': self.lr, "weight_decay": self.weight_decay1},
                    {"params": self.net.module.postnet.parameters(), 'lr': self.lr2, "weight_decay": self.weight_decay2},
                ], momentum = 0.9, nesterov = True)
            elif self.optim == "mix":
                opt1 = torch.optim.Adamax(self.net.module.prenet.parameters(), self.lr, weight_decay = self.weight_decay1)
                opt2 = torch.optim.SGD(self.net.module.postnet.parameters(), lr = self.lr2, weight_decay = self.weight_decay2, momentum = 0.9, nesterov = True)
                return opt1, opt2
        else:
            raise NotImplementedError("Not supported mission")

    def __lr_step(self, i):
        if self.lr_decay != -1:
            self.lr = self.lr / (1 + i * self.lr_decay)
        elif i % self.lrstep == 0 and i != 0:
            self.lr /= 10
        self.opt = self.__get_opt()

    def __load_net(self):
        if self.mission == "ClassSurv":
            try:
                prenet = torch.load(self.savedmodel).module
            except:
                prenet = torch.load(self.savedmodel)
            prenet.fc = torch.nn.Identity()
            try:
                postnet = torch.load(self.savedmodel2).module
            except:
                postnet = torch.load(self.savedmodel2)
            net = torch.nn.Sequential(OrderedDict([
                ('prenet', prenet),
                ('postnet', postnet)
            ]))
            for p in net.parameters():
                p.requires_grad = True
        else:
            raise NotImplementedError("Not supported mission")
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

    def __call_back(self, i, ifprint = True):
        if i % self.cbstep != 0:
            return
        out = "%d" % i
        cis = {}
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
            cis[name] = ci
        if ifprint:
            print_to_out(out)
        return cis

    def train(self):
        for i in range(self.epochs):
            for pats, y in self.loaders['train']:
                x = self.__get_instance(pats, True); y = y.cuda()
                self.net.train()
                if self.optim != 'mix':
                    self.opt.zero_grad()
                else:
                    for opt in self.opt: opt.zero_grad()
                yhat = self.net(x)
                cost = self.loss(yhat, y)
                cost.backward()
                if self.optim != 'mix': 
                    self.opt.step()
                else: 
                    for opt in self.opt: opt.step()
            cis = self.__call_back(i, self.ifprint)
            self.__lr_step(i)
        torch.save(self.net, modelpath)
        return cis

if __name__ == "__main__":
    global modelpath, plotpath, outfile
    modelpath, plotpath, outfile = sys.argv[1:4]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    params = {
        "savedmodel": os.path.join(root, "ImageProcessing/survival_analysis/_models/FINAL_PRENET.model"),
        "savedmodel2": os.path.join(root, "ImageProcessing/survival_analysis/_models/FINAL_POSTNET.model"),
        "h5path": os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"),
        "lr": 2e-7, # for resnet part
        "lr2": 4e-8, # for fc part
        "batch_size": 32,
        "epochs": 100,
        "gpus": [0, 1],
        "cbstep": 1,
        "lr_decay": 5.6e-6,
        "optim": "SGD",
        "weight_decay1": 1e-2,
        "weight_decay2": 0.85,
        "mission": "ClassSurv",
        "ifprint": True,
        "ifcombine": False
    }
    for key, value in params.items():
        print_to_out(key, ":", value)
    Train(**params).train()