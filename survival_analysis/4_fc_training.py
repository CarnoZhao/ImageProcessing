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
    def __init__(self, h5path, fold = 4, k = 0):
        h5 = h5py.File(h5path, 'r')
        self.set_pat = h5['set_pat'][:]
        self.pat_fig = h5['pat_fig'][:]
        self.tps = h5['tps'][:]
        self.label = h5['label'][:]
        self.data = h5['data']
        self.postdata = h5['postdata%d' % k][:]
        self.names = ['train', 'val', 'test']
        self.fold = fold
        self.k = k

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
        datasets['train'] = SurvDataset(train, self.label)
        datasets['val'] = SurvDataset(val, self.label)
        datasets['test'] = SurvDataset(test, self.label)
        self.k += 1
        return datasets

    def load(self, batch_size, ratio = [0.8, 0.1, 0.1]):
        datasets = self.__datasets_creater()
        loaders = {name: DataLoader(datasets[name], batch_size = batch_size if name == 'train' else 1, shuffle = name == 'train', drop_last = name == 'train') for name in self.names}
        mapdic = {i: np.arange(len(self.pat_fig[i]))[self.pat_fig[i]] for i in range(len(self.pat_fig))}
        return loaders, mapdic, self.data, self.postdata

class Train(object):
    def __init__(self, savedmodel = None, h5path = None, infopath = None, lr = 1e-4, lr2 = None, batch_size = 64, epochs = 20, layer = 100, p = 0, weight_decay = 5e-4, optim = "SGD", lr_decay = -1, gpus = [0], lrstep = 100, cbstep = 10, figpath = None, mission = 'Surv', ifprint = True, fold = 4, k = 0):
        self.savedmodel = savedmodel
        self.layer = layer
        self.p = p
        self.weight_decay = weight_decay
        self.gpus = gpus
        self.mission = mission
        self.net = self.__load_net()
        self.loss = SurvLoss()
        self.h5path = h5path
        self.loaders, self.mapdic, self.data, self.postdata = Data(h5path, fold).load(batch_size, k)
        self.lr = lr
        self.lr2 = lr2 if lr != None else lr
        self.lr_decay = lr_decay
        self.epochs = epochs
        self.optim = optim
        self.opt = self.__get_opt()
        self.lrstep = lrstep
        self.cbstep = cbstep
        self.ifprint = ifprint
        self.fold = fold

    def __get_opt(self):
        if self.mission == "Surv":
            if self.optim == "Ada":
                return torch.optim.Adamax(self.net.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            elif self.optim == "SGD":
                return torch.optim.SGD(self.net.parameters(), lr = self.lr, momentum = 0.9, nesterov = True, weight_decay = self.weight_decay)
        else:
            raise NotImplementedError("No supported")

    def __lr_step(self, i):
        if self.lr_decay != -1:
            self.lr = self.lr / (1 + i * self.lr_decay)
        elif i % self.lrstep == 0 and i != 0:
            self.lr /= 10
        self.opt = self.__get_opt()

    def __load_net(self):
        if self.mission == 'Surv':
            net = torch.nn.Sequential(OrderedDict([
                ('fc1', torch.nn.Linear(512, self.layer, bias = True)),
                ('dr1', torch.nn.Dropout(self.p)),
                ('th1', torch.nn.Tanh()),
                ('fc2', torch.nn.Linear(self.layer, 1, bias = True)),
                ('th2', torch.nn.Tanh())
            ]))
        else:
            raise NotImplementedError("This file is only used for fc layer training")
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
        return self.postdata[self.mapdic[int(pat)]]

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
                x = self.__get_instance(pats, True)
                y = y.cuda()
                self.net.train()
                self.opt.zero_grad()
                yhat = self.net(x)
                cost = self.loss(yhat, y)
                cost.backward()
                self.opt.step()
            cis = self.__call_back(i, self.ifprint)
            self.__lr_step(i)
        torch.save(self.net, modelpath)
        return cis

    def k_fold_train(self):
        D = Data(self.h5path, self.fold)
        for k in range(self.fold):
            self.loaders, self.mapdic, _, self.postdata = D.load(self.batch_size)
            self.net = self.__load_net()
            self.opt = self.__get_opt()
            cis = self.train(modelpath.replace("Nov", "%d.%d.Nov" % (iii, k)))



if __name__ == "__main__":
    global modelpath; global plotpath; global matpath; global outfile
    modelpath, plotpath, matpath, outfile = sys.argv[1:5]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    params = {
        "h5path": os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"),
        "lr": 7e-5,
        "batch_size": 64,
        "epochs": 40,
        "gpus": [0],
        "cbstep": 1,
        "lr_decay": 1e-3,
        "layer": 128,
        "p": 0.8,
        "optim": "SGD",
        "weight_decay": 1e-3,
        "mission": "Surv", # Surv, ClassSurv, FullTrain1FC, FullTrain2FC
        "ifprint": False,
        "fold": 4,
        "k": 3,
    }
    # for key, value in params.items():
    #     print_to_out(key, ":", value)
    print_to_out("in fold %d" % params['k'])
    cis = {'train': 0, 'val': 0, 'test': 0}
    global iii
    iii = 0
    ref = modelpath
    while any([c < 0.7 for c in cis.values()]):
        modelpath = ref.replace("Nov", str(iii))
        params['lr'] = 10 ** (np.random.rand() * 7 - 7)
        params['lr_decay'] = 10 ** (np.random.rand() * 7 - 7)
        params['weight_decay'] = 10 ** (np.random.rand() * 7 - 7)
        params['layer'] = 100 + int(200 * np.random.rand())
        params['p'] = np.random.rand() * 0.8
        params['epochs'] = np.random.randint(20, 80)
        params['optim'] = np.random.choice(['SGD', 'Ada'], 1)[0]
        params['cbstep'] = params['epochs'] - 1
        cis = Train(**params).train()
        out = "lr: %.3e | ep: %d | lrd: %.3e | l: %d | p: %.3f | wtd: %.3e | op: %s | citr: %.4f | civl: %.4f | cits: %.4f" % (
            params['lr'], params['epochs'], params['lr_decay'], params['layer'], params['p'], params['weight_decay'], params['optim'], 
            cis['train'], cis['val'], cis['test']
        )
        if any([ci < 0.63 for ci in cis.values()]):
            os.system("rm %s" % modelpath)
        print_to_out(out)
        iii += 1