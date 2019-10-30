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
if os.path.exists('/wangshuo/zhaox'):
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

class H5Datasets(torch.utils.data.Dataset):
    def __init__(self, pats, patdic):
        super(H5Datasets, self).__init__()
        self.pats = pats
        self.label = [patdic[pat] for pat in self.pats]
        self.length = len(self.pats)

    def __getitem__(self, i):
        return self.pats[i], self.label[i]

    def __len__(self):
        return self.length
      
class Data(object):
    def __init__(self, h5path, infopath, figpath = None):
        self.h5path = h5path
        self.infopath = infopath
        if not os.path.exists(self.h5path):
            self.__make_h5(figpath)
        self.h5 = h5py.File(self.h5path, 'r')
        self.data = self.h5['data']
        self.pats = self.h5['pat']
        self.tp = self.h5['tp']
        self.hosi = self.h5['hosi']
        info = pd.read_csv(self.infopath)
        self.patdic = {info.number[i]: info.time[i] * 30 * (1 if info.event[i] > 0 else -1) for i in range(len(info))}
        self.mapdic = {}
        for i, pat in enumerate(self.pats):
            if pat not in self.patdic:
                continue
            if pat not in self.mapdic:
                self.mapdic[pat] = [i]
            else:
                self.mapdic[pat].append(i)
        for pat in self.mapdic:
            self.mapdic[pat] = np.array(self.mapdic[pat])
        self.pats = list(self.patdic.keys())

    def __make_h5(self, figpath):
        tpdic = {'huaisi': 0, 'jizhi': 1, 'tumor': 2, 'tumorln': 3}
        tiffiles = [f.strip() for f in os.popen("find %s -name \"*.tif\"" % figpath)]
        pat = [eval(os.path.basename(f).split('_')[0]) for f in tiffiles]
        tp = [tpdic[os.path.basename(f).split('_')[1]] for f in tiffiles]
        hosi = [1 if os.path.basename(f).split('_')[3] == 'ZF' else 0 for f in tiffiles]
        h5 = h5py.File(self.h5path, 'w')
        h5.create_dataset('pat', data = np.array(pat))
        h5.create_dataset('tp', data = np.array(tp))
        h5.create_dataset('hosi', data = np.array(hosi))
        img = cv2.imread(tiffiles[0])[:, :, ::-1]
        h5.create_dataset('data', shape = (len(tiffiles), img.shape[-1], *img.shape[:2]))
        for i, tif in enumerate(tiffiles):
            img = cv2.imread(tif)[:, :, ::-1].transpose((2, 0, 1)) / 255
            h5['data'][i, :, :, :] = img
        h5.close()

    def load(self, batch_size, ratio = [0.8, 0.1, 0.1]):
        np.random.shuffle(self.pats)
        trainpats = self.pats[:int(ratio[0] * len(self.pats))]
        valpats = self.pats[int(ratio[0] * len(self.pats)):int((ratio[0] + ratio[1]) * len(self.pats))]
        testpats = self.pats[int((ratio[0] + ratio[1]) * len(self.pats)):]
        namepats = {'train': trainpats, 'val': valpats, 'test': testpats}
        namesets = {name: H5Datasets(namepats[name], self.patdic) for name in namepats}
        loaders = {name: torch.utils.data.DataLoader(
            namesets[name], 
            batch_size = batch_size if name == 'train' else 1,
            shuffle = name == 'train')
            for name in namesets}
        return loaders, self.mapdic, self.data

class SurvNet(torch.nn.Module):
    def __init__(self, savedmodel, savedmodel2):
        super(SurvNet, self).__init__()
        self.prenet = torch.load(savedmodel)
        self.prenet.fc = torch.nn.Identity()
        self.postnet = torch.load(savedmodel2)

    def forward(self, x):
        x = self.prenet(x)
        x = self.postnet(x)
        return x

class Train(object):
    def __init__(self, savedmodel, savedmodel2, h5path, infopath, lr, batch_size, epochs, gpus = [0], lrstep = 100, cbstep = 10, figpath = None):
        self.savedmodel = savedmodel
        self.savedmodel2 = savedmodel2
        self.gpus = gpus
        self.net = self.__load_net()
        self.loss = SurvLoss()
        self.loaders, self.mapdic, self.data = Data(h5path, infopath, figpath).load(batch_size)
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
        net.eval()
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

if __name__ == "__main__":
    global modelpath; global plotpath; global matpath; global outfile
    modelpath, plotpath, matpath, outfile = sys.argv[1:5]

    params = {
        "savedmodel": "",
        "savedmodel2": "",
        "h5path": "",
        "infopath": "",
        "figpath": "",
        "lr": 1e-7,
        "batch_size": 64,
        "epochs": 50,
        "gpus": [0],
        "lrstep": 20,
        "cbstep": 10,
    }