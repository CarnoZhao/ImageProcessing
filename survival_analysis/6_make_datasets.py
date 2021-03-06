import os
import cv2
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
root = "/wangshuo/zhaox" if os.path.exists("/wangshuo/zhaox") else "/home/tongxueqing/zhao"

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
    def __init__(self, h5path, infopath = None, figpath = None, ratio = [0.7, 0.15, 0.15], createpost = False):
        if not os.path.exists(h5path):
            self.__make_h5(h5path, figpath, infopath)
        h5 = h5py.File(h5path, 'a')
        self.__make_post(h5)
        # self.__make_pred(h5)
        self.set_pat = h5['set_pat'][:]
        self.pat_fig = h5['pat_fig'][:]
        self.tps = h5['tps'][:]
        self.label = h5['label'][:]
        self.data = h5['data']
        self.pats = h5['pats']
        self.names = ['train', 'val', 'test']

    def __make_post(self, h5):
        with torch.no_grad():
            for fold in range(4):
                net = torch.load(os.path.join(root, "ImageProcessing/stain_classification/_models/fold%d.resnet.model" % fold)).module
                net.fc = torch.nn.Identity()
                net = net.cuda()
                post = "postdata%d" % fold
                try: h5.pop(post)
                except: pass
                h5.create_dataset(post, shape = (len(h5['data']), 512))
                for i in range(len(h5['data'])):
                    img = h5['data'][i:i+1, :, :, :]
                    yhat = net(torch.FloatTensor(img).cuda())
                    h5[post][i, :] = yhat.cpu()
    
    def __make_pred(self, h5):
        with torch.no_grad():
            net = torch.load(os.path.join(root, "ImageProcessing/survival_analysis/_models/success.Nov.06_14:18.model")).module
            net.fc = torch.nn.Identity()
            net = net.cuda()
            try: h5.pop("pred")
            except: pass
            h5.create_dataset("pred", shape = (len(h5['data']), 1))
            for i in range(len(h5['data'])):
                img = h5['data'][i:i+1, :, :, :]
                yhat = net(torch.FloatTensor(img).cuda())
                h5['pred'][i] = yhat.cpu()

    def __make_h5(self, h5path, figpath, infopath):
        tpdic = {'huaisi': 0, 'jizhi': 1, 'tumor': 2, 'tumorln': 3}
        tiffiles = [f.strip() for f in os.popen("find %s -name \"*.tif\"" % figpath)]
        info = pd.read_csv(infopath)
        pats = np.array(list(set([eval(os.path.basename(f).split('_')[0]) for f in tiffiles]) & set(info.number)))
        np.random.shuffle(pats)

        # get pat index: names == 0/1/2
        pat_fig = np.zeros((len(pats), len(tiffiles)), dtype = np.bool)
        for j, tif in enumerate(tiffiles):
            for i, pat in enumerate(pats):
                if str(pat) in tif:
                    pat_fig[i, j] = np.True_
                    break
        # get fig index: np.sum(pat_fig[names == 0/1/2], axis = 0, dtype = np.bool)
        tps = np.array([tpdic[os.path.basename(f).split('_')[1]] for f in tiffiles])
        label = np.zeros(len(pats))
        names = np.zeros(len(pats))
        for j, pat in enumerate(pats):
            for i in range(len(info)):
                if info.number[i] == pat:
                    label[j] = info.time[i] * (1 if info.event[i] else -1) * 30
                    names[j] = 0 if info.hosi[i] == "ZF" else 1
                    break
        h5 = h5py.File(h5path, 'w')
        h5.create_dataset('set_pat', data = names)
        h5.create_dataset('pat_fig', data = pat_fig)
        h5.create_dataset('tps', data = tps)
        h5.create_dataset('label', data = label)
        h5.create_dataset('pats', data = pats)
        h5.create_dataset('data', shape = (len(tiffiles), 3, 512, 512))
        for i, tif in enumerate(tiffiles):
            img = cv2.imread(tif)[:, :, ::-1].transpose((2, 0, 1)) / 255
            h5['data'][i, :, :, :] = img
        h5.close()

    def load(self, batch_size):
        Dataset = ClassifierDataset
        datasets = {name: Dataset(i, self.set_pat, self.pat_fig, self.data, self.tps) for i, name in enumerate(self.names)}
        loaders = {name: DataLoader(datasets[name], batch_size = batch_size if name == 'train' else 1, shuffle = name == 'train') for name in self.names}
        return loaders

d = Data(os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"))

def make_summary():
    from collections import Counter
    h5path = "/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/compiled.h5"
    d = Data(h5path)
    set_pat = d.set_pat
    print(Counter(set_pat))
    pat_fig = d.pat_fig
    tps = d.tps
    for i in range(3):
        tpscount(i, set_pat, pat_fig, tps)

def tpscount(i, set_pat, pat_fig, tps):
    pats = set_pat == i
    fig = pat_fig[pats]
    fig = np.sum(fig, axis = 0, dtype = np.bool)
    tp = tps[fig]

