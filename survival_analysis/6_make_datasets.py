import os
import cv2
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader


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


class SurvDataset(torch.utils.data.Dataset):
    pass

class Data(object):
    def __init__(self, h5path, infopath = None, figpath = None, ratio = [0.7, 0.15, 0.15]):
        if not os.path.exists(h5path):
            self.__make_h5(h5path, figpath, infopath, ratio)
        h5 = h5py.File(h5path, 'r')
        self.set_pat = h5['set_pat'][:]
        self.pat_fig = h5['pat_fig'][:]
        self.tps = h5['tps'][:]
        self.label = h5['label'][:]
        self.data = h5['data']
        self.pats = h5['pats']
        self.names = ['train', 'val', 'test']

    def __make_h5(self, h5path, figpath, infopath, ratio):
        tpdic = {'huaisi': 0, 'jizhi': 1, 'tumor': 2, 'tumorln': 3}
        tiffiles = [f.strip() for f in os.popen("find %s -name \"*.tif\"" % figpath)]
        info = pd.read_csv(infopath)
        # pats = np.array(list(set([eval(os.path.basename(f).split('_')[0]) for f in tiffiles]) & set(info.number)))
        # np.random.shuffle(pats)
        pats = np.array([1505796, 1405481, 1503566, 1510442, 1706442, 1402299, 1701138,
       1700462, 1511382, 1504186, 1500694, 1407606, 1401317, 1608527,
       1702583, 1507884, 1520263, 1512377, 1512396, 1611422, 1520851,
       1519706, 1701663, 1701668, 1605283, 1313783, 1407348, 1405152,
       1401150, 1614456, 1607174, 1613361, 1408401, 1405035, 1407127,
       1615309, 1702589, 1602586, 1610256, 1522923, 1501417, 1409419,
       1704471, 1410317, 1601239, 1600991, 1602137, 1507284, 1600296,
       1503175, 1510039, 1505414, 1501415, 1602926, 1602238, 1404116,
       1404637, 1505902, 1316664, 1607437, 1613107, 1602751, 1604103,
       1600294, 1607972, 1605606, 1513602, 1401438, 1403311, 1406199,
       1501684, 1603591, 1604547, 1402909, 1508466, 1512101, 1402805,
       1415838, 1610465, 1508191, 1406322, 1422841, 1503005, 1301630,
       1507217, 1306115,  140633,  152216, 1700342, 1601181, 1609874,
       1503718, 1703032, 1600870, 1509038, 1505057, 1510358, 1700177,
       1501416, 1509570, 1403521, 1300134, 1604176, 1614914, 1604846,
       1501050, 1603444, 1607236, 1502675, 1510385, 1610180, 1510941,
       1511384, 1501010, 1601350,  154320, 1403257, 1700618, 1404315,
       1502357, 1615043, 1805044, 1411030, 1602186, 1504545, 1516078,
       1700457, 1512384, 1402355, 1604426, 1615495, 1602453, 1608980,
       1602296, 1403602, 1515420, 1605686, 1508163, 1408018, 1506707,
       1405036, 1404316, 1508005,  153461, 1608839, 1400974, 1500695,
       1504406, 1509767, 1401920, 1505679, 1601789, 1518062, 1522757,
       1423521, 1408272, 1507859, 1410915, 1615914, 1419091, 1602620,
       1406489, 1404309, 1501611, 1516928, 1508362, 1301415, 1614913,
       1505470, 1509056, 1504357, 1523839, 1605918, 1612055, 2134639,
       1607688, 1410116, 1508384, 1503058, 1403144, 1509612, 1404924,
       1613971, 1501049, 1510392, 1602748, 1510261, 1510827, 1605530,
       1513218, 1614786, 1510123, 1601095, 1506777, 1702593, 1406407,
       1508175, 1401942, 1701811, 2154014, 1702746, 1403098, 1405541,
       1315715, 1409660, 1422007, 1604060, 1700572, 1605385, 1400860,
       1501820, 1608463, 1507472, 1511058, 1604044, 1414189, 1507248,
       1315312, 1701475, 1410575])
        names = np.zeros(len(pats))
        names[int(ratio[0] * len(names)):] += 1
        names[int((ratio[0] + ratio[1]) * len(names)):] += 1
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
        for j, pat in enumerate(pats):
            for i in range(len(info)):
                if info.number[i] == pat:
                    label[j] = info.time[i] * (1 if info.event[i] else -1) * 30
                    break
        h5 = h5py.File(h5path, 'w')
        h5.create_dataset('set_pat', data = names)
        h5.create_dataset('pat_fig', data = pat_fig)
        h5.create_dataset('tps', data = tps)
        h5.create_dataset('label', data = label)
        h5.create_dataset('pats', data = pats)
        h5.create_dataset('data', shape = (len(tiffiles), 3, 512, 512))
        h5.create_dataset('postdata', shape = (len(tiffiles), 1024))
        for i, tif in enumerate(tiffiles):
            img = cv2.imread(tif)[:, :, ::-1].transpose((2, 0, 1)) / 255
            with torch.no_grad():
                net = torch.load("/home/tongxueqing/zhao/ImageProcessing/stain_classification/_models/success.Nov.02_22:27.model").module
                net.classifier = torch.nn.Identity()
                net = net.cuda()
                yhat = net(torch.FloatTensor(img[np.newaxis, :, :, :]).cuda())
                h5['postdata'][i, :] = yhat.cpu()
            h5['data'][i, :, :, :] = img
        h5.close()

    def load(self, batch_size, mission):
        Dataset = ClassifierDataset if mission == 'classifier' else SurvDataset
        datasets = {name: Dataset(i, self.set_pat, self.pat_fig, self.data, self.tps) for i, name in enumerate(self.names)}
        loaders = {name: DataLoader(datasets[name], batch_size = batch_size if name == 'train' else 1, shuffle = name == 'train') for name in self.names}
        return loaders

d = Data("/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/compiled.h5", "/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/merged.csv", "/home/tongxueqing/zhao/ImageProcessing/stain_classification/_data/subsets")

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
