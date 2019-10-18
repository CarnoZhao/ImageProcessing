import cv2
import os
import sys
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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

def load_data():
    path = "/home/tongxueqing/data/zhaox/stain_classification/cutted/"
    trainfiles = [l.strip() for l in open(fileidxpath + ".train")]
    testfiles = [l.strip() for l in open(fileidxpath + ".test")]
    traindataset = MyDataset(trainfiles, path)
    testdataset = MyDataset(testfiles, path)
    trainLoader = torch.utils.data.DataLoader(traindataset)
    testLoader = torch.utils.data.DataLoader(testdataset)
    return trainLoader, testLoader

def predict(net, loader, K):
    Y = np.zeros((K, len(loader)))
    Yhat = np.zeros((K, len(loader)))
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to('cuda', dtype = torch.float)
            Y[y, i] = 1
            yhat = torch.softmax(net(x), dim = 1).cpu()
            Yhat[:, i] = yhat
    fpr, tpr, thresholds = metrics.roc_curve(Y.ravel(),Yhat.ravel())
    auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, c = 'r', lw = 2, alpha = 0.7, label = u'AUC=%.3f' % auc)
    plt.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    plt.xlim((-0.01, 1.02))
    plt.ylim((-0.01, 1.02))
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(b = True, ls = ':')
    plt.legend()
    plt.savefig("/home/tongxueqing/zhao/ImageProcessing/stain_classification/auc.png")
    return Y, Yhat
    
def plot_roc_auc(trY, trYhat, tsY, tsYhat):
    trfpr, trtpr, _ = metrics.roc_curve(trY.ravel(),trYhat.ravel())
    tsfpr, tstpr, _ = metrics.roc_curve(tsY.ravel(),tsYhat.ravel())
    trauc = metrics.auc(trfpr, trtpr)
    tsauc = metrics.auc(tsfpr, tstpr)
    fig, ax = plt.subplots()
    ax.plot(trfpr, trtpr, c = 'red', lw = 1, alpha = 0.7, label = u'AUC=%.3f' % trauc)
    ax.plot(tsfpr, tstpr, c = 'green', lw = 1, alpha = 0.7, label = u'AUC=%.3f' % tsauc)
    ax.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    ax.set_xlim((-0.01, 1.02))
    ax.set_ylim((-0.01, 1.02))
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(b = True, ls = ':')
    plt.legend()
    plt.savefig(plotfile)

def main():
    K = 4
    trainLoader, testLoader = load_data()
    net = torch.load(modelpath)
    trY, trYhat = predict(net, trainLoader, K)
    tsY, tsYhat = predict(net, testLoader, K)
    plot_roc_auc(trY, trYhat, tsY, tsYhat)

global modelpath
global fileidxpath
global plotfile
modelpath = sys.argv[1]
fileidxpath = sys.argv[2]
plotfile = sys.argv[3]
main()
