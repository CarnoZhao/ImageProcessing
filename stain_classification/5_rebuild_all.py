import os
import sys
import cv2
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


class Data(object):
    def __init__(self, path, batch_size, type_weights=None, ignore_classes=[]):
        self.path = path
        self.setnames = ['train', 'test']
        self.type_weights = type_weights
        self.batch_size = batch_size
        self.ignore_classes = ignore_classes

    class _RandomNoise(object):
        def __init__(self, mean=0.0, sigma=1.0, p=0.5):
            self.mean = mean
            self.sigma = sigma
            self.p = p

        def __call__(self, img):
            if np.random.random() < self.p:
                img = img + \
                    np.random.randn(*img.size, 3) * self.sigma + self.mean
                img = np.minimum(255, np.maximum(0, img))
                img = Image.fromarray(np.uint8(img))
            return img

    class _GaussianBlur(object):
        def __init__(self, sigma=1.0, H=5, W=5, p=0.5):
            self.sigma = sigma
            self.H = H
            self.W = W
            self.p = p

        def __call__(self, img):
            if np.random.random() < self.p:
                img = np.array(img)
                kernelx = cv2.getGaussianKernel(self.W, self.sigma, cv2.CV_32F)
                kernelx = np.transpose(kernelx)
                kernely = cv2.getGaussianKernel(self.H, self.sigma, cv2.CV_32F)
                for i in range(3):
                    img[:, :, i] = signal.convolve2d(
                        img[:, :, i], kernelx, mode='same', boundary='fill', fillvalue=0)
                    img[:, :, i] = signal.convolve2d(
                        img[:, :, i], kernely, mode='same', boundary='fill', fillvalue=0)
                img = Image.fromarray(np.uint8(img))
            return img

    def _pseudo_path(self, path, name):
        combinepath = os.path.join(path, name)
        if self.ignore_classes:
            classes = os.listdir(combinepath)
            time = os.path.basename(modelpath)[:-6]
            tmpdir = os.path.join("/wangshuo/zhaox/.tmp", time, name)
            os.system('mkdir -p %s' % tmpdir)
            classes = [cl for cl in classes if cl not in self.ignore_classes]
            for cl in classes:
                os.system('ln %s %s -s' %
                          (os.path.join(combinepath, cl), os.path.join(tmpdir, cl)))
            combinepath = tmpdir
        return combinepath

    def load(self):
        image_datasets = {name: ImageFolder(self._pseudo_path(
            self.path, name), transform=ToTensor()) for name in self.setnames}
        trainloader = DataLoader(
            image_datasets['train'], shuffle=True, batch_size=self.batch_size)
        testloader = DataLoader(image_datasets['test'], shuffle=True)
        return {'train': trainloader, 'test': testloader}


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


class Prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        X, Y = self.next_input, self.next_target
        self.preload()
        return [X, Y]


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
            net = torchvision.models.resnet34(num_classes=self.K)
        net = torch.nn.DataParallel(net, device_ids=self.gpus)
        net = net.cuda()
        return net


    def _call_back(self, i, net, loaders, costs, loss, subsample):
        if i % self.step == 0:
            testcosts = 0
            with torch.no_grad():
                J = len(loaders['test'])
                for j, (x, y) in enumerate(loaders['test']):
                    net.eval()
                    x = x.cuda(); y = y.cuda()
                    testcosts += float(loss(net(x), y))
                    if j == int(J * subsample):
                        break
            print_to_out("%d:\ttrcost: %.6f\ttscost: %.6f" % (i, costs, testcosts))

    def train(self):
        loaders = Data(self.path, self.batch_size,
                       self.type_weights, self.ignore_classes).load()
        loader = loaders['train']
        net = self._load_net()
        loss = Loss(self.K, self.smoothing, self.gamma)
        opt = torch.optim.Adamax(net.parameters(), lr=self.lr)
        for i in range(1, self.iters + 1):
            if i % 30 == 0:
                self.lr /= 10
                opt = torch.optim.Adamax(net.parameters(), lr=self.lr)
            costs = 0
            J = len(loader)
            net.train()
            for j, (x, y) in enumerate(loader):
                x = x.cuda(); y = y.cuda()
                opt.zero_grad()
                cost = loss(net(x), y)
                cost.backward()
                costs += float(cost)
                opt.step()
                if j == int(J * self.subsample):
                    break
            self._call_back(i, net, loaders, costs, loss, self.subsample)
        torch.save(net, modelpath)
        net.eval()
        Evaluation().printplot(net, loaders, self.K, self.subsample)
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
             "iters":    10,
                 "K":    4,
          "pretrain":    False,
                "lr":    0.00001,
        "batch_size":    32,
      "loss_weights":    None,
             "gamma":    0,
         "smoothing":    0.001,
              "step":    1,
         "subsample":    0.3,
    "ignore_classes":    [],
              "gpus":    [0, 1, 2]
}

if __name__ == "__main__":
    net = Train(**params).train()
