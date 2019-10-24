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
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Loss(torch.nn.Module):
    def __init__(self, K, smoothing = 0, gamma = 0, weights = 1):
        super(Loss, self).__init__()
        self.K = K
        self.smoothing = smoothing
        self.gamma = gamma
        self.weights = weights

    def _label_smoothing(self, Y, K, smoothing):
        ls = torch.zeros((len(Y), K)).cuda()
        ls.fill_(smoothing / K)
        ls.scatter(1, Y.data.unsqueeze(1), 1 - smoothing)
        return ls
    
    def forward(self, Yhat, Y):
        sm = Yhat.softmax(dim = -1)
        logYhat = Yhat.log_softmax(dim = -1) * (1 - sm) ** self.gamma
        Y = self._label_smoothing(Y, self.K, self.smoothing)
        return torch.mean(torch.sum(-Y * logYhat * self.weights, dim = -1))

class Data(object):
    def __init__(self, path, batch_size, type_weights = None, ignore_classes = []):
        self.path = path
        self.setnames = ['train', 'test']
        self.type_weights = type_weights
        self.batch_size = batch_size
        self.ignore_classes = ignore_classes

    class _RandomNoise(object):
        def __init__(self, mean = 0.0, sigma = 1.0, p = 0.5):
            self.mean = mean
            self.sigma = sigma
            self.p = p

        def __call__(self, img):
            if np.random.random() < self.p:
                img = img + np.random.randn(*img.size, 3) * self.sigma + self.mean
                img = np.minimum(255, np.maximum(0, img))
                img = Image.fromarray(np.uint8(img))
            return img

    class _GaussianBlur(object):
        def __init__(self, sigma = 1.0, H = 5, W = 5, p = 0.5):
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
                    img[:, :, i] = signal.convolve2d(img[:, :, i], kernelx, mode = 'same', boundary = 'fill', fillvalue = 0)
                    img[:, :, i] = signal.convolve2d(img[:, :, i], kernely, mode = 'same', boundary = 'fill', fillvalue = 0)
                img = Image.fromarray(np.uint8(img))
            return img

    def _pseudo_path(self, path, name):
        combinepath = os.path.join(path, name)
        if self.ignore_classes:
            classes = os.listdir(combinepath)
            time = os.path.basename(modelpath)[:-6]
            tmpdir = os.path.join("/wangshuo/zhaox/.tmp", time, name)
            os.system('mkdir -p %s' % tmpdir)
            # os.system("rm %s -rf" % os.path.join(tmpdir, '*'))
            classes = [cl for cl in classes if cl not in self.ignore_classes]
            for cl in classes:
                os.system('ln %s %s -s' % (os.path.join(combinepath, cl), os.path.join(tmpdir, cl)))
            combinepath = tmpdir
        return combinepath

    def load(self):
        image_datasets = {name: ImageFolder(self._pseudo_path(self.path, name), transform = ToTensor()) for name in self.setnames}
        # traintypes = [os.path.basename(filename[0]).split('_')[1] for filename in image_datasets['train'].samples]
        # typecnter = Counter(traintypes)
        # weights = [self.type_weights[traintype] / typecnter[traintype] for traintype in traintypes]
        # trainsampler = BatchSampler(WeightedRandomSampler(weights, self.num_batch * self.batch_size, replacement = True), batch_size = self.batch_size, drop_last = True)
        trainloader = DataLoader(image_datasets['train'], shuffle = True, batch_size = self.batch_size)
        testloader = DataLoader(image_datasets['test'])
        return {'train': trainloader, 'test': testloader}
    
class Evaluation(object):
    def __init__(self):
        pass

    def predict(self, net, loader, K):
        if loader.batch_sampler:
            length = sum([len(batch) for batch in list(loader.batch_sampler)])
        else:
            length = len(loader) * loader.batch_size 
        Y = np.zeros((length, K))
        Yhat = np.zeros((length, K))
        with torch.no_grad():
            i = 0
            for x, y in loader:
                x = x.to('cuda', dtype = torch.float)
                Y[i:i + len(y), :] = torch.nn.functional.one_hot(y, num_classes = K)
                yhat = torch.softmax(net(x), dim = 1).cpu()
                Yhat[i:i + len(y), :] = yhat
                i += len(y)
        accu = np.mean(np.equal(np.argmax(Y, axis = 1), np.argmax(Yhat, axis = 1)))
        return Y, Yhat, accu

    def plot_roc_auc(self, data):
        fig, ax = plt.subplots()
        colors = ['r', 'g', 'b']
        for i, name in enumerate(data['names']):
            Y = data[name + 'Y']
            Yhat = data[name + 'Yhat']
            fpr, tpr, _ = metrics.roc_curve(Y.ravel(), Yhat.ravel())
            auc = metrics.auc(fpr, tpr)
            ax.plot(fpr, tpr, c = colors[i], lw = 1, alpha = 0.7, label = u'%sAUC=%.3f' % (name, auc))
        ax.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
        ax.set_xlim((-0.01, 1.02))
        ax.set_ylim((-0.01, 1.02))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.grid(b = True, ls = ':')
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
            print('Accuracy in %s = %.6f' % (name, accus[name]))

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
            self.next_input = self.next_input.cuda(non_blocking = True)
            self.next_target = self.next_target.cuda(non_blocking = True)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        X, Y = self.next_input, self.next_target
        self.preload()
        return [X, Y]

class Train(object):
    def __init__(self, path, iters, K, pretrain, lr, batch_size, type_weights, loss_weights = None, gamma = 0, smoothing = 0, step = 3, ignore_classes = []):
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

    def _load_net(self):
        if self.pretrain:
            net = torchvision.models.densenet121(pretrained = True)
            # for p in net.parameters():
            #     p.requires_grad = False
            # for p in net.features.denseblock4.parameters():
            #     p.requires_grad = True
            net.classifier = torch.nn.Linear(in_features = 1024, out_features = self.K, bias = True)
        else:
            net = torchvision.models.densenet121(num_classes = self.K)
        # net = torch.nn.DataParallel(net, device_ids=[1, 2])
        net = net.cuda()
        return net

    def _load_weights(self, loader):
        if self.loss_weights is not None:
            class2idx = loader.dataset.class_to_idx
            weights = [0 for _ in range(self.K)]
            for key in class2idx:
                weights[class2idx[key]] = self.type_weights[key]
        else:
            weights = 1
        return weights

    def _load_loss_opt(self, net, weights):
        loss = Loss(self.K, self.smoothing, self.gamma, weights)
        optimizer = torch.optim.Adamax(net.parameters(), lr = self.lr)
        return loss, optimizer

    def _call_back(self, i, net, loaders, costs):
        if i % self.step == 0:
            print("Costs after iter %d: %.3f" % (i, costs))
            # _, testYhat, testaccu = Evaluation().predict(net, loaders['test'], self.K)
            # _, trainYhat, trainaccu = Evaluation().predict(net, loaders['train'], self.K)
            # print("Accuracy after iteration %d: train %.3f; test %.3f" % (i, trainaccu, testaccu))
            # print("Yhat sample in test:")
            # print(testYhat)
            # print("Yhat sample in train:")
            # print(trainYhat)

    def train(self):
        loaders = Data(self.path, self.batch_size, self.type_weights, self.ignore_classes).load()
        loader = loaders['train']
        net = self._load_net()
        weights = self._load_weights(loader)
        loss, opt = self._load_loss_opt(net, weights)
        for i in range(1, self.iters + 1):
            costs = 0
            for x, y in loader:
                x = x.cuda(); y = y.cuda()
                opt.zero_grad()
                cost = loss(net(x), y)
                cost.backward()
                costs += float(cost)
                opt.step()
            self._call_back(i, net, loaders, costs)
        torch.save(net, modelpath)
        Evaluation().printplot(net, loaders, self.K)
        return net

global modelpath
global plotpath
global matpath
modelpath = sys.argv[1]
plotpath = sys.argv[2]
matpath = sys.argv[3]      

type_weights = {
    "jizhi": 1.5,
    "tumor": 1.5,
    "tumorln": 1,
    "huaisi": 1
    }

loss_weights = {
    "jizhi": 0,
    "tumor": 0,
    "tumorln": 0,
    "huaisi": 0
}

params = {
         "path": "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/imagenet_normed",
        "iters":    50,
            "K":    4,
     "pretrain":    True,
           "lr":    0.0001,
   "batch_size":    16,
 "type_weights":    type_weights,
 "loss_weights":    None,
        "gamma":    0,
    "smoothing":    0.001,
         "step":    10,
"ignore_classes":   []
}

if __name__ == "__main__":
    net = Train(**params).train()