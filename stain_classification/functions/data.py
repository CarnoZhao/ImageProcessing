import os
import cv2
import torch
import torchvision
from collections import Counter
from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader
import numpy as np
from scipy import signal
from PIL import Image

__all__ = ['basicLoader', 'forceRatioLoader']

class RandomNoise(object):
    def __init__(self, mean = 0.0, sigma = 5, p = 0.5):
        self.mean = mean
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img):
        if np.random.random() < self.p:
            img = img + np.random.randn(*img.size, 3) * self.sigma + self.mean
            img = np.minimum(255, np.maximum(0, img))
            img = Image.fromarray(np.uint8(img))
        return img

    def __repr__(self):
        return self.__class__.__name__ + 'noised with mean = %.2f and sigma = %.2f' % (self.mean, self.sigma)

class GaussianBlur(object):
    def __init__(self, sigma = 1, H = 7, W = 7, p = 0.5):
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
                img[i] = signal.convolve2d(img[i], kernelx, mode = 'same', boundary = 'fill', fillvalue = 0)
                img[i] = signal.convolve2d(img[i], kernely, mode = 'same', boundary = 'fill', fillvalue = 0)
            img = Image.fromarray(np.uint8(img))
        return img

def forceRatioLoader(path, batch_size, num_batch, type_weight):
    dsets = ['train', 'test', 'val']
    transformer = {
        'train': Compose([
            ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),
            RandomHorizontalFlip(p = 0.5),
            RandomVerticalFlip(p = 0.5),
            RandomRotation(degrees = 180),
            RandomNoise(p = 0.5),
            GaussianBlur(p = 0.5),
            ToTensor()
        ]),
        'test': ToTensor(),
        'val': ToTensor()
    }
    image_datasets = {name: ImageFolder(os.path.join(path, name), transform = transformer[name]) for name in dsets}
    traintypes = [os.path.basename(filename[0]).split('_')[1] for filename in image_datasets['train'].samples]
    typecnter = Counter(traintypes)
    weights = [type_weight[traintype] / typecnter[traintype] for traintype in traintypes]
    trainsampler = BatchSampler(WeightedRandomSampler(weights, num_batch * batch_size, replacement = True), batch_size = batch_size, drop_last = True)
    trainloader = DataLoader(image_datasets['train'], batch_sampler = trainsampler)
    valloader = DataLoader(image_datasets['val'])
    testloader = DataLoader(image_datasets['test'])
    return {'train': trainloader, 'val': valloader, 'test': testloader}
    
if __name__ == "__main__":
    path = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/cutted"
    batch_size = 32
    num_batch = 200
    type_weight = {'jizhi': 1.5, 'tumor': 1.5, 'tumorln': 1, 'huaisi': 1}
    loaders = forceRatioLoader(path, batch_size, num_batch, type_weight)
    t = loaders['train']
    cnt = Counter()
    for x, y in t:
        break