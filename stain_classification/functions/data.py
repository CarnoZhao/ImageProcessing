import os
import torch
import torchvision
from collections import Counter
from torchvision.transforms import *
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, BatchSampler, DataLoader
import numpy as np

__all__ = ['basicLoader', 'forceRatioLoader']

class RandomNoise(object):
    def __init__(self, mean = 0, sigma = 1,p = 0.5):
        self.mean = mean
        self.sigma = sigma
        self.p = p
    
    def __call__(self, img):
        if np.random.random() < p:
            img += np.random.rand(**img.shape) * self.sigma + self.mean
            img = np.where(img > 255, 255, img)
            img = np.where(img < 0, 0, img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + 'noised with mean = %.2f and sigma = %.2f' % (self.mean, self.sigma)

def forceRatioLoader(path, batch_size, num_batch, type_weight):
    dsets = ['train', 'test', 'val']
    train_transform = Compose([
        RandomHorizontalFlip(p = 0.5),
        RandomVerticalFlip(p = 0.5),
        RandomRotation(degrees = 180),
        RandomNoise(p = 0.5),
        ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    ])
    image_datasets = {name: ImageFolder(os.path.join(path, name), transform = ToTensor()) for name in dsets}
    traintypes = [os.path.basename(filename[0]).split('_')[1] for filename in image_datasets['train'].samples]
    typecnter = Counter(traintypes)
    weights = [type_weight[traintype] / typecnter[traintype] for traintype in traintypes]
    trainsampler = BatchSampler(WeightedRandomSampler(weights, num_batch * batch_size,  replacement = True), batch_size = batch_size, drop_last = True)
    trainloader = DataLoader(image_datasets['train'], batch_sampler = trainsampler)
    valloader = DataLoader(image_datasets['val'], shuffle = True, batch_size = batch_size)
    testloader = DataLoader(image_datasets['test'], shuffle = True)
    return {'train': trainloader, 'val': valloader, 'test': testloader}
    
if __name__ == "__main__":
    path = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/cutted"
    batch_size = 32
    num_batch = 200
    type_weight = {'jizhi': 1.5, 'tumor': 1.5, 'tumorln': 1, 'huaisi': 1}
    loaders = forceRatioLoader(path, batch_size, num_batch, type_weight)