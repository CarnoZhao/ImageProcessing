import scipy.io as sio
import numpy as np
from collections import Counter
import torch
import functions
import cv2
import os

def yhat_distribution():
    path = "/wangshuo/zhaox/ImageProcessing/stain_classification/_mat/success.Oct.21_13:42.mat"
    data = sio.loadmat(path)
    for name in data['names']:
        name = name.strip()
        if name == "val":
            continue
        print("In %s:" % name)
        Y = data[name + "Y"]
        Yhat = data[name + "Yhat"]
        argYhat = np.argmax(Yhat, axis = 1)
        argY = np.argmax(Y, axis = 1)
        print("Y = ")
        print(Y)
        print("Yhat = ")
        print(np.round(Yhat, decimals = 2))
        print()
    Yhat = data['trainYhat']

def img2tensor(path):
    img = cv2.imread(path)
    img = img / 255
    img = img.transpose((2, 0, 1))
    img = img[np.newaxis, :, :, :]
    img = torch.FloatTensor(img)
    img = img.cuda()
    return img

def manual_accuracy():
    modelpath = "/wangshuo/zhaox/ImageProcessing/stain_classification/_models/success.Oct.21_16:09.model"
    net = torch.load(modelpath)
    Yhat = []
    for filename in os.listdir("/wangshuo/zhaox/ImageProcessing/stain_classification/_data/cutted/train/tumorln/"):
        filename = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/cutted/train/tumorln/" + filename
        train = img2tensor(filename)
        Yhat.append(net(train).cpu().data.numpy()[0])
    Yhat = np.array(Yhat)

if __name__ == '__main__':
    manual_accuracy()
