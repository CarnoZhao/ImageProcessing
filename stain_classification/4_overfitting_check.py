import scipy.io as sio
import numpy as np
from collections import Counter
import torch
import functions
import cv2
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

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

def tsne_fit(a, b):
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    data = sio.loadmat('/wangshuo/zhaox/val.mat')
    Yhat = data['Yhat']
    Y = data['Y']
    Yhat = Yhat[np.bitwise_or(Y[0] == a, Y[0] == b), np.array([False if i not in (a, b) else True for i in range(4)])]
    Y = Y[0][np.bitwise_or(Y[0] == a, Y[0] == b)]
    X_tsne = tsne.fit_transform(Yhat)
    X_norm = X_tsne
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne - x_min) / (x_max - x_min)
    fig, ax = plt.subplots()
    ax.scatter(X_norm[:, 0], X_norm[:, 1], color = [plt.cm.Set1(yi) for yi in Y])
    plt.legend()
    plt.savefig('./tsne.png')


def cnter(matpath):
    data = sio.loadmat(matpath)
    for name in data['names']:
        name = name.strip()
        print(name + ":")
        Y = np.argmax(data[name + 'Y'], axis = 1)
        Yhat = np.argmax(data[name + 'Yhat'], axis = 1)
        cnt = np.zeros((4, 4))
        for yi, yih in zip(Y, Yhat):
            cnt[yi, yih] += 1
        print(cnt)

if __name__ == '__main__':
    cnter("/wangshuo/zhaox/ImageProcessing/stain_classification/_mat/success.Oct.24_11:41.mat")
