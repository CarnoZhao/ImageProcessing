import torch
import scipy.io as sio
import os
import numpy as np

def main():
    modelpath = "/wangshuo/zhaox/ImageProcessing/naso_cancer/_data/models/121.model"
    matpath = "/wangshuo/zhaox/ImageProcessing/naso_cancer/_data/cut_slice/"
    model = torch.load(modelpath)
    matfiles = [f for f in os.listdir(matpath) if "data1.mat" in f]
    yhats = []
    ys = []
    for matfile in matfiles:
        data = sio.loadmat(matpath + matfile)['data'][:1, :, :, :]
        data = np.concatenate([data, np.zeros((1, 1, *data.shape[-2:]))], axis = 1)
        data = torch.FloatTensor(data).to('cuda')
        yhat = model(data).cpu().data.numpy()[0]
        yhat = 1 if yhat[1] > yhat[0] else 0
        y = 1 if matfile.startswith("1") else 0
        yhats.append(yhat)
        ys.append(y)
    yhats = np.array(yhats)
    ys = np.array(ys)
    accu = np.sum(yhats == ys)
    fp = np.sum(np.bitwise_and(yhats == 1, ys == 0))
    tp = np.sum(np.bitwise_and(yhats == 1, ys == 1))
    tn = np.sum(np.bitwise_and(yhats == 0, ys == 0))
    fn = np.sum(np.bitwise_and(yhats == 0, ys == 1))
    sensi = tp / (tp + fn) # 0.905
    speci = tn / (tn + fp) # 0.464
    print("Sensitivity is %.3f" % sensi)
    print("Specificity is %.3f" % speci)
    print("Accuracy is %.3f" % accu)

main()