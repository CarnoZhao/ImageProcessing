import torch
import scipy.io as sio
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(name):
    modelpath = "/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/models/121.model"
    matpath = "/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/cut_slice/"
    model = torch.load(modelpath)
    matfiles = []
    with open("/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/fileidx/%s.files" % name) as f:
        for line in f:
            matfiles.append(line.strip())
    matfiles = [f for f in matfiles if "data1.mat.0.rotate" in f]
    rawyhats = []
    yhats = []
    ys = []
    for matfile in matfiles:
        data = sio.loadmat(matpath + matfile)['data']
        data = np.concatenate([data, np.zeros((1, *data.shape[-2:]))], axis = 0)
        data = data[np.newaxis, :, :, :]
        with torch.no_grad():
            data = torch.FloatTensor(data).to('cuda')
            yhat = model(data).cpu().data.numpy()[0]
        yhat = np.exp(yhat) / np.sum(np.exp(yhat))
        newyhat = 1 if yhat[1] > yhat[0] else 0
        y = 1 if matfile.startswith("1") else 0
        rawyhats.append(yhat[1])
        yhats.append(newyhat)
        ys.append(y)
    yhats = np.array(yhats)
    ys = np.array(ys)
    accu = np.mean(yhats == ys)
    fp = np.sum(np.bitwise_and(yhats == 1, ys == 0))
    tp = np.sum(np.bitwise_and(yhats == 1, ys == 1))
    tn = np.sum(np.bitwise_and(yhats == 0, ys == 0))
    fn = np.sum(np.bitwise_and(yhats == 0, ys == 1))
    sensi = tp / (tp + fn) # 0.905
    speci = tn / (tn + fp) # 0.464
    print("Sensitivity in %s is %.3f" % (name, sensi))
    print("Specificity in %s is %.3f" % (name, speci))
    print("Accuracy in %s is %.3f" % (name, accu))
    with open('/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/roc/121model.%s.roc.csv' % name, 'w') as f:
        for i, j in zip(rawyhats, ys):
            f.write(str(i) + ',' + str(j) + '\n')
    return rawyhats, ys

main('train')
main('test')
    