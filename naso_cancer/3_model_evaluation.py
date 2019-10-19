import torch
import scipy.io as sio
import os
import sys
import numpy as np
sys.path.insert(1, "/home/tongxueqing/zhao/MachineLearning/Python_ML/")
import densenet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main(name):
    matpath = "/home/tongxueqing/zhao/ImageProcessing/naso_cancer/_data/cut_slice/"
    model = torch.load(modelfile)
    matfiles = []
    with open(fileidx + "." + name) as f:
        for line in f:
            matfiles.append(line.strip())
    matfiles = [f for f in matfiles if "data1.mat.0.rotate" in f]
    rawyhats = []
    yhats = []
    ys = []
    for matfile in matfiles:
        data = sio.loadmat(matpath + matfile)['data']
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
    print("%s\t%.3f\t%.3f\t%.3f" % (name, sensi, speci, accu))
    with open(rocfile + ".%s.csv" % name, 'w') as f:
        for i, j in zip(rawyhats, ys):
            f.write(str(i) + ',' + str(j) + '\n')
    return rawyhats, ys

global rocfile
global modelfile
global fileidx
rocfile = sys.argv[1]
modelfile = sys.argv[2]
fileidx = sys.argv[3]
print("\t\tSE\t\tSP\t\tACC")
main('train')
main('test')
    
