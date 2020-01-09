import torch
import torchvision
import h5py
import numpy as np
import os
from lifelines.utils import concordance_index
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

h = h5py.File("/wangshuo/zhaox/ImageProcessing/gene_association/_data/sliced.h5", 'r')
net = torch.load("/wangshuo/zhaox/ImageProcessing/survival_analysis/_models/success.Jan.06_10:53.model")
net = net.module.cuda()
net.eval()

data = h['data']

sig = np.zeros(len(data))
for i in range(len(data)):
    img = torch.FloatTensor(data[i:i + 1, :, :, :]).cuda()
    res = net(img)
    sig[i] = res

name = h['name'][:]

uniname = np.array(list(set(name)))
unisig = np.zeros(len(uniname))

for i, one in enumerate(uniname):
    valid = name == one
    res = np.max(sig[valid])
    unisig[i] = res

for i in range(len(unisig)):
    print("%s,%s" % (uniname[i].decode(), str(unisig[i])))
# saved in /wangshuo/zhaox/ImageProcessing/gene_association/_data/DP.sig.txt

def test():
    testh = h5py.File("/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/compiled.h5", 'r')
    data = testh['data']
    pats = testh['pats']
    patfig = testh['pat_fig']
    testsig = np.zeros(len(pats))
    for i, pat in enumerate(pats):
        reses = []
        for j in np.arange(len(patfig[i]))[patfig[i]]:
            img = torch.FloatTensor(data[j:j + 1, :, :, :]).cuda()
            res = net(img).cpu().data.numpy()
            reses.append(res[0, 0])
        testsig[i] = max(reses)
    label = testh['label'][:]
    ci = concordance_index(np.abs(label), -testsig, np.where(label > 0, 1, 0))
    # 0.7364257071637585


