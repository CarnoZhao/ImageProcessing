import h5py
import torch
import os 

if os.path.exists("/wangshuo/zhaox"):
    root = "/wangshuo/zhaox"
else:
    root = "/home/tongxueqing/zhao"

h = h5py.File(os.path.join(root, "ImageProcessing/survival_analysis/_data/compiled.h5"), "r")
data = h['data']

net = torch.load(os.path.join(root, "ImageProcessing/survival_analysis/_data/PRENET.model"))
net = torch.load(os.path.join(root, "ImageProcessing/stain_classification/_models/fail.Jan.03_23:32.model")) # 1.0, 0.55, 0.31
net = net.module.cuda()
net.fc = torch.nn.Identity()

post = h['postdata']

net.eval()
for i in range(len(data)):
    x = data[i:i + 1]
    x = torch.FloatTensor(x).cuda()
    yhat = net(x)[0].cpu().data.numpy()
    post[i] = yhat

preds = []
for i in range(len(data)):
    x = data[i:i + 1]
    x = torch.FloatTensor(x).cuda()
    yhat = net(x)
    yhat = int(torch.argmax(yhat).cpu())
    preds.append(yhat)

import numpy as np
preds = np.array(preds)
tps = h['tps'][:]
pats = h["pats"][:]
set_pat = h["set_pat"][:]
pat_fig = h["pat_fig"][:]

np.mean((tps == preds)[np.sum(pat_fig[set_pat == 0], axis = 0) == 1])

h.close()

net = torch.load(os.path.join(root, "ImageProcessing/survival_analysis/_models/success.Jan.04_11:52.model"))
net = net.module.cuda()
net.eval()
preds = []
pats = h["pats"][:]
for i in range(len(h["pats"])):
    patfig = h["pat_fig"][i]
    figs = data[patfig, :, :, :]
    yhats = []
    for j in range(len(figs)):
        fig = figs[j:j + 1]
        yhatj = float(net(torch.FloatTensor(fig).cuda()).cpu().data[0])
        yhats.append(yhatj)
    preds.append(max(yhats))

labels = h["label"][:]
from lifelines.utils import concordance_index
import numpy as np
preds = np.array(preds)

import pandas as pd
info = pd.read_csv(os.path.join(root, "ImageProcessing/combine_model/_data/ClinicMessageForAnalysis.csv"))
csv = pd.read_csv(os.path.join(root, "ImageProcessing/combine_model/_data/preds.csv"))
for i, pat in enumerate(pats):
    name = int(info["name"][info["number"] == pat])
    idx = list(csv["name"]).index(name)
    csv.loc[idx, "sig_deep"] = preds[i]
csv.to_csv(os.path.join(root, "ImageProcessing/combine_model/_data/preds.csv"))

tr = h["set_pat"][:] == 0
ts = h["set_pat"][:] == 1
labelsi = labels[tr]
predsi = preds[tr]
ci = concordance_index(np.abs(labelsi), -predsi, np.where(labelsi > 0, 1, 0))