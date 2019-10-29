import os
import cv2
import numpy as np
import torch
import h5py

def compute(datapath, modelpath):
    tpdic ={
        'huaisi': 0,
        'jizhi': 1,
        'tumor': 2,
        'tumorln': 3
    }
    data = []
    filenames = []
    net = torch.load(modelpath)
    res = next(net.children())
    res.fc = torch.nn.Identity()
    with torch.no_grad():
        for name in os.listdir(datapath):
            nap = os.path.join(datapath, name)
            for tp in os.listdir(nap):
                tpp = os.path.join(nap, tp)
                for f in os.listdir(tpp):
                    fp = os.path.join(tpp, f)
                    x = torch.FloatTensor(cv2.imread(fp)[np.newaxis, :, :, ::-1].transpose((0, 3, 1, 2)) / 255)
                    yhat = net(x.cuda()).cpu().numpy()[0]
                    data.append(yhat)
                    filenames.append(f)
    data = np.array(data)
    patnum = [eval(f.split('_')[0]) for f in filenames]
    tps = [tpdic[f.split('_')[1]] for f in filenames]
    hos = [1 if f.split('_')[2] == 'ZF' else 0 for f in filenames]
    return data, patnum, tps, hos

def save(data, patnum, tps, hos, outfile):
    h5 = h5py.File(outfile, 'w')
    h5.create_dataset('data', data = data)
    h5.create_dataset('patnum', data = patnum)
    h5.create_dataset('types', data= tps)
    h5.create_dataset('hosi', data = hos)

if __name__ == "__main__":
    modelpath = "/wangshuo/zhaox/ImageProcessing/stain_classification/_models/success.Oct.27_14:40.model"
    datapath = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/subsets"
    outfile = "/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/prestore.h5"
    data, patnum, tps, hos = compute(datapath, modelpath)
    save(data, patnum, tps, hos, outfile)
