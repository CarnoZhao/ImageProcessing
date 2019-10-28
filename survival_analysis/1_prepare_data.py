import os
import numpy as np
import torch
import h5py

def compute(datapath, modelpath):
    data = []
    filenames = []
    net = torch.load(modelpath)
    with torch.no_grad:
        for name in os.listdir(datapath):
            np = os.path.join(datapath, name)
            for tp in os.listdir(np):
                tpp = os.path.join(np, tp)
                for f in os.listdir(tpp):
                    fp = os.path.join(tpp, f)
                    x = cv2.imread(fp)[:, :, ::-1] / 255
                    yhat = net(x.cuda()).cpu().data.numpy()[0]
                    data.append(yhat)
                    filenames.append(f)
    data = np.array(data)
    return data, filenames

def save(data, filenames, outfile):
    h5 = h5py.File(outfile, 'w')
    h5.create_dataset('data', data = data)
    h5.create_dataset('label', data = filenames)

if __name__ == "__main__":
    modelpath = "/wangshuo/zhaox/ImageProcessing/stain_classification/_models/"
    datapath = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/subsets"
    outfile = "/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/prestore.h5"
    data, filenames = compute(datapath, modelpath)
    save(data, filenames, outfile)
