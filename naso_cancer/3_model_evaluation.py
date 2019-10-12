import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def main():
    modelpath = '/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/models/121.model'
    net = torch.load(modelpath)
    