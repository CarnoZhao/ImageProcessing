import numpy as np
import os
import sys
import torch
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import functions

def dense_net_model(loader, decay, numIters, lr, preTrain, focal, weighted, smoothing, gamma, K):

    ## Pre-Train ##
    if preTrain:
        net = torchvision.models.densenet121(pretrained = preTrain)
        for p in net.parameters():
            p.requires_grad = False
        for p in net.features.denseblock4.parameters():
            p.requires_grad = True
        net.classifier = torch.nn.Linear(in_features = 1024, out_features = K, bias = True)
    else:
        net = torchvision.models.densenet121(num_classes = K)
    net.to('cuda')

    ## weight ##
    if weighted:
        # jizhi: 1850, tumor: 5798, tumorln: 521, huaisi: 337
        t = np.array([1850, 5798, 521, 337])
        t = (1 / t) / np.sum(1 / t)
        weight = torch.FloatTensor(t).to('cuda')
    else:
        weight = None

    ## loss-opt ##
    # loss = functions.losses.LabelSmoothingFocalLoss(classes = K, smoothing = smoothing, weight = weight, focal = focal, gamma = gamma)
    loss = functions.losses.Loss(K = K, focal = focal, gamma = gamma, smoothing = smoothing, weights = weight)
    optimizer = torch.optim.Adamax(net.parameters(), lr = lr)

    ## iterating ##
    for iteration in range(numIters):
        if decay and iteration % 20 == 0 and iteration != 0:
            lr /= 10
        costs = 0
        for x, y in loader:
            x = x.to('cuda', dtype = torch.float)
            y = y.to('cuda')
            optimizer.zero_grad()
            yhat = net(x)
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            optimizer.step()
        if iteration % 5 == 0:
            print("Cost after iteration %d: %.3f" % (iteration, costs))
    return net

def main(batch_size, num_batch, type_weight, decay, numIters, lr, preTrain, focal, weighted, smoothing, gamma, K):
    path = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/cutted"
    loaders = functions.data.forceRatioLoader(path, batch_size, num_batch, type_weight)
    trainloader = loaders['train']
    net = dense_net_model(trainloader, decay, numIters, lr, preTrain, focal, weighted, smoothing, gamma, K)
    torch.save(net, modelpath)
    functions.evaluation.printplot(net, loaders, K, plotpath, matpath)

global modelpath
global plotpath
global matpath
modelpath = sys.argv[1]
plotpath = sys.argv[2]
matpath = sys.argv[3]

type_weight = {
    "jizhi": 1.5,
    "tumor": 1.5,
    "tumorln": 1,
    "huaisi": 1
    }

params = {
         "numIters":    30,
       "batch_size":    32,
        "num_batch":    200,
         "preTrain":    True,
            "focal":    True,
         "weighted":    False,
            "decay":    True,
               "lr":    0.001,
        "smoothing":    0.01,
            "gamma":    2,
                "K":    4,
      "type_weight":    type_weight,
    }

for key, value in params.items():
    print("{:<10s}{:>10s}".format(key, str(value)))
main(**params)