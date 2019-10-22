import numpy as np
import os
import sys
import torch
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import functions

def dense_net_model(loaders, decay, numIters, lr, preTrain, focal, loss_weights, smoothing, gamma, K):

    ## Pre-Train ##
    loader = loaders['train']
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
    if loss_weights is not None:
        class_to_idx = loader.dataset.class_to_idx
        weights = [0 for _ in range(K)]
        for key in class_to_idx:
            weights[class_to_idx[key]] = loss_weights[key]

    ## loss-opt ##
    # loss = functions.losses.LabelSmoothingFocalLoss(classes = K, smoothing = smoothing, weight = weight, focal = focal, gamma = gamma)
    loss = functions.losses.Loss(K = K, focal = focal, gamma = gamma, smoothing = smoothing, weights = loss_weights)
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
        if iteration % 3 == 0:
            print("Cost after iteration %d: %.3f" % (iteration, costs))
            _, testYhat, testaccu = functions.evaluation.predict(net, loaders['test'], K)
            _, trainYhat, trainaccu = functions.evaluation.predict(net, loader, K)
            print("Accuracy after iteration %d: train %.3f; test %.3f" % (iteration, trainaccu, testaccu))
            print("Yhat sample in test:")
            print(testYhat)
            print("Yhat sample in train:")
            print(trainYhat)
    return net

def main(batch_size, num_batch, type_weight, decay, numIters, lr, preTrain, focal, loss_weights, smoothing, gamma, K):
    path = "/wangshuo/zhaox/ImageProcessing/stain_classification/_data/cutted"
    loaders = functions.data.forceRatioLoader(path, batch_size, num_batch, type_weight)
    net = dense_net_model(loaders, decay, numIters, lr, preTrain, focal, loss_weights, smoothing, gamma, K)
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

loss_weights = {
    "jizhi": 0,
    "tumor": 0,
    "tumorln": 0,
    "huaisi": 0
}

params = {
         "numIters":    20,
       "batch_size":    32,
        "num_batch":    1000,
         "preTrain":    True,
            "focal":    True,
     "loss_weights":    None,
            "decay":    True,
               "lr":    0.00001,
        "smoothing":    0.001,
            "gamma":    2,
                "K":    4,
      "type_weight":    type_weight,
    }

# {'huaisi': 0, 'jizhi': 1, 'tumor': 2, 'tumorln': 3}

for key, value in params.items():
    print("{:<10s}{:>10s}".format(key, str(value)))
main(**params)