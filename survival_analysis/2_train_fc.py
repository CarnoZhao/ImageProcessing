import sys
import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

class SurvLoss(torch.nn.Module):
    def __init__(self):
        super(SurvLoss, self).__init__()
        pass

    def forward(self, Yhat, Y):
        Yhat = torch.squeeze(Yhat)
        Y = torch.squeeze(Y)
        order = torch.argsort(Y)
        Y = Y[order]
        Yhat = Yhat[order]
        E = (Y > 0).float()
        T = torch.abs(Y)
        Yhr = torch.log(torch.cumsum(torch.exp(Yhat), dim = 0))
        obs = torch.sum(E)
        Es = torch.zeros_like(E)
        Yhrs = torch.zeros_like(Yhr)
        j = 0
        for i in range(1, len(T)):
            if T[i] != T[i - 1]:
                Es[i - 1] = torch.sum(E[j:i])
                Yhrs[i - 1] = torch.max(Yhr[j:i])
                j = i
        Es[-1] = torch.sum(E[j:])
        Yhrs[-1] = torch.max(Yhr[j:])
        loss2 = torch.sum(torch.mul(Es, Yhrs))
        loss1 = torch.sum(torch.mul(Yhat, E))
        loss = torch.div(torch.sub(loss2, loss1), obs)
        return loss
        

class Datasets(Dataset):
    def __init__(self, pats, label, hosi, istrain):
        super(Datasets, self).__init__()
        if istrain:
            self.pats = [p for p in pats if p in hosi['ZF']]
            self.label = [label[i] for i in range(len(label)) if pats[i] in hosi['ZF']]
        else:
            self.pats = [p for p in pats if p in hosi['GX']]
            self.label = [label[i] for i in range(len(label)) if pats[i] in hosi['GX']]
        self.length = len(self.pats)

    def __getitem__(self, i):
        return self.pats[i], self.label[i]

    def __len__(self):
        return self.length

class Net(torch.nn.Module):
    def __init__(self, layers, p):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 512, out_features = layers[0], bias = True)
        self.dr1 = torch.nn.Dropout(p)
        self.fc2 = torch.nn.Linear(in_features = layers[0], out_features = 1, bias = True)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.Tanh()(x)
        x = self.dr1(x)
        x = self.fc2(x)
        x = torch.nn.Tanh()(x)
        return x

def print_to_out(*args):
    with open(outfile, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

def load_data(h5path, csvpath):
    h5 = h5py.File(h5path, 'r')
    csv = pd.read_csv(csvpath)
    num_to_label = {csv.number[i]: csv.time[i] * 30 * (1 if csv.event[i] else -1) for i in range(len(csv))}
    pats = np.array(list(set(csv.number) & set(h5['patnum'])))
    label = np.array([num_to_label[num] for num in pats])
    data = {}
    for num in pats:
        data[num] = h5['data'][np.equal(h5['patnum'][:], num), :]
    hosi = {h: np.array(csv.number[np.equal(csv.hosi, h)]) for h in ['ZF', 'GX']}
    return pats, label, data, hosi

def main(h5path, csvpath, lr, epochs, batch_size):
    
    ## load_data
    pats, label, data, hosi = load_data(h5path, csvpath)

    ## laoder
    traindataset = Datasets(pats, label, hosi, True)
    trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True)
    # trainloader = DataLoader(traindataset, batch_size = len(traindataset), shuffle = True)
    testdataset = Datasets(pats, label, hosi, False)
    testloader = DataLoader(testdataset)

    ## pre parameter
    net = Net([128], 0).cuda()
    loss = SurvLoss()
    # opt = torch.optim.Adamax(net.parameters(), lr = lr)
    opt = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9, nesterov = True)

    ## iteration
    for i in range(epochs):
        costs = 0
        for pats, y in trainloader:
            x = torch.zeros((len(pats), 512)).cuda()
            y = y.cuda()

            # select instance
            net.eval()
            for j, p in enumerate(pats):
                pdata = data[int(p)]
                pyhat = net(torch.FloatTensor(pdata).cuda())
                maxidx = torch.argmax(pyhat)
                x[j] = torch.Tensor(pdata[maxidx])

            # train
            net.train()
            yhat = net(x)
            opt.zero_grad()
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            opt.step()
        if i % 10 == 0:          
            print_to_out("%d | costs: %.3f" % (i, costs))

    ## evalu
    with torch.no_grad():
        for loader in [trainloader, testloader]:
            Y = np.zeros(len(loader.dataset))
            Yhat = np.zeros(len(loader.dataset))
            i = 0
            for pats, y in loader:
                Y[i: i + len(y)] = y
                for p in pats:
                    pdata = data[int(p)]
                    pyhat = net(torch.FloatTensor(pdata).cuda())
                    preds = torch.max(pyhat, dim = 0).values
                    Yhat[i] = float(preds)
                    i += 1
            ci = concordance_index(np.abs(Y), -1 * Yhat, np.sign(Y))
            print_to_out("ci = %.3f" % ci)

    ## save
    torch.save(net, modelpath)
    

if __name__ == "__main__":
    global modelpath; global plotpath; global matpath; global outfile
    modelpath, plotpath, matpath, outfile = sys.argv[1:5]

    params = {
        "h5path": "/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/computed_data.h5",
        "csvpath": "/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/merged.csv",
        "lr": 1e-4,
        "epochs": 100,
        "batch_size": 64,
    }
    main(**params)