import sys
import h5py
import torch
import numpy as np
import pandas as pd
from itertools import product
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
    def __init__(self, pats, label, hosi, istrain, K = None, k = None):
        super(Datasets, self).__init__()
        if istrain == 'train':
            self.pats = [p for i, p in enumerate(pats) if p in hosi['ZF'] and not self.__val_check(i, k, K, len(pats))]
            self.label = [label[i] for i in range(len(label)) if pats[i] in hosi['ZF'] and not self.__val_check(i, k, K, len(pats))]
        elif istrain == 'val':
            self.pats = [p for i, p in enumerate(pats) if p in hosi['ZF'] and self.__val_check(i, k, K, len(pats))]
            self.label = [label[i] for i in range(len(label)) if pats[i] in hosi['ZF'] and self.__val_check(i, k, K, len(pats))]
        else:
            self.pats = [p for p in pats if p in hosi['GX']]
            self.label = [label[i] for i in range(len(label)) if pats[i] in hosi['GX']]
        self.length = len(self.pats)

    def __val_check(self, i, k, K, length):
        return int(k * length / K) <= i < int((k + 1) * length / K)

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
    order = list(range(len(pats)))
    np.random.shuffle(order)
    pats = pats[order]
    label = label[order]
    data = {}
    for num in pats:
        data[num] = h5['data'][np.equal(h5['patnum'][:], num), :]
    hosi = {h: np.array(csv.number[np.equal(csv.hosi, h)]) for h in ['ZF', 'GX']}
    return pats, label, data, hosi

def call_back(i, step, net, data, loaders):
    if i % step != 0:
        return
    net.eval()
    out = "%d" % i
    ret = []
    with torch.no_grad():
        for name, loader in loaders.items():
            Y = np.zeros(len(loader.dataset))
            Yhat = np.zeros(len(loader.dataset))
            i = 0
            for pats, y in loader:
                Y[i: i + len(y)] = y
                for pat in pats:
                    pdata = data[int(pat)]
                    pyhat = net(torch.FloatTensor(pdata).cuda())
                    preds = torch.max(pyhat, dim = 0).values
                    Yhat[i] = float(preds)
                    i += 1
            ci = concordance_index(np.abs(Y), -1 * Yhat, np.sign(Y))
            out += " | %s: %.3f" % (name, ci)
            ret.append(ci)
    # print_to_out(out)
    return ret

def main(h5path, csvpath, p, lr, l, K, epochs, batch_size, step, weight_decay):
    
    ## load_data
    print_to_out("lr : %.3e" % lr)
    print_to_out("dropout_p: %.3f" % p)
    print_to_out("mid_layer: %d" % l)
    print_to_out("weight_decay: %.3e" % weight_decay)
    pats, label, data, hosi = load_data(h5path, csvpath)

    ## laoder
    testdataset = Datasets(pats, label, hosi, 'test')
    testloader = DataLoader(testdataset)

    ## pre parameter
    RET = np.zeros((K, 2))
    for k in range(K):
        traindataset = Datasets(pats, label, hosi, 'train', K, k)
        trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True)
        valdataset = Datasets(pats, label, hosi, 'val', K, k)
        valloader = DataLoader(valdataset)
        loaders = {'train': trainloader, 'val': valloader}

        net = Net([l], p).cuda()
        loss = SurvLoss()
        # opt = torch.optim.Adamax(net.parameters(), lr = lr)
        opt = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9, nesterov = True, weight_decay = weight_decay)

        ## iteration
        for i in range(1, epochs + 1):
            if i % 250 == 0:
                lr /= 10
                opt = torch.optim.SGD(net.parameters(), lr = lr, momentum = 0.9, nesterov = True, weight_decay = weight_decay)
            costs = 0
            for ps, y in trainloader:
                x = torch.zeros((len(ps), 512)).cuda()
                y = y.cuda()

                # select instance
                net.eval()
                for j, pat in enumerate(ps):
                    pdata = data[int(pat)]
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
            call_back(i, step, net, data, loaders)

        ret = call_back(0, step, net, data, loaders)
        RET[k, :] = ret
    RET = np.mean(RET, axis = 0)
    print_to_out("K-fold mean C_index: train: %.3f, val: %.3f" % (RET[0], RET[1]))
    ## save
    # torch.save(net, modelpath)
    return ret
    

if __name__ == "__main__":
    global modelpath; global plotpath; global matpath; global outfile
    modelpath, plotpath, matpath, outfile = sys.argv[1:5]
    for rep in range(300):
        lr = 10 ** (np.random.random() * 3 - 5)
        p = np.random.random() * 0.8
        l = np.random.randint(100, 300)
        weight_decay = 10 ** (np.random.random() * 3 - 7)
        params = {
            "h5path": "/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/computed_data.h5",
            "csvpath": "/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/merged.csv",
            "lr": lr,
            "epochs": 700,
            "batch_size": 64,
            "step": 10,
            "p": p,
            "l": l,
            "weight_decay": weight_decay,
            "K": 3
        }
        main(**params)
        print_to_out("")