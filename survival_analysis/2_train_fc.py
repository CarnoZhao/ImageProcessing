import torch
import h5py
import sys
import numpy as np
import pandas as pd
sys.path.insert(1, "/wangshuo/zhaox/ImageProcessing/survival_analysis/_ref")
from DeepsurvLoss import SurvLoss
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index

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
        self.dr1 = torch.nn.Dropout(p)
        self.fc1 = torch.nn.Linear(in_features = 512, out_features = layers[0], bias = True)
        self.dr2 = torch.nn.Dropout(p)
        self.fc2 = torch.nn.Linear(in_features = layers[0], out_features = 1, bias = True)

    def forward(self, x):
        #x = self.dr1(x)
        x = self.fc1(x)
        x = torch.nn.Tanh()(x)
        x = self.dr2(x)
        x = self.fc2(x)
        x = torch.nn.Tanh()(x)
        return x

def print_to_out(*args):
    with open(outfile, 'a') as f:
        f.write(' '.join([str(arg) for arg in args]))
        f.write('\n')

def load_data(h5path, csvpath):
    h5 = h5py.File(h5path)
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
    pats, label, data, hosi = load_data(h5path, csvpath)
    traindataset = Datasets(pats, label, hosi, True)
    trainloader = DataLoader(traindataset, batch_size = batch_size, shuffle = True)
    testdataset = Datasets(pats, label, hosi, False)
    testloader = DataLoader(testdataset)
    net = Net([256], 0.5).cuda()
    loss = SurvLoss()
    opt = torch.optim.Adamax(net.parameters(), lr = lr)
    for i in range(epochs):
        costs = 0
        for pats, y in trainloader:
            order = torch.argsort(-torch.abs(torch.squeeze(y)))
            y = y[order]
            pats = pats[order]
            net.eval()
            x = []
            for p in pats:
                pdata = data[int(p)]
                pyhat = net(torch.FloatTensor(pdata).cuda())
                maxidx = torch.argmax(pyhat)
                x.append(pdata[maxidx])
            x = torch.FloatTensor(np.array(x)).cuda()
            net.train()
            y = y.cuda()
            yhat = net(x)
            opt.zero_grad()
            cost = loss(yhat, y)
            cost.backward()
            costs += float(cost)
            opt.step()
        if i % 10 == 0:
            #print("%d | costs: %.3f" % (i, costs))            
            print_to_out("%d | costs: %.3f" % (i, costs))
    with torch.no_grad():
        for loader in [trainloader, testloader]:
            Y = []
            Yhat = []
            for pats, y in loader:
                Y.extend([yi for yi in y])
                for p in pats:
                    pdata = data[int(p)]
                    pyhat = net(torch.FloatTensor(pdata).cuda())
                    preds = torch.max(pyhat, dim = 0).values
                    Yhat.append(float(preds))
            Y = np.array(Y)
            Yhat = np.array(Yhat)
            ci = concordance_index(np.abs(Y), -1 * np.array(Yhat), np.where(Y > 0, 1, -1))
            print_to_out("ci = %.3f" % ci)
    torch.save(net, modelpath)
    

if __name__ == "__main__":
    global modelpath; global plotpath; global matpath; global outfile
    modelpath, plotpath, matpath, outfile = sys.argv[1:5]

    params = {
        "h5path": "/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/computed_data.h5",
        "csvpath": "/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/merged.csv",
        "lr": 1e-6,
        "epochs": 100,
        "batch_size": 32,
    }
    main(**params)