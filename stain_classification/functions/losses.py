import torch
import torchvision

__all__ = ["Loss", "LabelSmoothingFocalLoss"]

def _onehot(Y, K):
    oh = torch.zeros((len(Y), K)).to('cuda')
    oh.scatter_(1, Y.data.unsqueeze(1), 1)
    return oh

def _label_smoothing(Y, K, smoothing):
    ls = torch.zeros((len(Y), K)).to('cuda')
    ls.fill_(smoothing / K)
    ls.scatter_(1, Y.data.unsqueeze(1), 1 - smoothing)
    return ls

class Loss(torch.nn.Module):
    def __init__(self, K, focal, gamma = 2, smoothing = 0.01, weights = None):
        super(Loss, self).__init__()
        self.K = K
        self.focal = focal
        self.gamma = gamma
        self.smoothing = smoothing
        self.weights = weights if weights is not None else 1

    def forward(self, Yhat, Y):
        if self.focal:
            sm = Yhat.softmax(dim = -1)
            logYhat = Yhat.log_softmax(dim = -1) * ((1 - sm) ** self.gamma)
        else:
            logYhat = Yhat.log_softmax(dim = -1)
        if self.smoothing != 0:
            Y = _label_smoothing(Y, self.K, self.smoothing)
        else:
            Y = _onehot(Y, self.K)
        return torch.mean(torch.sum(-Y * logYhat * self.weights, dim = -1))

class LabelSmoothingFocalLoss(torch.nn.Module):
    def __init__(self, classes, smoothing = 0, dim = -1, weight = None, focal = False, gamma = 2):
        super(LabelSmoothingFocalLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if weight is None:
            self.weight = 1
        else:
            self.weight = weight
        self.focal = focal
        self.gamma = gamma

    def forward(self, pred, target):
        sf = pred.softmax(dim = self.dim)
        pred = pred.log_softmax(dim = self.dim)
        if self.focal:
            pred = (1 - sf) ** self.gamma * pred
        with torch.no_grad():
            H = torch.zeros_like(pred)
            H.fill_(self.smoothing / self.cls)
            H.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-H * pred * self.weight, dim = self.dim))

if __name__ == '__main__':
    import numpy as np
    from copy import deepcopy
    l1 = Loss(2, True)
    l2 = LabelSmoothingFocalLoss(2, smoothing = 0.01, focal = True)
    X = np.random.randint(0, 9, (10, 3))
    Y = np.random.randint(0, 2, 10)
    X = torch.FloatTensor(X).to('cuda')
    Y = torch.tensor(Y).to('cuda')
    net = torch.nn.Linear(3, 2, bias = True)
    net2 = deepcopy(net)
    net = net.to('cuda')
    net2 = net2.to('cuda')
    op1 = torch.optim.Adamax(net.parameters(), lr = 0.01)
    op2 = torch.optim.Adamax(net2.parameters(), lr = 0.01)
    for pi in net.parameters():
        print(pi)
    for pi in net2.parameters():
        print(pi)
    cost1 = l1(net(X), Y)
    cost2 = l2(net2(X), Y)
    cost1.backward()
    cost2.backward()
    op1.step()
    op2.step()
