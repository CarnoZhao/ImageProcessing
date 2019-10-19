import torch
import torchvision

class LabelSmoothingFocalLoss(torch.nn.Module):
    def __init__(self, classes, smoothing = 0, dim = -1, weight = None, focal = False, gamma = 2):
        super(LabelSmoothingFocalLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        if weight is None:
            self.weight = (torch.ones(classes) / classes).to('cuda')
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