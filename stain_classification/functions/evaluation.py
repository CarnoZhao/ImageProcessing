import numpy as np
import torch
import scipy.io as sio
from sklearn import metrics
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt

def predict(net, loader, K):
    Y = np.zeros((sum([len(y) for x, y in loader]), K))
    Yhat = np.zeros((sum([len(y) for x, y in loader]), K))
    with torch.no_grad():
        i = 0
        for x, y in loader:
            x = x.to('cuda', dtype = torch.float)
            Y[i:i + len(y), :] = torch.nn.functional.one_hot(y, num_classes = K)
            yhat = torch.softmax(net(x), dim = 1).cpu()
            Yhat[i:i + len(y), :] = yhat
            i += len(y)
    accu = np.mean(np.equal(np.argmax(Y, axis = 1), np.argmax(Yhat, axis = 1)))
    return Y, Yhat, accu
    
def plot_roc_auc(data, plotpath):
    fig, ax = plt.subplots()
    colors = ['r', 'g', 'b']
    for i, name in enumerate(data['names']):
        Y = data[name + 'Y']
        Yhat = data[name + 'Yhat']
        fpr, tpr, _ = metrics.roc_curve(Y.ravel(), Yhat.ravel())
        auc = metrics.auc(fpr, tpr)
        ax.plot(fpr, tpr, c = colors[i], lw = 1, alpha = 0.7, label = u'%sAUC=%.3f' % (name, auc))
    ax.plot((0, 1), (0, 1), c = '#808080', lw = 1, ls = '--', alpha = 0.7)
    ax.set_xlim((-0.01, 1.02))
    ax.set_ylim((-0.01, 1.02))
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(b = True, ls = ':')
    plt.legend()
    plt.savefig(plotpath)

def printplot(net, loaders, K, plotpath, matpath):
    data = {'names': list(loaders.keys())}
    accus = {}
    for key, loader in loaders.items():
        Y, Yhat, accu = predict(net, loader, K)
        data[key + 'Y'] = Y
        data[key + 'Yhat'] = Yhat
        accus[key] = accu
    sio.savemat(matpath, data)
    plot_roc_auc(data, plotpath)
    for name in data['names']:
        print('Accuracy in %s = %.6f' % (name, accus[name]))