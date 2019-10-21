import scipy.io as sio
import numpy as np
from collections import Counter

path = "/wangshuo/zhaox/ImageProcessing/stain_classification/_mat/success.Oct.20_11:44.mat"

data = sio.loadmat(path)
for name in data['names']:
    name = name.strip()
    print("In %s:" % name)
    Y = data[name + "Y"]
    Yhat = data[name + "Yhat"]
    argYhat = np.argmax(Yhat, axis = 1)
    argY = np.argmax(Y, axis = 1)
    cnt = np.zeros((4, 4))
    for yi, yihat in zip(argY, argYhat):
        cnt[yi, yihat] += 1
    print(cnt)