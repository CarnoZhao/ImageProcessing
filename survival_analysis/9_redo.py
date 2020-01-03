import h5py
import torch

h = h5py.File("/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/compiled.h5", "a")
data = h['data']

net = torch.load("/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/PRENET.model")
net = net.module.cuda()
net.fc = torch.nn.Identity()

post = h['postdata']

net.eval()
for i in range(len(data)):
    x = data[i:i + 1]
    x = torch.FloatTensor(x).cuda()
    yhat = net(x)[0].cpu().data.numpy()
    post[i] = yhat

h.close()