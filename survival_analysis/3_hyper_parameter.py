import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt

def f():
    names = ['lr', 'dropout_p', 'mid_layer', 'weight_decay', 'train', 'val']
    d = {name: [] for name in names}
    with open("/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_outs/Oct.29_23:35.out", 'r') as f:
        for line in f:
            for name in names[:4]:
                if line.startswith(name):
                    d[name].append(eval(line.split(':')[1].strip()))
                    break
            if line.startswith('K-fold'):
                for sep in line.split(','):
                    name, idx = ('train', 2) if 'train' in sep else ('val', 1)
                    d[name].append(eval(sep.split(':')[idx].strip()))
    minlen = min(len(slot) for slot in d.values())
    for name in names:
        d[name] = d[name][:minlen]
    d = pd.DataFrame(d)
    d.to_csv('/home/tongxueqing/zhao/ImageProcessing/survival_analysis/_data/hyperparameter.csv')

def f2(mode, A = [1, 1, 1]):
    if isinstance(A, list):
        F = lambda x: np.dot(x, A) / sum(A)
    else:
        F = A
    import os
    from collections import defaultdict
    import numpy as np
    times = ["20:40", "20:29", "20:28", "20:32", "22:17", "15:35", "15:36", "15:37", "15:38"]
    files = ["/wangshuo/zhaox/ImageProcessing/survival_analysis/_outs/" + f for f in os.listdir("/wangshuo/zhaox/ImageProcessing/survival_analysis/_outs") if any([t in f for t in times])]
    d = defaultdict(list)
    for fname in files:
        with open(fname) as f:
            ln = 0
            for l in f:
                if not l.startswith('lr'):
                    continue
                fs = l.split(' | ')
                d['f'].append(os.path.basename(fname).split('.')[2])
                d['line'].append(ln)
                for k, v in [field.split(':')[:2] for field in fs]:
                    k = k.strip()
                    v = v.strip()
                    try:
                        v = eval(v)
                    except:
                        pass
                    d[k].append(v)
                ln += 1
    m = -2 ** 31
    n = -1
    for idx, p in enumerate(zip(d['citr'], d['civl'], d['cits'])):
        if mode == 'min':
            if all([i > m for i in p]):
                m = min(p)
                n = idx
        elif mode == 'mean':
            if np.mean(p) > m:
                m = np.mean(p)
                n = idx
        elif mode == "custom":
            if F(p) > m:
                m = F(p)
                n = idx
    print(round(m, ndigits = 3))
    print(' | '.join([k + ": " + str(v[n]) for k, v in d.items()]))

f2('custom', lambda x: x[0] if x[1] > x[0] and x[2] > x[0] else -2 ** 32)
