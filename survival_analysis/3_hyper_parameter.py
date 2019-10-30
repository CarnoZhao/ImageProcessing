import pandas as pd
import matplotlib as mpl
mpl.use('AGG')
import matplotlib.pyplot as plt

def main():
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