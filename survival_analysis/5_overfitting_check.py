from scipy import io
import pandas as pd
import numpy as np
from collections import Counter

def event_count(matpath):
    mat = io.loadmat(matpath)
    csv = pd.read_csv("/wangshuo/zhaox/ImageProcessing/survival_analysis/_data/merged.csv")
    patdic = {csv.number[i]: (csv.event[i], csv.time[i]) for i in range(len(csv))}
    for key in ['train', 'val', 'test']:
        values = mat[key]
        events = [patdic[pat][0] for pat in np.squeeze(values)]
        cnt = Counter(events)
        print(key, cnt, cnt[0] / cnt[1])

if __name__ == "__main__":
    event_count("/wangshuo/zhaox/ImageProcessing/survival_analysis/_mat/Oct.31_12:15.mat")
    