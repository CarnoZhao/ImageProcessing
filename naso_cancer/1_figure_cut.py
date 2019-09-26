'''
@Author: Xun Zhao
@Date: 2019-09-26 14:15:57
@LastEditors: Xun Zhao
@LastEditTime: 2019-09-26 15:27:00
@Description: read from mat file and cut the ROI
'''
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import tqdm

def find_bound(array, axis):
    xsum = np.sum(array, axis = axis)
    xarray = [i for i, value in enumerate(xsum) if value != 0]
    return int(round((np.min(xarray) + np.max(xarray)) / 2))

def plot_test(cut_img, cut_roi, matfile, datapath):
    plotpath = datapath + 'cut_test_plots/'
    if not os.path.exists(plotpath):
        os.mkdir(plotpath)
    plot_img = 127 * (cut_img - np.min(cut_img)) / np.ptp(cut_img) + 127 * cut_roi
    plt.imshow(plot_img, cmap = 'gray')
    plt.savefig(plotpath + os.path.splitext(matfile)[0] + '.png')

def main(cut_size):
    datapath = '/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/'
    slicespath = datapath + 'DL_slice/'
    savepath = datapath + 'cut_slice/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    bar = tqdm.tqdm(os.listdir(slicespath))
    bar.set_description('Processing: ')
    for idx, matfile in enumerate(bar):
        mat = sio.loadmat(slicespath + matfile)
        img = mat['v_o']
        roi = mat['v_s']
        xcenter = find_bound(roi, 1)
        ycenter = find_bound(roi, 0)
        if cut_size % 2 == 1:
            xmin = xcenter - (cut_size - 1) // 2
            xmax = xcenter + (cut_size + 1) // 2
            ymin = ycenter - (cut_size - 1) // 2
            ymax = ycenter + (cut_size + 1) // 2
        else:
            xmin = xcenter - cut_size // 2
            xmax = xcenter + cut_size // 2
            ymin = ycenter - cut_size // 2
            ymax = ycenter + cut_size // 2
        cut_img = img[xmin:xmax, ymin:ymax]
        cut_roi = roi[xmin:xmax, ymin:ymax]
        sio.savemat(savepath + matfile, {'img': cut_img, 'roi' : cut_roi})
        if idx % 100 == 0:
            plot_test(cut_img, cut_roi, matfile, datapath)

main(128)