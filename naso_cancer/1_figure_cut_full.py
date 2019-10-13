'''
@Author: Xun Zhao
@Date: 2019-09-26 14:15:57
@LastEditors: Xun Zhao
@LastEditTime: 2019-09-26 15:27:00
@Description: read from mat file and cut the ROI
'''
import numpy as np
import scipy.ndimage as simg
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

def generate_rotate(cat, indices, savepath, matfile, num_rotate = 5):
    xmin, xmax, ymin, ymax = indices
    base = 360 // num_rotate
    for angle in range(num_rotate):
        sio.savemat(savepath + matfile + ".%s.rotate" % angle, {'data': simg.rotate(cat, angle * base, reshape = False)[:, xmin:xmax:, ymin:ymax]})
    fliped = np.flip(cat, axis = 1)
    for angle in range(num_rotate):
        sio.savemat(savepath + matfile + ".%s.flirot" % angle, {'data': simg.rotate(fliped, angle * base, reshape = False)[:, xmin:xmax:, ymin:ymax]})

def noise_bright_dark(cat, savepath, matfile, num_noise = 3, sigma = 5 ** 0.5, darkness = 5):
    for i in range(num_noise):
        noise = np.random.randn(*cat.shape) * sigma
        noise[1, :, :] = 0
        sio.savemat(savepath + matfile + ".%s.noise" % i, {'data': cat + noise})
    bright = cat[:, :, :]
    bright[0, :, :] += darkness
    sio.savemat(savepath + matfile + ".bright", {'data': bright})
    dark = cat[:, :, :]
    dark[0, :, :] -= darkness
    sio.savemat(savepath + matfile + ".dark", {'data': dark})

def position_trans(cat, indices, savepath, matfile, move = 5):
    xmin, xmax, ymin, ymax = indices
    le = cat[:, xmin - move:xmax - move, ymin:ymax]
    sio.savemat(savepath + matfile + ".left", {'data': le})
    ri = cat[:, xmin + move:xmax + move, ymin:ymax]
    sio.savemat(savepath + matfile + ".right", {'data': ri})
    up = cat[:, xmin:xmax, ymin - move:ymax - move]
    sio.savemat(savepath + matfile + ".up", {'data': up})
    do = cat[:, xmin:xmax, ymin + move:ymax + move]
    sio.savemat(savepath + matfile + ".down", {'data': do})

def main(cut_size, series):
    datapath = '/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/'
    slicespath = datapath + 'DL_slice/'
    savepath = datapath + 'cut_slice/'
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    bar = tqdm.tqdm([f for f in os.listdir(slicespath) if "data%s.mat" % series in f])
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
        indices = (xmin, xmax, ymin, ymax)
        cat = np.stack((img, roi), axis = 0)
        generate_rotate(cat, indices, savepath, matfile)
        noise_bright_dark(cat[:, xmin:xmax, ymin:ymax], savepath, matfile)
        position_trans(cat, indices, savepath, matfile)
        sio.savemat(savepath + matfile + '.raw', {'data': cat[:, xmin:xmax, ymin:ymax]})

main(128, '1')
