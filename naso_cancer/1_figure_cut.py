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

def generate_rotate(cat, indices, num_rotate = 12):
    xmin, xmax, ymin, ymax = indices
    base = 360 // num_rotate
    rotates = np.array([simg.rotate(cat, angle * base, reshape = False) for angle in range(num_rotate)])
    fliped = np.flip(cat, axis = 1)
    flip_rotate = np.array([simg.rotate(fliped, angle * base, reshape = False) for angle in range(num_rotate)])
    catted = np.concatenate((rotates, flip_rotate), axis = 0)
    return catted[:, :, xmin:xmax, ymin:ymax]

def noise_bright_dark(cat, indices, num_noise = 5, sigma = 5 ** 0.5, darkness = 5):
    xmin, xmax, ymin, ymax = indices
    noises = []
    for i in range(num_noise):
        noise_add = np.random.randn(*cat.shape) * sigma
        noise_add[:, 1, :] = 0
        noises.append(cat + noise_add)
    bright = cat[:, :, :]
    bright[:, 0, :] += darkness
    dark = cat[:, :, :]
    dark[:, 0, :] -= darkness
    catted = np.concatenate((*noises, bright, dark), axis = 0)
    return catted[:, :, xmin:xmax, ymin:ymax]

def position_trans(cat, indices, move = 5):
    xmin, xmax, ymin, ymax = indices
    le = cat[:, xmin - move:xmax - move, ymin:ymax]
    ri = cat[:, xmin + move:xmax + move, ymin:ymax]
    up = cat[:, xmin:xmax, ymin - move:ymax - move]
    do = cat[:, xmin:xmax, ymin + move:ymax + move]
    return np.concatenate((le, ri, up, do), axis = 0)

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
        indices = (xmin, xmax, ymin, ymax)
        cat = np.stack(img, roi, axis = 0)
        rotated = generate_rotate(cat, indices)
        noised = noise_bright_dark(cat, indices)
        moved = position_trans(cat, indices)
        catted = np.concatenate((rotated, noised, moved), axis = 0)
        sio.savemat(savepath + matfile, {'data': catted})

main(224)