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

def generate_rotate(img, roi, num_rotate = 12):
    base = 360 // num_rotate
    rotates = np.array([
        [simg.rotate(img, angle * base, reshape = False),
        simg.rotate(roi, angle * base, reshape = False)]
        for angle in range(num_rotate)])
    fimg = np.fliplr(img)
    froi = np.fliplr(roi)
    flip_rotate = np.array([
        [simg.rotate(fimg, angle * base, reshape = False),
        simg.rotate(froi, angle * base, reshape = False)]
        for angle in range(num_rotate)])
    return np.concatenate((rotates, flip_rotate), axis = 0)

def noise_bright_dark(rotates, num_noise = 5, sigma = 5 ** 0.5, darkness = 5):
    noises = []
    for i in range(num_noise):
        noise_add = np.random.randn(*rotates.shape) * sigma
        noise_add[:, :, 1, :] = 0
        noises.append(rotates + noise_add)
    bright = rotates[:, :, :, :]
    bright[:, :, 0, :] += darkness
    dark = rotates[:, :, :, :]
    dark[:, :, 0, :] -= darkness
    return np.concatenate((rotates, *noises, bright, dark), axis = 0)

def position_trans(noises, xmin, xmax, ymin, ymax, move = 5):
    ce = noises[:, :, xmin:xmax, ymin:ymax]
    le = noises[:, :, xmin - move:xmax - move, ymin:ymax]
    ri = noises[:, :, xmin + move:xmax + move, ymin:ymax]
    up = noises[:, :, xmin:xmax, ymin - move:ymax - move]
    do = noises[:, :, xmin:xmax, ymin + move:ymax + move]
    return np.concatenate((ce, le, ri, up, do), axis = 0)

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
        all_flip_rotates = generate_rotate(img, roi)
        noises = noise_bright_dark(all_flip_rotates)
        positions = position_trans(noises, xmin, xmax, ymin, ymax)
        sio.savemat(savepath + matfile, {'data': positions})

main(224)