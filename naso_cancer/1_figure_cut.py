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
    rotates = [(simg.rotate(img, angle * base), simg.rotate(roi, angle * base)) for angle in range(num_rotate)]
    fimg, froi = np.fliplr(img), np.fliplr(roi)
    flip_rotate = [(simg.rotate(fimg, angle * base), simg.rotate(froi, angle * base)) for angle in range(num_rotate)]
    return rotates + flip_rotate

def noise_bright_dark(rotates, num_noise = 5, sigma = 5 ** 0.5, darkness = 5):
    noises = []
    for img, roi in rotates:
        noises.append((img, roi))
        noises.append((img + darkness, roi))
        noises.append((img - darkness, roi))
        for i in range(num_noise):
            nimg = img + np.random.randn(*img.shape) * sigma
            noises.append((nimg, roi))
    return noises

def position_trans(noises, xmin, xmax, ymin, ymax, move = 5):
    positions = []
    for img, roi in noises:
        oimg = img[xmin:xmax, ymin:ymax]
        limg = img[xmin - 5:xmax - 5, ymin:ymax]
        rimg = img[xmin + 5:xmax + 5, ymin:ymax]
        uimg = img[xmin:xmax, ymin - 5:ymax - 5]
        dimg = img[xmin:xmax, ymin + 5:ymax + 5]
        oroi = roi[xmin:xmax, ymin:ymax]
        lroi = roi[xmin - 5:xmax - 5, ymin:ymax]
        rroi = roi[xmin + 5:xmax + 5, ymin:ymax]
        uroi = roi[xmin:xmax, ymin - 5:ymax - 5]
        droi = roi[xmin:xmax, ymin + 5:ymax + 5]
        positions.extend([(oimg, oroi), (limg, lroi), (rimg, rroi), (uimg, uroi), (dimg, droi)])
    return positions

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
        for i, (aug_img, aug_roi) in positions:
            sio.savemat(savepath + os.path.splitext(matfile)[0] + '_%d.mat' % i, {'img': aug_img, 'roi' : aug_roi})
            if idx % 1000 == 0 and i % 1000 == 0:
                plot_test(aug_img, aug_roi, os.path.splitext(matfile)[0] + '_%d.mat' % i, datapath)

main(224)