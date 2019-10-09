# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 08:45:41 2019

@author: Zhong.Lianzhen
"""
import tensorflow as tf
import numpy as np
import scipy.io
import cv2
from scipy import signal
import os

img_width = 128
img_heigh = 128

def shuffe_center(x_center,y_center,order = 0):
    if order == 0:
        return x_center,y_center
        # print('No shuffe_center')
    if order == 1:
        x_center -= 10
        return x_center,y_center
    if order == 2:
        x_center += 10
        return x_center,y_center
    if order == 3:
        y_center -= 10
        return x_center,y_center
    if order == 4:
        y_center += 10
    
    return x_center,y_center

def shuffe_grayValue(v0):
    alpha = np.random.randint(3) - 1
    # if alpha:
        # print('jitter grayvalue')
    v0 = v0 - alpha*10
    return v0

def rotate_center(slice0, slice1, angle = 36):
    # 这里的第一个参数为旋转中心，第二个为旋转角度（逆时针），第三个为旋转后的缩放因子
    # 可以通过设置旋转中心，缩放因子，以及窗口大小来防止旋转后超出边界的问题
    nn = int(360/angle)
    rows,cols = slice0.shape
    rotate = np.random.randint(nn)
    # if rotate:
        # print('rotate')
    M = cv2.getRotationMatrix2D((cols/2,rows/2), angle*rotate, 1)
    # 第三个参数是输出图像的尺寸大小

    slice0 = cv2.warpAffine(slice0, M, (rows, cols),cv2.INTER_LINEAR)
    slice1 = cv2.warpAffine(slice1, M, (rows, cols),cv2.INTER_MAX)

    return slice0,slice1
#添加高斯噪声
def gasuss_noise(image, mean=0, var=0.001):
    ''' 
        添加高斯噪声
        mean : 均值 
        var : 方差
    '''

    noise = 0.001*2048*np.random.normal(mean, var ** 0.5, image.shape)

    return noise
#高斯平滑
def gaussBlur(img, sigma, H, W, _boundary='fill', _fillvalue=0):
    # 构建水平方向上的高斯卷积核
    gaussKernel_x = cv2.getGaussianKernel(W, sigma, cv2.CV_32F)
    # 转置
    gaussKernel_x = np.transpose(gaussKernel_x)
    # 图像矩阵与水平高斯核卷积
    gaussBlur_x = signal.convolve2d(img, gaussKernel_x, mode="same",
                                    boundary=_boundary, fillvalue=_fillvalue)
    # 构建垂直方向上的高斯卷积核
    gaussKernel_y = cv2.getGaussianKernel(H, sigma, cv2.CV_32F)
    # 与垂直方向上的高斯卷核
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode="same",
                                     boundary=_boundary, fillvalue=_fillvalue)
    return gaussBlur_xy

def data_aug(v0,v1):
    v0 = v0.astype(np.float32)
#    filp
    if np.random.randint(2) == 1:
        # print('flip')
        flip = np.random.randint(2)
        v0 = cv2.flip(v0, flip)
        v1 = cv2.flip(v1, flip)
#    rotate
    v0,v1 = rotate_center(v0,v1)
#    add noise
    add = np.random.randint(2)
    # if add:
        # print('add noise')
    v0 = v0 + add*gasuss_noise(v0)
    v0 = np.clip(v0, 0.0, 2048)
#    smooth_gaussian
    if np.random.randint(2) == 1:
        # print('smooth_gaussian')
        v0 = gaussBlur(v0,1,7,7)
#    grayvalue jitter
    # v0 = shuffe_grayValue(v0)
#    center jitter
    x_coord,y_coord = np.where(v1 == 1)
    x_min = np.min(x_coord)
    x_max = np.max(x_coord)
    y_min = np.min(y_coord)
    y_max = np.max(y_coord)
    x_center = int((x_min+x_max)/2)
    y_center = int((y_min+y_max)/2)
    x_center,y_center = shuffe_center(x_center, y_center, order = np.random.randint(5))
    x_start = int(x_center-img_width/2)
    x_end = int(x_center+img_width/2)
    y_start = int(y_center-img_heigh/2)
    y_end = int(y_center+img_heigh/2)
    v0 = v0[x_start:x_end,y_start:y_end]
    v1 = v1[x_start:x_end,y_start:y_end]
#    数据标准化
    v0 = np.clip(v0, 0.0, 2048)
    v0 = v0/2048
    
    v1 = v1.astype(np.float32)
    v0 = v0.astype(np.float32)
    img = np.stack([v0,v1],axis = 0)
    img = img.transpose(2,1,0)
    
#    import matplotlib.pyplot as plt
#    plt.figure(1)
#    plt.imshow(v0,'gray')
#    plt.figure(2)
#    plt.imshow(img[0,:,:],'gray')
#    plt.show()
    return img

def preprocess(img_path_list,is_train = True):
    input_img = []
    if is_train:
        for sub_path in img_path_list:
            sub_path = sub_path.rstrip()
            data1 = scipy.io.loadmat(os.path.join(r'/wangshuo/ZhongLianzhen/ICT-NPC/DL_slice/',sub_path))
            v0 = data1['v_o']
            v1 = data1['v_s']
            img = data_aug(v0,v1)
            input_img.append(img)
    else:
        for sub_path in img_path_list:
            sub_path = sub_path.rstrip()
            data1 = scipy.io.loadmat(os.path.join(r'/wangshuo/ZhongLianzhen/ICT-NPC/DL_slice/',sub_path))
            v0 = data1['v_o']
            v1 = data1['v_s']
            v0 = v0.astype(np.float32)
            v1 = v1.astype(np.float32)
            x_coord,y_coord = np.where(v1 == 1)
            x_min = np.min(x_coord)
            x_max = np.max(x_coord)
            y_min = np.min(y_coord)
            y_max = np.max(y_coord)
            x_center = int((x_min+x_max)/2)
            y_center = int((y_min+y_max)/2)
            x_start = int(x_center-img_width/2)
            x_end = int(x_center+img_width/2)
            y_start = int(y_center-img_heigh/2)
            y_end = int(y_center+img_heigh/2)
            v0 = v0[x_start:x_end,y_start:y_end]
            v1 = v1[x_start:x_end,y_start:y_end]
            img = np.stack([v0,v1],axis = 0)
            img = img.transpose(2,1,0)
            input_img.append(img)
    input_img = np.stack(input_img)
    
    return tf.convert_to_tensor(input_img, tf.float32)