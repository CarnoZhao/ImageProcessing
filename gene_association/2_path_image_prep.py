# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:11:52 2019

@author: Zhong.Lianzhen
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mp
from scipy import misc
import os

def Img_Outline(original_img,pt = 65):
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)                     # 高斯模糊去噪（设定卷积核大小影响效果）
    ind=(blurred != 255)
    mx=np.percentile(blurred[ind],pt)
    _, RedThresh = cv2.threshold(blurred, mx, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))          # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)       # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)           # 开运算（去噪点）
    return opened


def findContours_img(original_img,opened):
    contours, hierarchy = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]          # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    # print("angle",angle)
#    box = np.int0(cv2.boxPoints(rect))
#    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    rows, cols = original_img.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (cols, rows),borderValue=(255,255,255))
    return result_img

def off_bg(original_img, opened1):
    contours, _ = cv2.findContours(opened1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[1]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    x=box[:,1];y=box[:,0]
    rows, cols = original_img.shape[:2]
    bx1=max(0,x.min()-10);bx2=min(rows,x.max()+10)
    by1=max(0,y.min()-10);by2=min(cols,y.max()+10)
    draw_img = original_img[bx1:bx2,by1:by2,:]
    # temp=(opened1[bx1,by1] == 255) and (opened1[bx1,by2] == 255) and (opened1[bx2,by1] == 255) and (opened1[bx2,by2] == 255)
    temp = None
    return draw_img,temp

def adj_img(input_dir):
    original_img = cv2.imread(input_dir)
    opened = Img_Outline(original_img)
    rotate_img = findContours_img(original_img,opened)
    #    k=72.994/90
    #    rotate_img1=np.rot90(rotate_img,k)
    opened1 = Img_Outline(rotate_img)
    result_img,_=off_bg(rotate_img, opened1)
    #    plt.imshow(result_img)
    return result_img, original_img

def manual_cut(filename):
    original_img = cv2.imread(filename)
    r, c = original_img.shape[:2]
    for i in range(r):
        if np.sum(original_img[i, :] - 255) != 0:
            xmin = i
            break
    for i in range(r - 1, -1, -1):
        if np.sum(original_img[i, :] - 255) != 0:
            xmax = i
            break
    for i in range(c):
        if np.sum(original_img[:, i] - 255) != 0:
            ymin = i
            break
    for i in range(c - 1, -1, -1):
        if np.sum(original_img[:, i] - 255) != 0:
            ymax = i
            break
    print(xmax - xmin, ymax - ymin)
    return original_img[xmin + 10:xmax - 10, ymin + 10:ymax - 10]

if __name__ == "__main__":
    input_dir = "/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/DP/"
    l = [f for f in os.listdir(input_dir) if "tif" in f]
    for filename in l:
        result_img, original_img =adj_img(input_dir + filename)
        if result_img.shape[0] < 1000 or result_img.shape[1] < 1000 or filename == "1508175_tumor_ln_GX.tif":
            result_img = manual_cut(input_dir + filename)
            cv2.imwrite(input_dir + "cutted." + filename, result_img)
        else:
            cv2.imwrite(input_dir + "cutted." + filename, result_img)
        print(filename, result_img.shape)
    # fig, axs = plt.subplots(2)
    # axs[0].imshow(result_img)
    # axs[1].imshow(original_img)
    # plt.savefig("/home/tongxueqing/zhao/ImageProcessing/stain_classification/output.png")