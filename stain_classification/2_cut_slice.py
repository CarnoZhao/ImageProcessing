import cv2
import numpy as np
import os
import scipy.io as sio

def cut(path, filename, savepath):
    img = cv2.imread(path + filename)
    # cv2.imwrite(path + filename, img)
    r, c = img.shape[:2]
    xmin = 10; ymin = 10
    i = 0
    while ymin + 512 < c:
        if xmin + 522 >= r:
            xmin = 10
            ymin += 522
            continue
        xmax = xmin + 512
        ymax = ymin + 512
        cutted = img[xmin:xmax, ymin:ymax, :]
        cv2.imwrite(savepath + os.path.splitext(filename)[0] + "_%d.tif" % i, cutted)
        # sio.savemat(savepath + os.path.splitext(filename)[0] + "_%d.mat" % i, {"data" : cutted})
        i += 1
        xmin = xmax + 10

def main():
    path = "/home/tongxueqing/data/zhaox/stain_classification/normed/"
    savepath = "/home/tongxueqing/data/zhaox/stain_classification/cutted/"
    for filename in os.listdir(path):
        cut(path, filename, savepath)

if __name__ == "__main__":
    main()