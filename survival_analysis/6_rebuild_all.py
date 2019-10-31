import os
import h5py
import torch


class Data(object):
    def __init__(self, h5path, infopath, figpath = None):
        self.h5path = h5path
        self.infopath = infopath
        if not os.path.exists(h5path):
            self.__make_h5(figpath)
        pass

    def __make_h5(self, figpath):
        tpdic = {'huaisi': 0, 'jizhi': 1, 'tumor': 2, 'tumorln': 3}
        tiffiles = [f.strip() for f in os.popen("find %s -name \"*.tif\"" % figpath)]
        h5 = h5py.File(self.h5path, 'w')

    def load(self):
        pass