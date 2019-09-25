'''
@Author: Xun Zhao
@Date: 2019-09-23 22:35:31
@LastEditors: Xun Zhao
@LastEditTime: 2019-09-25 10:10:30
@Description: 
'''

import SimpleITK as sitk
from radiomics import featureextractor
import tqdm
import os
import pandas as pd
from itertools import product

reader = sitk.ImageSeriesReader()

params = '/home/tongxueqing/zhaox/ImageProcessing/cervix_cancer/exampleMR_3mm.yaml'
extractor = featureextractor.RadiomicsFeatureExtractor(params)
extractor.enableAllFeatures()
extractor.enableAllImageTypes()

root = '/data/tongxueqing/zhaox/data1/rawdata/'
savepath = '/data/tongxueqing/zhaox/data1/featuredata/'
dcmpath = savepath + 'dcms/'
maskpath = savepath + 'masks/'
hospital = ('SG', 'NH')
leep_none = ('leep', 'none')
label = ('1', '0')
subroots = ['/'.join(sub) + '/' for sub in product(hospital, leep_none, label)]
if False:
    for subroot in subroots:
        path = root + subroot
        if not os.path.exists(path):
            continue
        names = os.listdir(path)
        for name in names:
            namepath = path + name + '/'
            dcm_names = reader.GetGDCMSeriesFileNames(namepath)
            dcm = sitk.ReadImage(dcm_names)
            sitk.WriteImage(dcm, dcmpath + name + '.dcm.nrrd')
            maskfile = namepath + name + '.nii'
            if not os.path.exists(maskfile):
                maskfile += '.gz'
            mask = sitk.ReadImage(maskfile)
            sitk.WriteImage(mask, maskpath + name + '.mask.nrrd')


dcms = sorted([dcmpath + dcm for dcm in os.listdir(dcmpath)])
masks = sorted([maskpath + mask for mask in os.listdir(maskpath)])

paths = dict(zip(dcms, masks))

results = pd.DataFrame()
bar = tqdm.tqdm(paths.items())
bar.set_description('Processing: ')

for dcm, mask in bar:
    dcmname = os.path.basename(dcm)[:-9]
    maskname = os.path.basename(mask)[:-10]
    if dcmname == maskname:
        name = dcmname
    else:
        raise IOError('Names do not fit')
    features = extractor.execute(dcm, mask)
    features = pd.Series(features)
    results[name] = features

results.to_csv(savepath + 'features.csv')