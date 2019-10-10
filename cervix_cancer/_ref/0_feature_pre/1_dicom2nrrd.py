# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:22:03 2018

@author: lhl19
"""

import SimpleITK as sitk
import pandas as pd
import os

paths = pd.read_csv('paths_1.csv')
n = 0

'''
开始转换格式
'''

cov_nrrdPath = 'F:\\111data\\cov_nrrd_1'
cov_nrrdMaskPath = 'F:\\111data\\cov_nrrdMask_1'


imageReader = sitk.ImageSeriesReader()



for i in paths.columns:
    if not i == 'Unnamed: 0':
        j = paths.loc[0,i]
        k = paths.loc[1,i]
        dicomNames = imageReader.GetGDCMSeriesFileNames(os.path.dirname(j))
        inputDCM = sitk.ReadImage(dicomNames)
        inputNII = sitk.ReadImage(k)
        #output_ImagePath = {i: os.path.join(cov_nrrdPath, i+'_image.nrrd')}
        output_ImagePath = {i: os.path.join(cov_nrrdPath, i+'_image.nrrd')}
        
        sitk.WriteImage(inputDCM, output_ImagePath[i])
        #output_MaskPath = {i: os.path.abspath(os.path.join(cov_nrrdMaskPath, i+'_mask.nrrd'))  for i in paths.columns}
        output_MaskPath = {i: os.path.abspath(os.path.join(cov_nrrdMaskPath, i+'_mask.nrrd'))}
        #sitk.WriteImage(inputNII, output_MaskPath)
        sitk.WriteImage(inputNII,output_MaskPath[i])



