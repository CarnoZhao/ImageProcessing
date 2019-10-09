# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 14:41:09 2019

@author: Zhong.Lianzhen
"""

from __future__ import print_function
import os
import logging
import scipy.io
import SimpleITK as sitk
import pandas as pd
import radiomics
from radiomics import featureextractor

Input_dir = r'I:\research\ICT-NPC\all_postprocessing_data'
Output_dir = r'I:\research\ICT-NPC\Pyradiomics-feature_LL'
if not os.path.exists(Output_dir):
    os.mkdir(Output_dir)
sequence_spacing = {
        'DWI': [1.2,1.2],
        'data1': [0.5,0.5],
        'data1c': [0.5,0.5],
        'data2': [0.5,0.5]}
xlsx_path = os.path.join(Output_dir,'DWI_feature.csv')
xlsx_path1 = os.path.join(Output_dir,'T1_feature.csv')
xlsx_path2 = os.path.join(Output_dir,'T2_feature.csv')
xlsx_path3 = os.path.join(Output_dir,'T1C_feature.csv')

params = r'E:\Spyder_file\experiment_test\radiomics\example_MR_for_wavelet_LL.yaml'

def extract_feature(sub_dir2,sequence,name,extractor):
        data1 = scipy.io.loadmat(sub_dir2)
        data1 = data1[sequence]
        v_o = data1['v_o']
        v_o = v_o[0][0]
#        import pdb; pdb.set_trace()
        v_s = data1['v_s']
        v_s = v_s[0][0][0][0]
        v_o = v_o.transpose((2,1,0))
        v_s = v_s.transpose((2,1,0))
    #        plt.imshow(v_o1[0,:,:], cmap=plt.cm.bone) #plt.cm.bone
    #        plt.show()
#        import pdb;pdb.set_trace()
        z_spacing = float(data1['SliceThickness'][0][0][0])
        spacing = list(sequence_spacing[sequence])
#        import pdb; pdb.set_trace()
        spacing.append(z_spacing)
        spacing = tuple(spacing)
        img = sitk.GetImageFromArray(v_o)
        img_mask = sitk.GetImageFromArray(v_s)
        #set PixelSapcing (1.0,1.0,1.0)
        img.SetSpacing(spacing)
        img_mask.SetSpacing(spacing)
        featureVector = extractor.execute(img, img_mask)
        aFeature = pd.Series(featureVector)
        aFeature = aFeature.to_frame()
        aFeature.columns = [name]
        
        return aFeature
    
def main():
    
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    
    # Get the PyRadiomics logger (default log-level = INFO)
    logger = radiomics.logger
    logger.setLevel(logging.DEBUG)  # set level to DEBUG to include debug log messages in log file
    
    # Set up the handler to write out all log entries to a file
    handler = logging.FileHandler(filename=os.path.join(Output_dir,'testLog.txt'), mode='w')
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    start_row = 1
    go_row = 1
    batch = 100
    featureVector1 = pd.DataFrame()
    featureVector2 = pd.DataFrame()
    featureVector3 = pd.DataFrame()
    featureVector4 = pd.DataFrame()
    sub_dirs = os.listdir(Input_dir)
    num_file = len(sub_dirs)
    for sub_dir in sub_dirs:
        if start_row < go_row:
            start_row += 1
            continue
        name = sub_dir
        print('#####################')
        print('Read %d-th file_ID: %s' % (start_row,name))
        sub_dir1 = os.path.join(Input_dir,sub_dir)
        sub_dirs_2 = os.listdir(sub_dir1)
        for sub_dir2 in sub_dirs_2:
            sequence = sub_dir2[:-4]
            sub_dir2 = os.path.join(sub_dir1,sub_dir2)
            result = extract_feature(sub_dir2,sequence,name,extractor)
            if sequence == 'DWI':
                featureVector1 = featureVector1.append(result.T)
                if ((start_row % batch) == 0) or (start_row == num_file):
                    div = start_row // batch
                    if div == 1:
                        featureVector1.to_csv(path_or_buf = xlsx_path,encoding='utf-8',index = True,header = True)
                        print('Read DWI')
                        featureVector1 = pd.DataFrame()
                    else:
                        featureVector1.to_csv(path_or_buf = xlsx_path,encoding='utf-8',index = True,header = False,mode = 'a')
                        print('Read DWI')
                        featureVector1 = pd.DataFrame()
                    
            if sequence == 'data1':
                featureVector2 = featureVector2.append(result.T)#不能写featureVector2.append(result.T)
                if ((start_row % batch) == 0) or (start_row == num_file):
                    div = start_row // batch
                    if div == 1:
                        featureVector2.to_csv(path_or_buf = xlsx_path1,encoding='utf-8',index = True,header = True)
                        print('Read data1')
                        featureVector2 = pd.DataFrame()
                    else:
                        featureVector2.to_csv(path_or_buf = xlsx_path1,encoding='utf-8',index = True,header = False,mode = 'a')
                        print('Read data1')
                        featureVector2 = pd.DataFrame()
            if sequence == 'data1c':
                featureVector3 = featureVector3.append(result.T)
                if ((start_row % batch) == 0) or (start_row == num_file):
                    div = start_row // batch
                    if div == 1:
                        featureVector3.to_csv(path_or_buf = xlsx_path2,encoding='utf-8',index = True,header = True)
                        print('Read data1c')
                        featureVector3 = pd.DataFrame()
                    else:
                        featureVector3.to_csv(path_or_buf = xlsx_path2,encoding='utf-8',index = True,header = False,mode = 'a')
                        print('Read data1c')
                        featureVector3 = pd.DataFrame()
            if sequence == 'data2':
                featureVector4 = featureVector4.append(result.T)
                if ((start_row % batch) == 0) or (start_row == num_file):
                    div = start_row // batch
                    if div == 1:
                        featureVector4.to_csv(path_or_buf = xlsx_path3,encoding='utf-8',index = True,header = True)
                        print('Read data2')
                        featureVector4 = pd.DataFrame()
                    else:
                        featureVector4.to_csv(path_or_buf = xlsx_path3,encoding='utf-8',index = True,header = False,mode = 'a')
                        print('Read data2')
                        featureVector4 = pd.DataFrame()
        start_row += 1
    
    logger.removeHandler(handler)
    handler.close()

if __name__ == '__main__':
    main()