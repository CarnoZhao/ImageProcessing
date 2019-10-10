# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:54:14 2018

@author: lhl19
"""

import os
from radiomics import featureextractor
import tqdm
import pandas as pd



'''
Init the feature extractor
'''

#extractor = featureextractor.RadiomicsFeaturesExtractor()
params = os.path.join(os.getcwd(), 'Params.yaml')
extractor = featureextractor.RadiomicsFeaturesExtractor(params)
extractor.enableAllFeatures()
extractor.enableAllImageTypes()


# extractor.settings['enableCExtensions'] = False


'''
Set image/mask path
'''
imagePathRoot = 'F:\\111data\\cov_nrrd_0'
maskPathRoot = 'F:\\111data\\cov_nrrdMask_0'
'''
Read the image/mask file path to a list
'''
imagePathList = [os.path.join(imagePathRoot, f) for f in os.listdir(imagePathRoot) if os.path.isfile(os.path.join(imagePathRoot, f))]
maskPathList = [os.path.join(maskPathRoot, f) for f in os.listdir(maskPathRoot) if os.path.isfile(os.path.join(maskPathRoot, f))]
'''
Put the corresponding image and mask in a dictionary.
'''
IMPathDict = dict(zip(imagePathList,maskPathList))


results = pd.DataFrame()

'''
set a bar
'''
bar = tqdm.tqdm(IMPathDict.items())
bar.set_description('processing：')    


'''
feature extractor
''' 
  
for imagePath , maskPath in bar:
    '''
    First, make sure that the image is from the same person as the mask.
    in this case:
        os.path.basename(imagePath)[:-11] :personName from image
        os.path.basename(maskPath)[:-10]  :personName from mask
    '''
# =============================================================================
#         print(os.path.basename(imagePath)[:-11])
#         print(os.path.basename(maskPath)[:-10])
# =============================================================================
    if (os.path.basename(imagePath)[:-11] == os.path.basename(maskPath)[:-10]):
        featureVector = extractor.execute(imagePath, maskPath)
        aFeature = pd.Series(featureVector)
        aFeature = aFeature.to_frame()
        name = os.path.basename(imagePath)[:-11]
        aFeature.columns = [name]
        results = results.append(aFeature.T)
    else:
        raise IOError('The image does not match the mask.')
'''
save
'''
results.to_csv('radiomicsFeatures_Pa_0.csv')


