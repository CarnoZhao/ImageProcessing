# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:38:58 2018

@author: Hayreen
"""

import pandas as pd

aa = pd.read_csv('D:\\Learning\\111data\\concatClinic\\Features_noNA_noRe_withTime.csv')
aa = aa.set_index('Unnamed: 0')

bb = pd.read_csv('D:\\Learning\\111data\\concatClinic\\111ClinicFeatures_noNA.csv')
bb = bb.set_index('Unnamed: 0')


cc = pd.concat([aa,bb],axis = 1,join='inner')

cc.to_csv('Features_multimode_noNA_noRe_withClinicTime.csv')