# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:38:58 2018

@author: Hayreen
"""

import pandas as pd

aa = pd.read_csv('D:\\Learning\\111data\\concatClinic\\north0524.csv')
aa = aa.set_index('Unnamed: 0')

bb = pd.read_csv('D:\\Learning\\111data\\concatClinic\\south0605.csv')
bb = bb.set_index('Unnamed: 0')


cc = pd.concat([aa,bb],axis = 0)

cc.to_csv('111dataClinic.csv')