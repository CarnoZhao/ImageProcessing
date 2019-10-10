# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:58:31 2018

@author: Hayreen
"""

import pandas as pd
import numpy as np

oData = pd.read_csv('D:\\Learning\\111data\\concatClinic\\111dataClinic.csv')
oData.set_index('Unnamed: 0',inplace=True)

for i in oData.columns:
     num = 0
     n = 0
     #算出每一列的均值
     for j in oData.index:
          #j = str(j)
          #print(j)
          #print(type(j))
          if not np.isnan(np.array(oData[i][j])):
               #print(oData[i][j])
               #print(type(oData[i][j]))
               num = num + oData[i][j]
               n = n + 1
     colmean = num/n
     #print(colmean)
     #用均值填充
     m = 0
     for j in oData.index:
          if np.isnan(np.array(oData[i][j])):
               oData[i][j] = colmean
               m = m+1
     print(i + "空值数量有%d/115"%(m))


     
     
# =============================================================================
# oData = oData.fillna(-0.0011)
# =============================================================================

# =============================================================================
# oData = oData.fillna(-0.0011)
# #把空值用-0.0011填补
# =============================================================================
ooData = oData.isnull().any()

#把所有元素值均相同的列去除
for i in oData.columns:
    num = 0
    k = oData[i]['chenhuafang']  #chenhuafang为第一个index
    for j in oData[i]:
        if k == j:
            num += 1
    if num == len(oData[i]):
        oData = oData.drop([i],axis=1)  #axis = 0时为删除行，1时删除列。默认 inplace = False，即不替换原DataFrame，此处，用oData接收一下，也可完成替换，等价写法 ：oData.drop([i],axis=1,inplace=True)

for i,j in enumerate(ooData):
    if j == True :
        print(i)
for i in oData.columns:
    #num = 0
    for j in oData.index:
        if not np.isfinite(np.array(oData[i][j])):
            oData = oData.drop([i],axis=1)
            break
oData.to_csv('FinallFeaturesWithInter_noNA_noRe.csv')
