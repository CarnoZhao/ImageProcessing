###
 # @Author: Xun Zhao
 # @Date: 2019-09-24 18:44:18
 # @LastEditors: Xun Zhao
 # @LastEditTime: 2019-09-25 09:58:54
 # @Description: combine radiomics features and clinic features
###

radiomics = read.csv('/home/tongxueqing/data/zhaox/data1/featuredata/features.csv', row.names = 1)
clinic = read.csv('/home/tongxueqing/data/zhaox/data1/rawdata/clinic.csv', row.names = 1)
radiomics = t(radiomics)

features = colnames(radiomics)
differ = sapply(features, function(feature){any(radiomics[,feature] != radiomics[1,feature])}, USE.NAMES = F)
differfeatures = features[differ]
radiomics = radiomics[,differfeatures]

combine = merge(radiomics, clinic, by = 0)

noNAcombine = sapply(colnames(combine), function(col) {ifelse(is.na(combine[,col]), mean(combine[,col], na.rm = T), combine[,col])})