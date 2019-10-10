#清理工作空间
rm(list=ls())

#读取特征数据
#na.fail(x)  #x中至少存在一个NA时，返回错误

##################### 数据预处理 #######################
workbook <- "D:/Learning/111data/Features_multimode_noNA_noRe_withClinicTime_noNAPatient.csv"

alldata <- read.table(workbook, header=TRUE,sep=",")
#print(alldata[,c(3:1211)])
alldata$label<-factor(alldata$label,order=F)

f <- length(alldata) - 20
alldata[,c(2:f)] <- scale(alldata[,c(2:f)])#标准化,消除量纲对数据结构的影响
#上一行出现错误：“Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric”，以下为找到data中具体哪一(几)列报错
# cnames <- colnames(alldata)[2:f]
# for (i in cnames){
#   alldata[,i] <- scale(alldata[,i])
#   print(i)
# }

 

#下面三行在最终效果不好时，可以尝试删掉--from振东师兄
R <- apply(alldata[,c(2:f)], 2, max) - apply(alldata[,c(2:f)],2,min)   #算出极差，即列上的最大值-最小值
alldata[,c(2:f)]<-sweep(alldata[,c(2:f)], 2, apply(alldata[,c(2:f)], 2, min),'-') #在列的方向上减去最小值，不加‘-’也行
alldata[,c(2:f)]<- sweep(alldata[,c(2:f)], 2, R, "/")


alldata[is.na(alldata)] <- 0
alldata <- na.omit(alldata)
newdata <- na.omit(alldata)#排除未分类的数据
summary(newdata$label)
#names(newdata)
dim(newdata)
newdata <- newdata[order(newdata[,'ctTime'],decreasing=F),]

################# 设置训练/验证集 ##############

#48 
#7991221   24256
#注意：训练集少于50时，（上一步筛选出的）特征数量少于10时，效果不好，数据很容易过拟合，可能是对于50个训练集来说，筛选出的特征数量过多，可能需要进一步补充数据
#set.seed(77)
a <- 70
traindata<-newdata[c(1:a),]
testdata <-newdata[c((a+1):105),]

##################### Boruta筛选特征 ########################
# library(Boruta)
# bfeature <- colnames(traindata)[1:1395]
# traindata_t <- traindata[bfeature]
# label <- traindata$label
# traindata_t <- data.frame(traindata_t,label)   #把符合p值的特征和标签重新组合成dataframe
# write.csv(newdata_t,file = 'newdata_t.csv')
# 
# boruta.train <- Boruta(label~., data = traindata_t,maxRuns=500,doTrace = 1)
# print(boruta.train)
# # plot(boruta.train, xlab = "", xaxt = "n")
# # lz<-lapply(1:ncol(boruta.train$ImpHistory),function(i)
# # boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])
# # names(lz) <- colnames(boruta.train$ImpHistory)
# # axis(side = 1,las=2,labels = names(Labels),
# #      at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)
# final.boruta <- TentativeRoughFix(boruta.train)
# bornames <- getSelectedAttributes(final.boruta, withTentative = F)
# boruta.df <- attStats(final.boruta)

###########################  单因素分析 p-value  ###############################

pfeature <- c()
cnames <- colnames(traindata)[2:f+18]
traindata_t <- traindata[cnames]
label <- traindata$label
traindata_t <- data.frame(traindata_t,label)
#write.csv(newdata_t,file = 'newdata_t.csv')
#y1=as.numeric(unlist(newdata_t[c("label_m3")]))
#选出在低和中高两组的显著性检验中符合p值的特征，写入pfeature中
pvalue <- data.frame()
for (i in cnames){
  x1=as.numeric(unlist(traindata_t[i]))
  #chisqtest <- chisq.test(x1,y1,simulate.p.value = TRUE)
  #t<-t.test(x1,y1)
  u<-wilcox.test(x1~label)
  #print(t)
  #print(u)
  #print(chisqtest)
  if(u$p.value <= 0.05){
    #print(u)
    p_value <- c(as.character(i),u$p.value)
    pvalue <- rbind(pvalue,t(as.matrix(p_value)))
    pfeature <- c(pfeature,i)
    
  }
}
pfeature

######################## mRMR #########################

library(survcomp)
#mRMR
mrmr.seq <- mrmr.cindex(x=traindata_t[,c(pfeature)],cl=as.numeric(traindata_t$label),method="norther")
mrmr.seq <- as.matrix(mrmr.seq)
mrmr_sort <- mrmr.seq[order(mrmr.seq,decreasing=TRUE),] #MRMR越大越好，是相对的排序
#decreasing=TRUE是按照由大到小排列，一般默认，有的电脑默认由小到大排列！
#所以最好加上decreasing=TRUE
#mRMR筛选的特征数量
#print(mrmr_sort)
mfeature <- names(mrmr_sort)[1:28]  #对特征进行排序，越靠前的信息量越大



#训练集
#traindata<-newdata[newdata$flag==0,]
#测试集
#testdata<-newdata[newdata$flag==1,]

# ############################### Lasso降维 #####################################
# 
# #LASSO
# library(Matrix)
# library(foreach)
# library(glmnet)
# #set.seed(675)
# 
# xd <- as.matrix(traindata[mfeature])#自变量
# yd <- as.matrix(traindata[, "label"])#响应变量
# set.seed(0)
# g <- cv.glmnet(xd, yd,family="binomial",type.measure="auc", nfolds = 5,alpha=1)
# plot(g)#绘制cv变化图
# #g$lambda.1se #最佳lambda值
# g.best <- g$glmnet.fit #对应的最佳模型
# g.coef <- coef(g$glmnet.fit, s = g$lambda.1se) #系数
# g.coef[which(g.coef != 0)]
# #选择的变量
# g.coef <- coef(g$glmnet.fit, s = g$lambda.1se) #系数
# name1=row.names(g.coef)
# a1 <- name1[which(g.coef != 0)] #选择nonzero的变量
# LAnames=a1[2:length(a1)]
# plot(g.best)
# 
# tempdata <- traindata[,c(LAnames,'label')]
########################### RFE特征选择 ############################
# sbfControls_rf <- sbfControl(functions = rfSBF,method = 'cv')
#
# #使用sbf函数进行特征选择(随机森林)
#
# fs_rf <- sbf(x = traindata[,mfeature],y = traindata[,'label'],sbfControl = sbfControls_rf)
# fs_rf$optVariables

set.seed(0)
# load the library
library(mlbench)
library(caret)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs,method = "cv")
# run the RFE algorithm
# set.seed(0)   #由于packages版本升级，且当时未保存变量空间，该种子点已经不能复现当时的结果。
results <- rfe(traindata[,mfeature],as.numeric(as.character(traindata[,'label'])), sizes=c(1:9), rfeControl=control)
rfefeature <- results$optVariables
results
rfefeature
plot(results, type = c('g','o'),xlim=c(0,9),pch=1,lwd=2)


#################################### 热图 ##########################################
Edfeature <- LAnames
c(Edfeature)
plot(g.best)
library(pheatmap)
phmdata <- alldata[,c(Edfeature)]
pheatmap(phmdata)
############################## Logistic 回归 #######################################

library(glmnet)
#p初筛-mRMR排序的前五个
# fitLg <- glm(label ~ EBV_DNA + N + c_wavelet.HHL_ngtdm_Complexity + c_wavelet.HLL_glrlm_LongRunLowGrayLevelEmphasis
#               + c_wavelet.HHL_gldm_LargeDependenceLowGrayLevelEmphasis,data=traindata,family=binomial(link='logit'))
#p初筛-Boruta500次，取筛选出的74个特征的前50%，将这37个特征mRMR排序，取前五
trainLg <- traindata[,c(rfefeature)]
fitLg <- glm(label ~. ,data=trainLg,family=binomial(link='logit'))
summary(fitLg)
traindata$radiomics_signature <- 3.9692*traindata$wavelet.HLL_firstorder_Mean + 0.5733*traindata$wavelet.HLL_firstorder_10Percentile + 1.4113*traindata$original_shape_Flatness
testdata$radiomics_signature <- 3.9692*testdata$wavelet.HLL_firstorder_Mean + 0.5733*testdata$wavelet.HLL_firstorder_10Percentile + 1.4113*testdata$original_shape_Flatness
traindata$p <- predict(fitLg,newdata=traindata,type="response")
testdata$prob<-predict(fitLg,newdata=testdata,type="response")

################################计算Rad-Score#####################################
# traindata$p<-predict(g,newx = xd)[,1]
# xt<-as.matrix(testdata[stepnames])
# testdata$prob<-predict(g,newx = xt)[,1]
# 这里加上 type="class"后，testdata$prob的值就是0和1
# testdata$prob<-predict(g,newx=xt,type="class")[,1]

################################ plot-AUC #####################################
library(pROC)
trainROC <- roc(label~p,traindata,col="1",print.auc=TRUE,print.thres=TRUE)
testROC <- roc(label~prob,testdata,col="2",print.auc=TRUE,print.thres=TRUE)

dev.new()
plot(trainROC,print.thres = T,print.auc = T)
ci.auc(trainROC)
coords(trainROC,'b', ret=c("threshold", "specificity", "sensitivity","npv", "ppv", "accuracy","tn","tp","fn","fp")) 
bth <- as.numeric(coords(trainROC,'b', ret="threshold"))

dev.new()
plot(testROC,col = '2',print.auc = T)
ci.auc(testROC)
coords(testROC, bth, ret=c("threshold", "specificity", "sensitivity","npv", "ppv", "accuracy","tn","tp","fn","fp")) 

#delong-test
roc.test(trainROC,testROC)

############### 诺模图 ###################
#signature加到nomogram中
trainLg <- traindata[,c('radiomics_signature','RBC')]
ddist <- datadist(trainLg)
options(datadist='ddist')
fitLg <- lrm(label ~. ,data=trainLg)
options(prType='plain')
print(fitLg, coefs=3)
traindata$p <- predict(fitLg,newdata=traindata)
testdata$prob<-predict(fitLg,newdata=testdata)

nom <- nomogram(fitLg,fun=plogis,lp=F, funlabel="Risk")
###naxes控制变量行间距，xfrac控制左右间距；lmgp控制数字和坐标紧密度
plot(nom,xfrac=0.5,lmgp=0.5, naxes=8)
