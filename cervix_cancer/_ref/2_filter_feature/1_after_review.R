#���������ռ�
rm(list=ls())

#��ȡ��������
#na.fail(x)  #x�����ٴ���һ��NAʱ�����ش���

##################### ����Ԥ���� #######################
workbook <- "D:/Learning/111data/Features_multimode_noNA_noRe_withClinicTime_NEW.csv"

alldata <- read.table(workbook, header=TRUE,sep=",")
#print(alldata[,c(3:1211)])
alldata$label<-factor(alldata$label,order=F)

f <- length(alldata) - 20
alldata[,c(2:f)] <- scale(alldata[,c(2:f)])#��׼��,�������ٶ����ݽṹ��Ӱ��
#��һ�г��ִ��󣺡�Error in colMeans(x, na.rm = TRUE) : 'x' must be numeric��������Ϊ�ҵ�data�о�����һ(��)�б���
# cnames <- colnames(alldata)[2:f]
# for (i in cnames){
#   alldata[,i] <- scale(alldata[,i])
#   print(i)



#��������������Ч������ʱ�����Գ���ɾ��--from��ʦ��
R <- apply(alldata[,c(2:f)], 2, max) - apply(alldata[,c(2:f)],2,min)   #�����������ϵ����ֵ-��Сֵ
alldata[,c(2:f)]<-sweep(alldata[,c(2:f)], 2, apply(alldata[,c(2:f)], 2, min),'-') #���еķ����ϼ�ȥ��Сֵ�����ӡ�-��Ҳ��
alldata[,c(2:f)]<- sweep(alldata[,c(2:f)], 2, R, "/")


alldata[is.na(alldata)] <- 0
alldata <- na.omit(alldata)
newdata <- na.omit(alldata)#�ų�δ���������
summary(newdata$label)
#names(newdata)
dim(newdata)
newdata <- newdata[order(newdata[,'ctTime'],decreasing=F),]

################# ����ѵ��/��֤�� ##############

#48 
#7991221   24256
#ע�⣺ѵ��������50ʱ������һ��ɸѡ���ģ�������������10ʱ��Ч�����ã����ݺ����׹���ϣ������Ƕ���50��ѵ������˵��ɸѡ���������������࣬������Ҫ��һ����������
#set.seed(77)
a <- 70
traindata<-newdata[c(1:a),]
testdata <-newdata[c((a+1):105),]

###########################  �����ط��� p-value  ###############################

pfeature <- c()
cnames <- colnames(traindata)[2:f+17]
traindata_t <- traindata[cnames]
label <- traindata$label
traindata_t <- data.frame(traindata_t,label)
#write.csv(newdata_t,file = 'newdata_t.csv')
#y1=as.numeric(unlist(newdata_t[c("label_m3")]))
#ѡ���ڵͺ��и�����������Լ����з���pֵ��������д��pfeature��

######################## mRMR #########################

library(survcomp)
#mRMR
cnames <- cnames[-which(cnames=='cy211')]··
mrmr.seq <- mrmr.cindex(x=traindata_t[,c(cnames)],cl=as.numeric(traindata_t$label),method="norther")
mrmr.seq <- as.matrix(mrmr.seq)
mrmr_sort <- mrmr.seq[order(mrmr.seq,decreasing=TRUE),] #MRMRԽ��Խ�ã�����Ե�����
#decreasing=TRUE�ǰ����ɴ�С���У�һ��Ĭ�ϣ��еĵ���Ĭ����С�������У�
#������ü���decreasing=TRUE
#mRMRɸѡ����������
#print(mrmr_sort)
mfeature <- names(mrmr_sort)[1:13]  #��������������Խ��ǰ����Ϣ��Խ��

###########������pֵ����##############
###########################  �����ط��� p-value  ###############################

pfeature <- c()

traindata_t <- traindata[mfeature]
label <- traindata$label
traindata_t <- data.frame(traindata_t,label)
#write.csv(newdata_t,file = 'newdata_t.csv')
#y1=as.numeric(unlist(newdata_t[c("label_m3")]))
#ѡ���ڵͺ��и�����������Լ����з���pֵ��������д��pfeature��
pvalue <- data.frame()
for (i in mfeature){
  x1=as.numeric(unlist(traindata_t[i]))
  #chisqtest <- chisq.test(x1,y1,simulate.p.value = TRUE)
  #t<-t.test(x1,y1)
  u<-wilcox.test(x1~label)
  #print(t)
  #print(u)
  #print(chisqtest)
  if(u$p.value <= 1){
    #print(u)
    p_value <- c(as.character(i),u$p.value)
    pvalue <- rbind(pvalue,t(as.matrix(p_value)))
    pfeature <- c(pfeature,i)
    
  }
}
pfeature
#ѵ����
#traindata<-newdata[newdata$flag==0,]
#���Լ�
#testdata<-newdata[newdata$flag==1,]

########################### RFE����ѡ�� ############################
# sbfControls_rf <- sbfControl(functions = rfSBF,method = 'cv')
#
# #ʹ��sbf������������ѡ��(���ɭ��)
#
# fs_rf <- sbf(x = traindata[,mfeature],y = traindata[,'label'],sbfControl = sbfControls_rf)
# fs_rf$optVariables

set.seed(0)
# load the library
library(mlbench)
library(caret)

# define the control using a random forest selection function
control <- rfeControl(functions=rfFuncs,method = "cv")
# run the RFE algorithm #29
set.seed(28)
results <- rfe(traindata[,pfeature],as.numeric(as.character(traindata[,'label'])), sizes=c(1:9), rfeControl=control)
rfefeature <- results$optVariables
results
rfefeature

plot(results, type = c('g','o'),xlim=c(0,9),pch=1,lwd=2)


#################################### ��ͼ ##########################################
Edfeature <- LAnames
c(Edfeature)
plot(g.best)
library(pheatmap)
phmdata <- alldata[,c(Edfeature)]
pheatmap(phmdata)
############################## Logistic �ع� #######################################

library(glmnet)
#p��ɸ-mRMR�����ǰ���
# fitLg <- glm(label ~ EBV_DNA + N + c_wavelet.HHL_ngtdm_Complexity + c_wavelet.HLL_glrlm_LongRunLowGrayLevelEmphasis
#               + c_wavelet.HHL_gldm_LargeDependenceLowGrayLevelEmphasis,data=traindata,family=binomial(link='logit'))
#p��ɸ-Boruta500�Σ�ȡɸѡ����74��������ǰ50%������37������mRMR����ȡǰ��
trainLg <- traindata[,c(rfefeature)]
fitLg <- glm(label ~. ,data=trainLg,family=binomial(link='logit'))
newdata$p <- predict(fitLg,newdata=newdata) #DC���߻��õ�
summary(fitLg)
traindata$radiomics_signature <- 3.8684*traindata$wavelet.HLL_firstorder_Mean + 0.5897*traindata$wavelet.HLL_firstorder_10Percentile + 1.0982*traindata$original_shape_Flatness
testdata$radiomics_signature <- 3.8684*testdata$wavelet.HLL_firstorder_Mean + 0.5897*testdata$wavelet.HLL_firstorder_10Percentile + 1.0982*testdata$original_shape_Flatness
newdata$radiomics_signature <- 3.8684*newdata$wavelet.HLL_firstorder_Mean + 0.5897*newdata$wavelet.HLL_firstorder_10Percentile + 1.0982*newdata$original_shape_Flatness
traindata$p <- predict(fitLg,newdata=traindata,type="response")
testdata$prob<-predict(fitLg,newdata=testdata,type="response")


################################����Rad-Score#####################################
# traindata$p<-predict(g,newx = xd)[,1]
# xt<-as.matrix(testdata[stepnames])
# testdata$prob<-predict(g,newx = xt)[,1]
# ������� type="class"��testdata$prob��ֵ����0��1
# testdata$prob<-predict(g,newx=xt,type="class")[,1]

################################ plot-AUC #####################################
library(pROC)
trainROC <- roc(label~p,traindata,col="1",print.auc=TRUE,print.thres=TRUE)
testROC <- roc(label~prob,testdata,col="2",print.auc=TRUE,print.thres=TRUE)
newROC <- roc(label~p,newdata,col="2",print.auc=TRUE,print.thres=TRUE)

dev.new()
plot(trainROC,col = '2',print.thres = F,print.auc = T)
ci.auc(trainROC)
coords(trainROC,'b', ret=c("threshold", "specificity", "sensitivity","npv", "ppv", "accuracy","tn","tp","fn","fp")) 
bth <- as.numeric(coords(trainROC,'b', ret="threshold"))

dev.new()
plot(testROC,col = '2',print.thres = F,print.auc = T)
ci.auc(testROC)
coords(testROC, bth, ret=c("threshold", "specificity", "sensitivity","npv", "ppv", "accuracy","tn","tp","fn","fp")) 

#delong-test
roc.test(trainROC,testROC)

