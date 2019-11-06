ind = c('性别','发病年龄','肉瘤样细胞','瘤内坏死','淋巴细胞浸润总',
    'T读片','N读片','总分期','time','event')
ex.msag$性别 = factor(ifelse(ex.msag$性别 == "男",1,2))
ex.msag$肉瘤样细胞 = factor(ex.msag$肉瘤样细胞)
ex.msag$瘤内坏死 = factor(ex.msag$瘤内坏死)

ex.msag$dividedGroup = 1
msag$dividedGroup = 0
TNM = ifelse(ex.msag$T读片 > (ex.msag$N读片+1),ex.msag$T读片,(ex.msag$N读片+1))
ex.msag$总分期 = TNM
ex.msag$TN2cut=factor(ifelse(ex.msag$总分期<3,0,1))
all.msag = rbind(msag[,ind],ex.msag[,ind])




source('RD_PyradFile.r')
wb = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature/T1_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature-Wave-HH/T1_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature-Wave-LL/T1_feature.xlsx"
py.T1 = RD_PyradFile(wb,wb1,wb2)

wb = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature/T1C_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature-Wave-HH/T1C_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature-Wave-LL/T1C_feature.xlsx"
py.T1C = RD_PyradFile(wb,wb1,wb2)

wb = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature/T2_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature-Wave-HH/T2_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/Pyradiomics-feature-Wave-LL/T2_feature.xlsx"
py.T2 = RD_PyradFile(wb,wb1,wb2)

###重复性实验30例
source('repeat_feature.r')
wb = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature/T1_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature-Wave-HH/T1_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature-Wave-LL/T1_feature.xlsx"
re.py.T1 = RD_PyradFile(wb,wb1,wb2)
coln = names(py.T1)
for(col in coln) {
  re.py.T1[,col] = as.numeric(as.character(re.py.T1[,col]))
  py.T1[,col] = as.numeric(as.character(py.T1[,col]))
}
re.T1.fea = repeat_feature(py.T1,re.py.T1)

wb = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature/T1C_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature-Wave-HH/T1C_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature-Wave-LL/T1C_feature.xlsx"
re.py.T1C = RD_PyradFile(wb,wb1,wb2)
coln = names(py.T1C)
for(col in coln) {
  re.py.T1C[,col] = as.numeric(as.character(re.py.T1C[,col]))
  py.T1C[,col] = as.numeric(as.character(py.T1C[,col]))
}
re.T1C.fea = repeat_feature(py.T1C,re.py.T1C)

wb = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature/T2_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature-Wave-HH/T2_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/Re-Pyradiomics-feature-Wave-LL/T2_feature.xlsx"
re.py.T2 = RD_PyradFile(wb,wb1,wb2)
coln = names(py.T2)
for(col in coln) {
  re.py.T2[,col] = as.numeric(as.character(re.py.T2[,col]))
  py.T2[,col] = as.numeric(as.character(py.T2[,col]))
}
re.T2.fea = repeat_feature(py.T2,re.py.T2)

name1=re.T1.fea$cname[re.T1.fea$icc>0.75]
name1 = as.vector(name1)
py.T1=py.T1[,name1]

name1=re.T1C.fea$cname[re.T1C.fea$icc>0.75]
name1 = as.vector(name1)
py.T1C=py.T1C[,name1]

name1=re.T2.fea$cname[re.T2.fea$icc>0.75]
name1 = as.vector(name1)
py.T2=py.T2[,name1]

rname = row.names(ex.msag)
wb = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T1_feature_test.xlsx"
wb1 = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T1_feature_test.xlsx"
wb2 = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T1_feature_test.xlsx"
ex.py.T1 = RD_PyradFile(wb,wb1,wb2)
ex.py.T1 = ex.py.T1[rname,]
cname = names(py.T1)
ex.py.T1 = ex.py.T1[,cname]
for(col in cname) {
  ex.py.T1[,col] = as.numeric(as.character(ex.py.T1[,col]))
}

wb = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T1C_feature_test.xlsx"
wb1 = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T1C_feature_test.xlsx"
wb2 = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T1C_feature_test.xlsx"
ex.py.T1C = RD_PyradFile(wb,wb1,wb2)
ex.py.T1C = ex.py.T1C[rname,]
cname = names(py.T1C)
ex.py.T1C = ex.py.T1C[,cname]
for(col in cname) {
  ex.py.T1C[,col] = as.numeric(as.character(ex.py.T1C[,col]))
}

wb = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T2_feature_test.xlsx"
wb1 = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T2_feature_test.xlsx"
wb2 = "E:/ZhangFan_BingLi_GuangXi_exteralVal/py-feature/T2_feature_test.xlsx"
ex.py.T2 = RD_PyradFile(wb,wb1,wb2)
ex.py.T2 = ex.py.T2[rname,]
cname = names(py.T2)
ex.py.T2 = ex.py.T2[,cname]
for(col in cname) {
  ex.py.T2[,col] = as.numeric(as.character(ex.py.T2[,col]))
}



set.seed(4321)
rname = row.names(msag)
py.T1 = py.T1[rname,]
py.T1C = py.T1C[rname,]
py.T2 = py.T2[rname,]
ind=sample(1:157,0.6*157,replace = F)
# ind = order(start_time)[1:round(0.7*157)]
ind1=rep(1,157)
ind1[ind]=0
msag$group=ind1
# DWI.dada1=DWI.dada[ind,];DWI.dada2=DWI.dada[-ind,]
py.T1.tra=py.T1[ind,];py.T1.val=py.T1[-ind,]
py.T1C.tra=py.T1C[ind,];py.T1C.val=py.T1C[-ind,]
py.T2.tra=py.T2[ind,];py.T2.val=py.T2[-ind,]
msag1=msag[ind,];msag2=msag[-ind,]

all.py.T1 = rbind(py.T1,ex.py.T1)
all.py.T1C = rbind(py.T1C,ex.py.T1C)
all.py.T2 = rbind(py.T2,ex.py.T2)
rname = row.names(all.msag)
all.py.T1 = all.py.T1[rname,]
all.py.T1C = all.py.T1C[rname,]
all.py.T2 = all.py.T2[rname,]
set.seed(1243)
ind=sample(1:201,0.6*201,replace = F)
# ind = order(start_time)[1:round(0.7*157)]
ind1=rep(1,201)
ind1[ind]=0
all.msag$group=ind1
all.msag1 = all.msag[ind,]
all.msag2 = all.msag[-ind,]
all.py.T1.tra = all.py.T1[ind,];all.py.T1.val = all.py.T1[-ind,]
all.py.T1C.tra = all.py.T1C[ind,];all.py.T1C.val = all.py.T1C[-ind,]
all.py.T2.tra = all.py.T2[ind,];all.py.T2.val = all.py.T2[-ind,]


h1=hazard.ratio(x= all.msag1$性别, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value
h1=hazard.ratio(x= all.msag1$肉瘤样细胞, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value
h1=hazard.ratio(x= all.msag1$瘤内坏死, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value
h1=hazard.ratio(x= all.msag1$淋巴细胞浸润总, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value

h1=hazard.ratio(x= all.msag1$T2cut, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value
h1=hazard.ratio(x= all.msag1$N1cut, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value
h1=hazard.ratio(x= all.msag1$TN2cut, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value
h1=hazard.ratio(x= all.msag1$agecut, surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$p.value

cox1=coxph(Surv(time,event) ~N1cut+淋巴细胞浸润总+瘤内坏死,data = all.msag1)
summary(cox1)


prod.tra2<-predict(cox1, newdata =all.msag1,type="lp")
h1 = hazard.ratio(x= (prod.tra2), surv.time=all.msag1$time, surv.event=all.msag1$event)
h1$hazard.ratio
h1$lower
h1$p.value
cc <- concordance.index(x=(prod.tra2),surv.time=all.msag1$time, surv.event=all.msag1$event,method="noether")
cc$c.index

prod.val2<-predict(cox1, newdata =all.msag2,type="lp")
h1 = hazard.ratio(x= (prod.val2), surv.time=all.msag2$time, surv.event=all.msag2$event)
h1$hazard.ratio
h1$lower
h1$p.value
cc <- concordance.index(x=(prod.val2),surv.time=all.msag2$time, surv.event=all.msag2$event,method="noether")
cc$c.index


#特征选择
new.cox1<-coxph(Surv(s.time.new,s.event.new)~X0_fos_skewness.x + X2_GLCM_variance.y + X2_GLRLM_LRHGLE.y, data = data)
summary(new.cox1)
#利用AIC优化模型
new.cox2<-step(new.cox1,direction = c("both"))#with a default of "both"
summary(new.cox2)

Features1$Fmrmr<-mrmr.cindex(x=data,surv.time=data$s.time.new,surv.event=data$s.event.new,method="norther")
mrmr_sort1<-Features1[rev(order(Features1$Fmrmr)),]#rev provides a reversed version of its argument.order返回的是索引，sort返回的是排序好的值

x <- as.matrix(tra.data)

Time = as.double(msag1$time)
Status = as.double(msag1$event)
surv = Surv(Time,Status) 
cv.fit<-cv.glmnet(x,surv,nfolds=7, family="cox",nlambda = 200, alpha=1) 

#randomForestSRC