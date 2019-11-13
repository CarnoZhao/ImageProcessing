# data prepare
{     
     nn = nrow(alldata)
     set.seed(324)
     x1 = sample(1:nn, 0.7*nn, replace = F)
     x2 = sample(x1, 0.2*nn, replace = F)
     ind1 = rep(3,nn)
     ind1[x1] = 1
     ind1[x2] = 2
     alldata$rand_group = ind1
     summary(factor(alldata$rand_group))
     library(prodlim)
     library(survival)
     library(survcomp)

     trainData1 = subset(alldata, rand_group == 1)
     trainData2 = subset(alldata, rand_group == 2)
     trainData = subset(alldata, rand_group != 3)
     valData = subset(alldata, rand_group == 3)
     trainData$LDHcut = ifelse(trainData$LDH>250,1,0)
     valData$LDHcut = ifelse(valData$LDH>250,1,0)
     #age +LDH + CRP  + DL_pred + DL_pred_T2 + agecut + sex_1 + DL_pred_T1C_1
     model1 <- coxph(Surv(FFS.time,FFS.event) ~age + LDH + DL_pred + DL_pred_T2 + DL_pred_T1C_1, data = trainData)
     summary(model1)
     tra1 <- predict(model1, newdata = trainData,type = "lp")
     cd = concordance.index(x = (tra1), surv.time=trainData$FFS.time, surv.event=trainData$FFS.event,method = "noether")
     cd$c.index

     val_pred = predict(model1, newdata = valData,type = "lp")
     cd = concordance.index(x = (val_pred), surv.time = valData$FFS.time, surv.event = valData$FFS.event,method = "noether")
     cd$c.index

     summary(tra1)
     summary(val_pred)
     trainData$rad_sig = tra1
     valData$rad_sig = val_pred
}

#寻找最优cutoff value
{
     library(survMisc)
     z1 = trainData$rad_sig
     b2 = data.frame(z1)
     b2$t2 = trainData$FFS.time
     b2$d3 = trainData$FFS.event
     coxm1 <- coxph(Surv(t2, d3) ~ z1, data = b2)
     coxm1 <- cutp(coxm1)$z1
     cutoff = round(coxm1$z1[1],3)
     trainData$stra = ifelse(trainData$rad_sig < cutoff,0,1)
     valData$stra = ifelse(valData$rad_sig < cutoff,0,1)
}

# train high-low-risk plot
{     
     dd<-data.frame("surv.time" = trainData$FFS.time, "surv.event" = trainData$FFS.event,"strat" = trainData$stra)
     dev.new()
     km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
               x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
               leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
               , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

     yd = Surv(trainData$FFS.time ,trainData$FFS.event)
     km1=survfit(yd~1)
     summary(km1)

     hazard.ratio(x = trainData$stra, surv.time = trainData$FFS.time, surv.event = trainData$FFS.event)
     yd = Surv(trainData$FFS.time[trainData$stra == 0],trainData$FFS.event[trainData$stra == 0])
     km3=survfit(yd~1)
     summary(km3)

     yd = Surv(trainData$FFS.time[trainData$stra == 1],trainData$FFS.event[trainData$stra == 1])
     km4 = survfit(yd~1)
     summary(km4)
}

# val high-low-risk plot
{
     dd<-data.frame("surv.time" = valData$FFS.time, "surv.event" = valData$FFS.event,"strat" = valData$stra)
     dev.new()
     km.coxph.plot(formula.s=Surv(surv.time, surv.event) ~ strat, data.s=dd ,leg.inset=0.02,.lwd=c(2,2),
               x.label="Time (days)", y.label="Probability of survival",main="",.col = c('black','red'),
               leg.text=paste(c("Low risk", "High risk")," ",sep=""), leg.pos="bottomright", xlim=360*c(0,6.4)
               , show.n.risk=TRUE, n.risk.step=360, n.risk.cex=1, mark.time=T, v.line = c(1080,1800))

     yd = Surv(valData$FFS.time ,valData$FFS.event)
     km1=survfit(yd~1)
     summary(km1)

     hazard.ratio(x = valData$stra, surv.time = valData$FFS.time, surv.event = valData$FFS.event)
     yd = Surv(valData$FFS.time[valData$stra == 0],valData$FFS.event[valData$stra == 0])
     km3=survfit(yd~1)
     summary(km3)

     yd = Surv(valData$FFS.time[valData$stra == 1],valData$FFS.event[valData$stra == 1])
     km4 = survfit(yd~1)
     summary(km4)
}


# 临床变量分析 useless part
{   
     cp <- coxph(Surv(FFS.time,FFS.event) ~LDH, data = trainData)
     summary(cp)
     h = hazard.ratio(x = trainData$LDH, surv.time=trainData$FFS.time, surv.event=trainData$FFS.event)
     h$p.value
     #
     cp <- coxph(Surv(FFS.time,FFS.event) ~LDH, data = valData)
     summary(cp)
     h = hazard.ratio(x = valData$LDH, surv.time=valData$FFS.time, surv.event=valData$FFS.event)
     h$p.value

     # Nomogram对其他终点事件OS,DMS,RRS的预测作用
     #OS
     cp <- coxph(Surv(OS.time,OS.event) ~rad_sig, data = trainData)
     summary(cp)
     #
     cp <- coxph(Surv(OS.time,OS.event) ~rad_sig, data = valData)
     summary(cp)

     #DFS
     cp <- coxph(Surv(DFS.time,DFS.event) ~rad_sig, data = trainData)
     summary(cp)
     #
     cp <- coxph(Surv(DFS.time,DFS.event) ~rad_sig, data = valData)
     summary(cp)

     #LRS
     cp <- coxph(Surv(LRS.time,LRS.event) ~rad_sig, data = trainData)
     summary(cp)
     #
     cp <- coxph(Surv(LRS.time,LRS.event) ~rad_sig, data = valData)
     summary(cp)


     #临床表格
     alldata$group = ifelse(alldata$rand_group != 3,0,1)
     alldata$肿瘤家族史 = ifelse(alldata$肿瘤家族史>0,1,0)
     pacman::p_load(knitr, wakefield, MatchIt, tableone, captioner)
     table1 <- CreateTableOne(vars = clinic_ind[-c(12,13:16)], 
                              data = alldata, 
                              factorVars = clinic_ind[c(2,4,6,8,10,14:16,19:21,23,25:29)],
                              testNonNormal = kruskal.test,
                              strata = 'group')
     table1 <- print(table1, 
                    printToggle = FALSE, 
                    noSpaces = TRUE)
     kable(table1[,1:3],  
          align = 'c', 
          caption = 'Table 1: Comparison of unmatched samples')
     name1 = colnames(alldata)
     name1 = name1[c(34:37)]
     table1 <- CreateTableOne(vars = name1, 
                              data = alldata, 
                              factorVars = name1,
                              testNonNormal = kruskal.test,
                              strata = 'group')
     table1 <- print(table1, 
                    printToggle = FALSE, 
                    noSpaces = TRUE)
     kable(table1[,1:3],  
          align = 'c', 
          caption = 'Table 1: Comparison of unmatched samples')


     ClinicalVariableName = c('WHO病理类型')
     source("chisq-fisher-test.r")

     for (i in 1:length(ClinicalVariableName)){
     tmp <- ClinicalVariableName[i]
     tmp1 <- xtabs(as.formula(paste('~group+',tmp)),data=alldata);print(tmp1)
     tmp1 <- get.test.method(tmp1);print(tmp1)
     }
     New_EBV_data$group = ifelse(New_EBV_data$rand_group != 3,0,1)
     tmp1 <- xtabs(as.formula(paste('~group+','EBV_4k')),data=New_EBV_data);print(tmp1)
     tmp1 <- get.test.method(tmp1);print(tmp1)
     summary(trainData_EBV$EBV.DNA)
     summary(valData_EBV$EBV.DNA)
     summary(trainData$LDH)
     summary(valData$LDH)
     summary(trainData$CRP)
     summary(valData$CRP)
     summary(trainData$ALB)
     summary(valData$ALB)
     summary(trainData$HGB)
     summary(valData$HGB)
     summary(trainData$age)
     summary(valData$age)
}

# survival ROC
{     
     library("survivalROC")
     nob1<-NROW(trainData$rad_sig)
     t.1<-survivalROC(Stime = trainData$FFS.time,status = trainData$FFS.event,(trainData$rad_sig),predict.time = 1080, span=0.001*nob1^(-0.2))
     round(t.1$AUC,3)
     t.2<-survivalROC(Stime = trainData$FFS.time,status = trainData$FFS.event,(trainData$rad_sig),predict.time = 1800, span=0.001*nob1^(-0.2))
     round(t.2$AUC,3)
     dev.new()
     plot(t.1$FP, t.1$TP, type="l", xlim=c(0,1), ylim=c(0,1),xlab = c("False positive rate (%)"),ylab="True positive rate (%)"
          , col = rgb(254/255,67/255,101/255), lwd = 2, cex.lab=1.5)
     legend("bottomright", legend=c('3-year: AUC = 0.698','5-year: AUC = 0.711'),col=c(rgb(254/255,67/255,101/255),rgb(0/255,0/255,0/255)), lwd=2, cex=1.5,lty=c(1,1))
     lines(c(0,1), c(0,1), lty = 6,col = rgb(113/255,150/255,159/255),lwd=2.0)#画45度基
     lines(t.2$FP,t.2$TP,lty = 1,lwd =2, col=rgb(0/255,0/255,0/255))


     #验证集
     nob1<-NROW(valData$rad_sig)
     v.1<-survivalROC(Stime = valData$FFS.time,status = valData$FFS.event,(valData$rad_sig),predict.time = 1080, span=0.001*nob1^(-0.2))
     round(v.1$AUC,3)
     v.2<-survivalROC(Stime = valData$FFS.time,status = valData$FFS.event,(valData$rad_sig),predict.time = 1800, span=0.001*nob1^(-0.2))
     round(v.2$AUC,3)
     dev.new()
     plot(v.1$FP, v.1$TP, type="l", xlim=c(0,1), ylim=c(0,1),xlab = c("False positive rate (%)"),ylab="True positive rate (%)"
          , col = rgb(254/255,67/255,101/255), lwd = 2, cex.lab=1.5)
     lines(c(0,1), c(0,1), lty = 6,col = rgb(113/255,150/255,159/255),lwd=2.0)#画45度基
     lines(v.2$FP,v.2$TP,lty = 1,lwd =2, col=rgb(0/255,0/255,0/255))
     legend("bottomright", legend=c('3-year: AUC = 0.684','5-year: AUC = 0.655'),col=c(rgb(254/255,67/255,101/255),rgb(0/255,0/255,0/255)), lwd=2, cex=1.5,lty=c(1,1))
}

##nomogram
{     
     library(lattice);library(survival);library(Formula);library(ggplot2);library(Hmisc);library(rms)

     ddist0 <- datadist(trainData)
     options(datadist='ddist0')
     f <- cph(Surv(FFS.time, FFS.event) ~ age + LDH + DL_pred + DL_pred_T2 + DL_pred_T1C_1, surv = TRUE, x = T, y = T, data = trainData)
     surv.prob <- Survival(f) # 构建生存概率函数
     nom <- nomogram(f, fun=list(function(x) surv.prob(1080, x),function(x) surv.prob(1800, x)),
                    funlabel=c("3-year DFS rate","5-year DFS rate"),
                    fun.at=c(1.00,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0),
                    lp=F)
     dev.new()
     plot(nom, xfrac=.3,cex.axis=1.3, cex.var=1.3)
}

##calibration curve
{    
     f_3 <- cph(Surv(FFS.time, FFS.event) ~ rad_sig, surv = TRUE, x = T, y = T, data = trainData
               ,time.inc = 1080)
     cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='boot', B = 80,m=100,surv=TRUE, time.inc=1080) 
     f_5 <- cph(Surv(FFS.time, FFS.event) ~ rad_sig, surv = TRUE, x = T, y = T, data = trainData
               ,time.inc = 1800,identity.lty=2)
     cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='boot', B = 80,m=100,surv=TRUE, time.inc=1080) 
     source("E:/RStudio_file/test_ICT/HL-test.r")
     HLtest(cal_3)
     HLtest(cal_5)
     dev.new()
     opar <- par(no.readonly = TRUE)
     par(lwd = 1.2, lty = 1)
     x1 = c(rgb(0,112,255,maxColorValue = 255))
     plot(cal_3,lty = 1,pch = 16,errbar.col = x1,par.corrected = list(col=x1),conf.int=T,lwd = 1.2
          ,xlim = c(0.5,1),ylim = c(0.5,1),riskdist = F,col = "blue",axes = F)
     # plot(cal_3,lty = 1,pch = 16,conf.int=F,lwd = 1.2,xlim = c(0.6,1),ylim = c(0.6,1),riskdist = F,col = x1,axes = F)
     x2 = c(rgb(220,220,220,maxColorValue = 255))
     abline(0, 1, lty = 5, col=x2 ,lwd=1)
     x2 = c(rgb(209,73,85,maxColorValue = 255))
     plot(cal_5,lty = 1,pch = 18,errbar.col = x2,xlim = c(0.5,1),ylim = c(0.5,1),par.corrected = list(col=x2),
          col = "red",lwd = 1.2,riskdist = F,add=T,conf.int=T)
     # plot(cal_5, lty = 1,pch = 16,conf.int=F,lwd = 1.2,xlim = c(0.5,1),ylim = c(0.5,1),riskdist = F,col = x1,axes = F,add = T)
     axis(1,at=seq(0.5,1,0.1),labels=seq(0.5,1,0.1),pos=0.5)
     axis(2,at=seq(0.5,1,0.1),labels=seq(0.5,1,0.1),pos=0.5)
     par(opar)

     #验证集
     f_3 <- cph(Surv(FFS.time, FFS.event) ~ rad_sig, surv = TRUE, x = T, y = T, data = valData
               ,time.inc = 1080)
     cal_3 <- calibrate(f_3,  u=1080, cmethod='KM',method='boot', B = 80,m=50,surv=TRUE, time.inc=1080) 
     f_5 <- cph(Surv(FFS.time, FFS.event) ~ rad_sig, surv = TRUE, x = T, y = T, data = valData
               ,time.inc = 1800,identity.lty=2)
     cal_5 <- calibrate(f_5,  u=1800, cmethod='KM',method='boot', B = 80,m=50,surv=TRUE, time.inc=1080) 
     source("E:/RStudio_file/test_ICT/HL-test.r")
     HLtest(cal_3)
     HLtest(cal_5)
     dev.new()
     opar <- par(no.readonly = TRUE)
     par(lwd = 1.2, lty = 1)
     x1 = c(rgb(0,112,255,maxColorValue = 255))
     # plot(cal_3,lty = 1,pch = 16,errbar.col = x1,par.corrected = list(col=x1),conf.int=T,lwd = 1.2
     #      ,xlim = c(0.5,1),ylim = c(0.5,1),riskdist = F,col = "blue",axes = F)
     plot(cal_3,lty = 1,pch = 16,conf.int=F,lwd = 1.2,xlim = c(0.6,1),ylim = c(0.6,1),riskdist = F,col = x1,axes = F)
     x2 = c(rgb(220,220,220,maxColorValue = 255))
     abline(0, 1, lty = 5, col=x2 ,lwd=1)
     x2 = c(rgb(209,73,85,maxColorValue = 255))
     plot(cal_5,lty = 1,pch = 18,errbar.col = x2,xlim = c(0.6,1),ylim = c(0.6,1),par.corrected = list(col=x2),
          col = "red",lwd = 1.2,riskdist = F,add=T,conf.int=T)
     # plot(cal_5, lty = 1,pch = 16,conf.int=F,lwd = 1.2,xlim = c(0.5,1),ylim = c(0.5,1),riskdist = F,col = x1,axes = F,add = T)
     axis(1,at=seq(0.6,1,0.1),labels=seq(0.6,1,0.1),pos=0.6)
     axis(2,at=seq(0.6,1,0.1),labels=seq(0.6,1,0.1),pos=0.6)
     legend("bottomright", legend=c('3-year DFS: p = 0.76','5-year DFS: p = 0.46'),col=c('blue','red'), lwd=2, cex=1.5,lty=c(1,1))
     par(opar)
}

#比较两个模型
#比较c-index：cindex.comp
library(survcomp)
c1 <- concordance.index(x=age, surv.time=stime, surv.event=sevent,
                        method="noether")
c2 <- concordance.index(x=size, surv.time=stime, surv.event=sevent,
                        method="noether")
cindex.comp(c1, c2)
#比较c-index
# This function compares two hazard ratios from their betas and standard errors as computed by a Cox
# model for instance. The statistical test is a Student t test for dependent samples. The two hazard
# ratios must be computed from the same survival data.
hr1 <- hazard.ratio(x=age, surv.time=stime, surv.event=sevent)
hr2 <- hazard.ratio(x=size, surv.time=stime, surv.event=sevent)
hr.comp(hr1=hr1, hr2=hr2)
#来效果不好时，试一下下面的
# coxm2 <- coxph(Surv(stime, sevent) ~ size)
# hr.comp2(x1=age, beta1=coxm1$coefficients, se1=drop(sqrt(coxm1$var)),
#          x2=size, beta2=coxm2$coefficients, se2=drop(sqrt(coxm2$var)), n=length(age))
# bhr1 <- balanced.hazard.ratio(x=age, surv.time=stime, surv.event=sevent)
# bhr2 <- balanced.hazard.ratio(x=size, surv.time=stime, surv.event=sevent)
# bhr.comp(bhr1=bhr1, bhr2=bhr2)

#EBV-DNA分析

alldata = rbind(trainData,valData)
name1 = row.names(EBV_data)
New_EBV_data = alldata[name1,]
New_EBV_data$EBV.DNA = EBV_data$EBV.DNA
New_EBV_data$EBV_log = EBV_data$EBV_log
New_EBV_data$EBV_1k = EBV_data$EBV_1k
New_EBV_data$EBV_2k = EBV_data$EBV_2k
New_EBV_data$EBV_4k = EBV_data$EBV_4k
New_EBV_data$EBV_6k = EBV_data$EBV_6k

trainData_EBV = subset(New_EBV_data, rand_group != 3)
valData_EBV = subset(New_EBV_data, rand_group == 3)
#age +LDH + CRP  + DL_pred + DL_pred_T2 + agecut + sex_1 + DL_pred_T1C_1
model2 <- coxph(Surv(FFS.time,FFS.event) ~age + EBV_4k + LDH, data = trainData_EBV)
summary(model2)
tra1 <- predict(model2, newdata = trainData_EBV,type = "lp")
cd = concordance.index(x = (tra1), surv.time=trainData_EBV$FFS.time, surv.event=trainData_EBV$FFS.event,method = "noether")
cd$c.index

val_pred = predict(model2, newdata = valData_EBV,type = "lp")
cd = concordance.index(x = (val_pred), surv.time = valData_EBV$FFS.time, surv.event = valData_EBV$FFS.event,method = "noether")
cd$c.index

#model3
model3 <- coxph(Surv(FFS.time,FFS.event) ~age + LDH + DL_pred + DL_pred_T2 + EBV_4k+ DL_pred_T1C_1, data = trainData_EBV)
summary(model3)
tra1 <- predict(model3, newdata = trainData_EBV,type = "lp")
cd = concordance.index(x = (tra1), surv.time=trainData_EBV$FFS.time, surv.event=trainData_EBV$FFS.event,method = "noether")
cd$c.index

val_pred = predict(model3, newdata = valData_EBV,type = "lp")
cd = concordance.index(x = (val_pred), surv.time = valData_EBV$FFS.time, surv.event = valData_EBV$FFS.event,method = "noether")
cd$c.index
