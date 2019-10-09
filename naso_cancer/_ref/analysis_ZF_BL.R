tra.data = py.T2.tra
in.val.data = py.T2.val

na_flag <- apply(is.na(tra.data), 2, sum)
tra.data <- tra.data[,which(na_flag == 0)]
in.val.data <- in.val.data[,which(na_flag == 0)]
# ex.val.data <- ex.val.data[,which(na_flag == 0)]

name1=names(tra.data)
for(i in 1:length(name1)){
  tra.data[,i]=scale(tra.data[,i])
  xm=attr(tra.data[,i],"scaled:center");xv=attr(tra.data[,i],"scaled:scale")
  in.val.data[,i]=scale(in.val.data[,i],center = xm,scale = xv)
  # ex.val.data[,i]=scale(ex.val.data[,i],center = xm,scale = xv)
}
na_flag <- apply(is.na(tra.data), 2, sum)
tra.data <- tra.data[,which(na_flag == 0)]
in.val.data <- in.val.data[,which(na_flag == 0)]
# ex.val.data <- ex.val.data[,which(na_flag == 0)]

source('sort_out.r')
name1 = sort_out(tdata=tra.data,time=msag1$time,event=msag1$event)
tra.data = tra.data[,name1]
in.val.data = in.val.data[,name1]

x <- as.matrix(tra.data)

Time = as.double(msag1$time)
Status = as.double(msag1$event)
surv = Surv(Time,Status)  #==1写不写都行，做cox回归，两者必须是numeric类型
library(glmnet)
###########lasso-cox
set.seed(213)
cv.fit<-cv.glmnet(x,surv,nfolds=7, family="cox",nlambda = 200, alpha=1)   #默认100次lambda实验
dev.new()
plot(cv.fit,xlab = 'log(λ)')  #绘制cv变化图
coef.min = coef(cv.fit, s = "lambda.min") #指定λ值，抓取出某一个模型的系数:观察不为零的系数lambda.1se
active.min = which(coef.min != 0)  #选出的特征的标号
non_0 = coef.min[active.min] #非零系数值
#coef.min  #非零系数和对应协变量
choose<-names(coef.min [coef.min[,1]!=0,]) #选择的变量名

tra.data$time = msag1$time
tra.data$event = msag1$event

in.val.data$time = msag2$time
in.val.data$event = msag2$event

# ex.val.data$time = ex.msag$time
# ex.val.data$event = ex.msag$event


tra.cox1<-coxph(as.formula(paste("Surv(time,event) ~ ",paste(choose,collapse = "+"))),data = tra.data)
summary(tra.cox1)
tra.cox2 = step(tra.cox1)
print(s <- summary( tra.cox2))

name2 = row.names(s$coefficients)

myvars=name2 %in% 'log.sigma.3.0.mm.3D_firstorder_Mean'
name2=name2[!myvars]
tra.cox2<-coxph(as.formula(paste("Surv(time,event) ~ ",paste(name2,collapse = "+"))),data = tra.data)
summary(tra.cox2)


tra1<-predict(tra.cox2, newdata =tra.data,type="lp")
h = hazard.ratio(x= (tra1), surv.time=tra.data$time, surv.event=tra.data$event)
h$hazard.ratio
h$lower
h$upper
h$p.value
cd = concordance.index(x=(tra1),surv.time=tra.data$time, surv.event=tra.data$event,method="noether")
cd$c.index

in.val<-predict(tra.cox2, newdata =in.val.data, type="lp")

h = hazard.ratio(x= (in.val), surv.time=in.val.data$time, surv.event=in.val.data$event)
h$hazard.ratio
h$lower
h$upper
h$p.value
cd = concordance.index(x=(in.val),surv.time=in.val.data$time, surv.event=in.val.data$event,method="noether")
cd$c.index

# ex.val<-predict(tra.cox2, newdata =ex.val.data, type="lp")
# 
# h = hazard.ratio(x= (ex.val), surv.time=ex.val.data$time, surv.event=ex.val.data$event)
# h$hazard.ratio
# h$lower
# h$upper
# h$p.value
# cd = concordance.index(x=(ex.val),surv.time=ex.val.data$time, surv.event=ex.val.data$event,method="noether")
# cd$c.index
# 
# val.data = rbind(in.val.data,ex.val.data)
# 
# val.pro<-predict(tra.cox2, newdata =val.data, type="lp")
# 
# h = hazard.ratio(x= (val.pro), surv.time=val.data$time, surv.event=val.data$event)
# h$hazard.ratio
# h$lower
# h$upper
# h$p.value
# cd = concordance.index(x=(val.pro),surv.time=val.data$time, surv.event=val.data$event,method="noether")
# cd$c.index

