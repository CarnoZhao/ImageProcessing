tra.data = py.T1C.tra
in.val.data = py.T1C.val
# ex.val.data = ex.py.T2
na_flag <- apply(is.na(tra.data), 2, sum)
tra.data <- tra.data[,which(na_flag == 0)]
name1=names(tra.data)
in.val.data <- in.val.data[,name1]
ex.val.data <- ex.val.data[,name1]
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
name1=names(tra.data)
in.val.data <- in.val.data[,name1]
# ex.val.data <- ex.val.data[,name1]

source('sort_out.r')
name1 = sort_out(tdata=tra.data,time=msag1$time,event=msag1$event)
tra.data = tra.data[,name1]
in.val.data = in.val.data[,name1]
# ex.val.data = ex.val.data[,name1]

tra.data$event = msag1$event
tra.data$time = msag1$time

in.val.data$event = msag2$event
in.val.data$time = msag2$time

# ex.val.data$event = ex.msag$event
# ex.val.data$time = ex.msag$time

tra.cox1<-coxph(as.formula(paste("Surv(time,event) ~ ",paste(name1,collapse = "+"))),data = tra.data)
summary(tra.cox1)
tra.cox2 = step(tra.cox1)
print(s <- summary( tra.cox2))

name2 = row.names(s$coefficients)

myvars=name2 %in% 'log.sigma.5.0.mm.3D_glrlm_LongRunLowGrayLevelEmphasis'
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
