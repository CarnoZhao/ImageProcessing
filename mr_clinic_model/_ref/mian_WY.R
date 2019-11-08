source('RD_PyradFile.r')
wb = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature/T1_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature-Wave-HH/T1_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature-Wave-LL/T1_feature.xlsx"
py.T1 = RD_PyradFile(wb,wb1,wb2)

wb = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature/T1C_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature-Wave-HH/T1C_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature-Wave-LL/T1C_feature.xlsx"
py.T1C = RD_PyradFile(wb,wb1,wb2)

wb = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature/T2_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature-Wave-HH/T2_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/new-feature/Pyradiomics-feature-Wave-LL/T2_feature.xlsx"
py.T2 = RD_PyradFile(wb,wb1,wb2)
Fea_name = colnames(py.T2)
###重复性实验30例
source('repeat_feature.r')
wb = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature/T1_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature-Wave-HH/T1_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature-Wave-LL/T1_feature.xlsx"
re.py.T1 = RD_PyradFile(wb,wb1,wb2)
coln = names(py.T1)
for(col in coln) {
  re.py.T1[,col] = as.numeric(as.character(re.py.T1[,col]))
  py.T1[,col] = as.numeric(as.character(py.T1[,col]))
}
re.T1.fea = repeat_feature(py.T1,re.py.T1)

wb = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature/T1C_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature-Wave-HH/T1C_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature-Wave-LL/T1C_feature.xlsx"
re.py.T1C = RD_PyradFile(wb,wb1,wb2)
coln = names(py.T1C)
for(col in coln) {
  re.py.T1C[,col] = as.numeric(as.character(re.py.T1C[,col]))
  py.T1C[,col] = as.numeric(as.character(py.T1C[,col]))
}
re.T1C.fea = repeat_feature(py.T1C,re.py.T1C)

wb = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature/T2_feature.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature-Wave-HH/T2_feature.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/new-feature/Re-Pyradiomics-feature-Wave-LL/T2_feature.xlsx"
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


#增加的21例MR数据
source('RD_PyradFile1.r')
wb = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature/T1_feature.csv"
wb1 = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature-Wavelet-HH/T1_feature.csv"
wb2 = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature-Wavelet-LL/T1_feature.csv"
py.T1_add = RD_PyradFile1(wb,wb1,wb2,Fea_name)
colnames(py.T1_add) = Fea_name
name1 = colnames(py.T1)
py.T1_add = py.T1_add[,name1]
py.T1 = rbind(py.T1,py.T1_add)

wb = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature/T1C_feature.csv"
wb1 = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature-Wavelet-HH/T1C_feature.csv"
wb2 = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature-Wavelet-LL/T1C_feature.csv"
py.T1C_add = RD_PyradFile1(wb,wb1,wb2,Fea_name)
colnames(py.T1C_add) = Fea_name
name1 = colnames(py.T1C)
py.T1C_add = py.T1C_add[,name1]
py.T1C = rbind(py.T1C,py.T1C_add)

wb = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature/T2_feature.csv"
wb1 = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature-Wavelet-HH/T2_feature.csv"
wb2 = "I:/research/ZhangFan+BingLi+MR/New_1911/add_21/Pyradiomics-feature-Wavelet-LL/T2_feature.csv"
py.T2_add = RD_PyradFile1(wb,wb1,wb2,Fea_name)
colnames(py.T2_add) = Fea_name
name1 = colnames(py.T2)
py.T2_add = py.T2_add[,name1]
py.T2 = rbind(py.T2,py.T2_add)

name1 = row.names(py.T1)
py.T1C = py.T1C[name1,]
py.T2 = py.T2[name1,]
wb = 'I:/research/ZhangFan+BingLi+MR/ClinicMessageForAnalysis.xlsx'
clinic_data = read.xlsx(wb,header = T,encoding = "UTF-8",sheetName = "ClinicMessageForAnalysis")
row.names(clinic_data) = clinic_data$病案号
clinic_data = clinic_data[name1,]
clinic_data$s.time = round(clinic_data$无失败生存时间 * 30)
clinic_data$s.event = clinic_data$发生治疗失败

set.seed(4321)
nn = nrow(clinic_data)
ind = sample(1:nn,0.7*nn,replace = F)
data_cohort = rep(1,nn)
data_cohort[ind] = 0
py.T1$data_cohort = data_cohort
py.T1C$data_cohort = data_cohort
py.T2$data_cohort = data_cohort
clinic_data$data_cohort = data_cohort
clinic_data.tra = subset(clinic_data, data_cohort == 0)
clinic_data.val = subset(clinic_data, data_cohort == 1)

#数据划分+特征筛选
tra.data = subset(py.T2, data_cohort == 0)
in.val.data = subset(py.T2, data_cohort == 1)
na_flag <- apply(is.na(tra.data), 2, sum)
tra.data <- tra.data[,which(na_flag == 0)]
name1 = names(tra.data)
in.val.data <- in.val.data[,name1]

for(i in 1:length(name1)){
  tra.data[,i] = scale(tra.data[,i])
  xm = attr(tra.data[,i],"scaled:center");xv = attr(tra.data[,i],"scaled:scale")
  in.val.data[,i] = scale(in.val.data[,i],center = xm,scale = xv)
}
na_flag <- apply(is.na(tra.data), 2, sum)
tra.data <- tra.data[,which(na_flag == 0)]
in.val.data <- in.val.data[,which(na_flag == 0)]
name1 = names(tra.data)
in.val.data <- in.val.data[,name1]

source('sort_out.r')
name1 = sort_out(tdata = tra.data,time=clinic_data.tra$s.time,event=clinic_data.tra$s.event)
tra.data = tra.data[,name1]
library(prodlim)
library(survival)
library(survcomp)
Features1 = colnames(tra.data)
Fmrmr<-mrmr.cindex(x = tra.data,surv.time = clinic_data.tra$s.time,surv.event = clinic_data.tra$s.event,method="norther")
mrmr_sort1<-Features1[rev(order(Fmrmr))]

For_tra_val = mrmr_sort1
train_data = tra.data[,For_tra_val]
train_data$s.time = clinic_data.tra$s.time
train_data$s.event = clinic_data.tra$s.event

val_data = in.val.data[,For_tra_val]
val_data$s.time = clinic_data.val$s.time
val_data$s.event = clinic_data.val$s.event

cox1 <- coxph(as.formula(paste("Surv(s.time,s.event) ~ ",paste(For_tra_val,collapse = "+"))),train_data)
summary(cox1)

#利用AIC优化模型
cox2 <- step(cox1)#with a default of "both"
print(s <- summary(cox2))
name2 = row.names(s$coefficients)

myvars=name2 %in% 'original_gldm_LargeDependenceHighGrayLevelEmphasis'
name2=name2[!myvars]
cox2<-coxph(as.formula(paste("Surv(s.time,s.event) ~ ",paste(name2,collapse = "+"))),data = train_data)
summary(cox2)

tra1<-predict(cox2, newdata =tra.data,type="lp")
h = hazard.ratio(x= (tra1), surv.time=train_data$s.time, surv.event=train_data$s.event)
h$hazard.ratio
h$lower
h$upper
h$p.value
cd = concordance.index(x=(tra1),surv.time=train_data$s.time, surv.event=train_data$s.event,method="noether")
cd$c.index

in.val<-predict(cox2, newdata =val_data, type="lp")

h = hazard.ratio(x= (in.val), surv.time=val_data$s.time, surv.event=val_data$s.event)
h$hazard.ratio
h$lower
h$upper
h$p.value
cd = concordance.index(x=(in.val),surv.time=val_data$s.time, surv.event=val_data$s.event,method="noether")
cd$c.index

#外部验证
rname = row.names(ex.msag)
wb = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature/T1_feature_test.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature_wavelet_HH/T1_feature_test.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature_wavelet_LL/T1_feature_test.xlsx"
ex.py.T1 = RD_PyradFile(wb,wb1,wb2)
ex.py.T1 = ex.py.T1[rname,]
cname = names(py.T1)
ex.py.T1 = ex.py.T1[,cname]
for(col in cname) {
  ex.py.T1[,col] = as.numeric(as.character(ex.py.T1[,col]))
}

wb = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature/T1_feature_test.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature_wavelet_HH/T1_feature_test.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature_wavelet_LL/T1_feature_test.xlsx"
ex.py.T1C = RD_PyradFile(wb,wb1,wb2)
ex.py.T1C = ex.py.T1C[rname,]
cname = names(py.T1C)
ex.py.T1C = ex.py.T1C[,cname]
for(col in cname) {
  ex.py.T1C[,col] = as.numeric(as.character(ex.py.T1C[,col]))
}

wb = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature/T1_feature_test.xlsx"
wb1 = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature_wavelet_HH/T1_feature_test.xlsx"
wb2 = "I:/research/ZhangFan+BingLi+MR/ZhangFan_BingLi_GuangXi_exteralVal/New_MR-feature/py-feature_wavelet_LL/T1_feature_test.xlsx"
ex.py.T2 = RD_PyradFile(wb,wb1,wb2)
ex.py.T2 = ex.py.T2[rname,]
cname = names(py.T2)
ex.py.T2 = ex.py.T2[,cname]
for(col in cname) {
  ex.py.T2[,col] = as.numeric(as.character(ex.py.T2[,col]))
}


