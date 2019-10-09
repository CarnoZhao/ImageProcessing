# 判断2个向量相同元素的个数

#wb1 = "C:/Users/18292/Desktop/115/test1.csv"
# data1 = read.table(wb1, header=TRUE, sep=",")
# 
# wb2 = "C:/Users/18292/Desktop/115/test2.csv"
# data2 = read.table(wb2, header=TRUE, sep=",")
# 
# seq1 = data1$name
# seq2 = data2$name
# seq1 = seq1[order(seq1)]
# seq2 = seq2[order(seq2)]
# sq = seq(0,249)
# nn = 0
# i_start = 1
# j_start = 1
# for(i in 1:1000){
#   # print(i)
#   if(j_start > length(seq2)){
#     break
#   }
#   if(seq1[i_start] > seq2[j_start]){
#     j_start = j_start + 1
#     next
#   }
#   if(seq1[i_start] == seq2[j_start]){
#     nn = nn + 1
#     sq[nn] = seq1[i_start]
#     i_start = i_start + 1
#     j_start = j_start + 1
#     next
#   }
#   if(seq1[i_start] < seq2[j_start]){
#     i_start = i_start + 1
#     next
#   }
# }


source('RD_PyradFile.r')
wb = "I:/research/ICT-NPC/Pyradiomics-feature/T1_feature.csv"
wb1 = "I:/research/ICT-NPC/Pyradiomics-feature_HH/T1_feature.csv"
wb2 = "I:/research/ICT-NPC/Pyradiomics-feature_LL/T1_feature.csv"
py.T1 = RD_PyradFile(wb,wb1,wb2)

source('RD_PyradFile.r')
wb = "I:/research/ICT-NPC/Pyradiomics-feature/T1C_feature.csv"
wb1 = "I:/research/ICT-NPC/Pyradiomics-feature_HH/T1C_feature.csv"
wb2 = "I:/research/ICT-NPC/Pyradiomics-feature_LL/T1C_feature.csv"
py.T1C = RD_PyradFile(wb,wb1,wb2)

source('RD_PyradFile.r')
wb = "I:/research/ICT-NPC/Pyradiomics-feature/T2_feature.csv"
wb1 = "I:/research/ICT-NPC/Pyradiomics-feature_HH/T2_feature.csv"
wb2 = "I:/research/ICT-NPC/Pyradiomics-feature_LL/T2_feature.csv"
py.T2 = RD_PyradFile(wb,wb1,wb2)

wb = "I:/research/ICT-NPC/PH_message.csv"
PH_clinic = read.csv(wb, header=TRUE,sep=",")
row.names(PH_clinic) = PH_clinic$影像科号
PH_clinic$source = 0
wb = "I:/research/ICT-NPC/BMC_message.csv"
BMC_clinic = read.csv(wb, header=TRUE,sep=",")
row.names(BMC_clinic) = BMC_clinic$病历号
BMC_clinic$source = BMC_clinic$组别数字

BMC_clinic$sex = BMC_clinic$性别    #1代表男，2代表女
PH_clinic$sex = PH_clinic$性别
BMC_clinic$age = BMC_clinic$年龄
PH_clinic$age = PH_clinic$年龄
BMC_clinic$MR.ID = BMC_clinic$影像号
PH_clinic$MR.ID = PH_clinic$影像科号
BMC_clinic$Histo.ID = as.vector(BMC_clinic$病历号)
PH_clinic$Histo.ID = as.vector(PH_clinic$病历号)

BMC_clinic$OS.time = round(30 * BMC_clinic$总生存时间)
BMC_clinic$OS.event = BMC_clinic$死亡2018更新
BMC_clinic$FFS.time = round(30 * BMC_clinic$无治疗失败时间)
BMC_clinic$FFS.event = BMC_clinic$治疗失败2018
BMC_clinic$DFS.time = round(30 * BMC_clinic$无远处转移时间)
BMC_clinic$DFS.event = BMC_clinic$远处转移2018更新
BMC_clinic$LRS.time = round(30 * BMC_clinic$无复发时间)
BMC_clinic$LRS.event = BMC_clinic$复发2018更新

PH_clinic$OS.time = round(30 * PH_clinic$OS.time)
PH_clinic$OS.event = PH_clinic$OS
PH_clinic$FFS.time = round(30 * PH_clinic$DFS.time)
PH_clinic$FFS.event = PH_clinic$DFS
PH_clinic$DFS.time = round(30 * PH_clinic$DMFS.time)
PH_clinic$DFS.event = PH_clinic$DMFS
PH_clinic$LRS.time = round(30 * PH_clinic$LRRFS.time)
PH_clinic$LRS.event = PH_clinic$LRRFS

ind = c('age','sex','OS.time','OS.event','FFS.time','FFS.event','DFS.time','DFS.event','LRS.time','LRS.event','source','Histo.ID','MR.ID')
data1 = PH_clinic[,ind]
data2 = BMC_clinic[,ind]
alldata = rbind(data1,data2)

alldata$FFS3_group = ifelse(alldata$FFS.time > 1080,1,ifelse(alldata$FFS.event == 1,0,-1))
alldata$OS3_group = ifelse(alldata$OS.time > 1080,1,ifelse(alldata$OS.event == 1,0,-1))

valid_data1 = subset(alldata, FFS3_group != -1)
summary(valid_data1$FFS3_group == 1)

valid_data2 = subset(alldata, OS3_group != -1)
summary(valid_data2$OS3_group == 1)

write.csv(valid_data1,file = "I:/research/ICT-NPC/classify_FFS3.csv")
# row.names(valid_data1) = valid_data1$Histo.ID

set.seed(1234)
nn = nrow(valid_data1)
ind = sample(1:nn,round(nn*0.3),replace = FALSE)
group = rep(0,nn)
group[ind] = 1
valid_data1$group = group

tra_clinic = subset(valid_data1, group == 0)
val_clinic = subset(valid_data1, group == 1)

name1 = row.names(tra_clinic)
df.tra.T1 = py.T1[name1,]
df.tra.T1C = py.T1C[name1,]
df.tra.T2 = py.T2[name1,]

name1 = row.names(val_clinic)
df.val.T1 = py.T1[name1,]
df.val.T1C = py.T1C[name1,]
df.val.T2 = py.T2[name1,]

