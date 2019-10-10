#清理工作空间
rm(list=ls())

##################### 数据预处理 #######################
workbook0 <- "D:/Learning/111data/ICC/Finall_ICC_First.csv"
workbook1 <- "D:/Learning/111data/ICC/Finall_ICC_reader1.csv"
workbook2 <- "D:/Learning/111data/ICC/Finall_ICC_reader2.csv"

reader0 <- read.table(workbook0,header=TRUE,sep=",")
reader0 <- reader0[order(reader0[,'Unnamed..0'],decreasing=F),]
reader1 <- read.table(workbook1,header=TRUE,sep=",")
reader1 <- reader1[order(reader1[,'Unnamed..0'],decreasing=F),]
reader2 <- read.table(workbook2,header=TRUE,sep=",")
reader2 <- reader2[order(reader2[,'Unnamed..0'],decreasing=F),]

library(psych)

ifeatures1 <- c() #组内
ifeatures2 <- c() #组间
for (i in colnames(reader0)){
  icc <- ICC(cbind(reader0[,i],reader1[,i]))
  print(i)
  print(icc$results$ICC[3])
  if (icc$results$ICC[3] < 0.75){
    ifeatures1 <- c(ifeatures1,i)
    print('******************########################*******************##################*********************')
  }
}

for (i in colnames(reader0)){
  icc <- ICC(cbind(reader0[,i],reader2[,i]))
  print(i)
  print(icc$results$ICC[3])
  if (icc$results$ICC[3] < 0.75){
    ifeatures2 <- c(ifeatures2,i)
    print('******************########################*******************##################*********************')
  }
}

ifeatures <- intersect(ifeatures1,ifeatures2)[-1]
write.table(ifeatures3,file = 'icc25P_features.txt',sep = ',',row.names = FALSE,col.names = FALSE)



