RD_PyradFile <- function(wb,wb1,wb2){
  #library(xlsx)
  py.T1.1 = read.csv(file=wb,header=TRUE,sep=",")
  row.names(py.T1.1) = py.T1.1$X
  py.T1.1=py.T1.1[order(row.names(py.T1.1)),]
  x_end1 = ncol(py.T1.1)
  py.T1.1 = py.T1.1[,24:x_end1]
  
  py.T1.HH = read.csv(file=wb1,header=TRUE,sep=",")
  row.names(py.T1.HH) = py.T1.HH$X
  py.T1.HH=py.T1.HH[order(row.names(py.T1.HH)),]
  py.T1.HH = py.T1.HH[210:295]
  coln = colnames(py.T1.HH)
  py.T1.1[,coln] = py.T1.HH

  py.T1.LL = read.csv(file=wb2, header=TRUE,sep=",")
  row.names(py.T1.LL) = py.T1.LL$X
  py.T1.LL=py.T1.LL[order(row.names(py.T1.LL)),]
  py.T1.LL = py.T1.LL[296:381]
  coln = colnames(py.T1.LL)
  py.T1.1[,coln] = py.T1.LL
  
  return(py.T1.1)
}