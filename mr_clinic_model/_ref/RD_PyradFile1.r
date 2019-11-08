RD_PyradFile1 <- function(wb,wb1,wb2,fea_name){
  py.T1.1 = read.csv(wb, header=F, sep=",")
  # colnames(py.T1.1) = fea_name
  row.names(py.T1.1) = py.T1.1$V1
  py.T1.1=py.T1.1[order(row.names(py.T1.1)),]
  x_end1 = ncol(py.T1.1)
  py.T1.1 = py.T1.1[,24:x_end1]
  
  py.T1.HH = read.csv(wb1, header=F, sep=",")
  # colnames(py.T1.HH) = fea_name
  row.names(py.T1.HH) = py.T1.HH$V1
  py.T1.HH=py.T1.HH[order(row.names(py.T1.HH)),]
  py.T1.HH = py.T1.HH[210:295]
  coln = colnames(py.T1.HH)
  py.T1.1[,coln] = py.T1.HH

  py.T1.LL = read.csv(wb2, header=F, sep=",")
  # colnames(py.T1.LL) = fea_name
  row.names(py.T1.LL) = py.T1.LL$V1
  py.T1.LL=py.T1.LL[order(row.names(py.T1.LL)),]
  py.T1.LL = py.T1.LL[296:381]
  coln = colnames(py.T1.LL)
  py.T1.1[,coln] = py.T1.LL
  
  return(py.T1.1)
}