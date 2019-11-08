RD_PyradFile <- function(wb,wb1,wb2){
  library(xlsx)
  py.T1.1 = read.xlsx(file=wb,header=TRUE,sheetName = 'Sheet1')
  row.names(py.T1.1) = py.T1.1$name
  py.T1.1=py.T1.1[order(row.names(py.T1.1)),]
  x_end1 = ncol(py.T1.1)
  py.T1.1 = py.T1.1[,24:x_end1]
  
  py.T1.HH = read.xlsx(file=wb1,header=TRUE,sheetName = 'Sheet1')
  row.names(py.T1.HH) = py.T1.HH$name
  py.T1.HH=py.T1.HH[order(row.names(py.T1.HH)),]
  py.T1.HH = py.T1.HH[210:295]
  coln = colnames(py.T1.HH)
  py.T1.1[,coln] = py.T1.HH

  py.T1.LL = read.xlsx(file=wb2,header=TRUE,sheetName = 'Sheet1')
  row.names(py.T1.LL) = py.T1.LL$name
  py.T1.LL=py.T1.LL[order(row.names(py.T1.LL)),]
  py.T1.LL = py.T1.LL[296:381]
  coln = colnames(py.T1.LL)
  py.T1.1[,coln] = py.T1.LL
  
  return(py.T1.1)
}