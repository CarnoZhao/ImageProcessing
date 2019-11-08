repeat_feature <- function(DWI.data,DWI.1){
  rname=row.names(DWI.1)
  cname=names(DWI.1)
  data=DWI.data[rname,]
  n1=length(cname);n2=ncol(data)
  if(n1 != n2)
  {stop("First and second must be same number of number")
  }
  #计算单个特征的ICC
  library(psych)
  Ficc.inter<-rep(0,n1)
  Ficc.inter.p<-rep(0,n1)
  ind2=rep(0,n1)
  for (i in 1:n1){
    # if(i == 16){
    #   print(data[,i])
    #   print('******')
    #   print(DWI.1[,i])
    # }
    ind2[i]=i
    icc<-ICC(cbind(data[,i],DWI.1[,i]))
    if (is.na(icc$results$ICC[3])){
      Ficc.inter[i]<-1
      Ficc.inter.p[i]<-0
    }
    else
    {
      Ficc.inter[i]<-icc$results$ICC[3]
      Ficc.inter.p[i]<-icc$results$p[3]
    }
    
  }
  rad.fearure<-data.frame(cname)
  rad.fearure$icc=Ficc.inter
  rad.fearure$icc.p=Ficc.inter.p
  rad.fearure$index=ind2
  return(rad.fearure)
}