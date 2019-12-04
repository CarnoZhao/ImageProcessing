library(xlsx)
if (dir.exists("/wangshuo")) {
    root = "/wangshuo/zhaox"
} else {
    root = "/home/tongxueqing/zhao"
}

root = file.path(root, "ImageProcessing/gene_association/_data/")
files = list.files(file.path(root, "MR"), recursive = T, full.names = T)
files = files[grepl("(xlsx|csv)$", files)]

series = c("T1", "T2", "T1C")

T = 24:811
H = 210:295
L = 296:381
TH = 640:725
TL = 726:811
cnames = c()
data = data.frame()
for (serie in series) {
    subfiles = files[grepl(paste('/', sep = ""), files) & grepl(paste(serie, "_", sep = ""), files)]
    D = subfiles[!grepl("(HH|LL)", subfiles)]
    DH = subfiles[grepl("HH", subfiles)]
    DL = subfiles[grepl("LL", subfiles)]
    D = read.csv(D, header = TRUE, stringsAsFactors = F)
    DH = read.csv(DH, header = TRUE, stringsAsFactors = F)
    DL = read.csv(DL, header = TRUE, stringsAsFactors = F)
    D[,TL] = DL[,L]
    D[,TH] = DH[,H]
    name = D[,1]
    D = D[,T]
    D$name = name
    D$series = match(serie, series)
    if (length(cnames) != 0) {
        colnames(D) = cnames
    } else {
        cnames = colnames(D)
    }
    data = rbind(data, D)
}


# data$isrep = ifelse(data$hosi == 3, 1, 0)
# data$set = ifelse(data$hosi %in% c(1, 2, 3), 0, 1)
# data$hosi = hosis[data$hosi]
for (col in colnames(data)) {
    if (col != "name") {
        data[,col] = as.numeric(data[,col])
    }
}
write.csv(data, file.path(root, "mr.merged.csv"))
