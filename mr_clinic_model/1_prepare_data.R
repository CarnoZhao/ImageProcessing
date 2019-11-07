library(xlsx)
if (dir.exists("/wangshuo")) {
    root = "/wangshuo/zhaox"
} else {
    root = "/home/tongxueqing/zhao"
}

root = file.path(root, "ImageProcessing/mr_clinic_model/_data")
files = list.files(root, recursive = T, full.names = T)
files = files[grepl("(xlsx|csv)$", files)]

hosis = c("ZF", "ZF_add", "ZF_re", "GX")
series = c("T1", "T2", "T1C")

T = 24:811
H = 210:295
L = 296:381
TH = 640:725
TL = 726:811
cnames = c()
data = data.frame()
for (hosi in hosis) {
    for (serie in series) {
        subfiles = files[grepl(paste('/', hosi, '/', sep = ""), files) & grepl(paste(serie, "_", sep = ""), files)]
        D = subfiles[!grepl("(HH|LL)", subfiles)]
        DH = subfiles[grepl("HH", subfiles)]
        DL = subfiles[grepl("LL", subfiles)]
        if (grepl("xlsx$", D)) {
            D = read.xlsx(D, 1, stringsAsFactors = F)
            DH = read.xlsx(DH, 1, stringsAsFactors = F)
            DL = read.xlsx(DL, 1, stringsAsFactors = F)
        } else {
            D = read.csv(D, header = F, stringsAsFactors = F)
            DH = read.csv(DH, header = F, stringsAsFactors = F)
            DL = read.csv(DL, header = F, stringsAsFactors = F)
        }
        D[,TL] = DL[,L]
        D[,TH] = DH[,H]
        name = D[,1]
        D = D[,T]
        D$name = name
        D$series = match(serie, series)
        D$hosi = match(hosi, hosis)
        if (length(cnames) != 0) {
            colnames(D) = cnames
        } else {
            cnames = colnames(D)
        }
        write.csv(D, file.path(root, hosi, paste(serie, "_replaced.csv", sep = "")))
        data = rbind(data, D)
    }
}

data$isrep = ifelse(data$hosi == 3, 1, 0)
data$set = ifelse(data$hosi %in% c(1, 2, 3), 0, 1)
data$hosi = hosis[data$hosi]
for (col in colnames(data)) {
    if (col != "hosi") {
        data[,col] = as.numeric(data[,col])
    }
}
write.csv(data, file.path(root, "mr.merged.csv"))
