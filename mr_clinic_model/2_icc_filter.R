library(psych)
if (dir.exists("/wangshuo")) {
    root = "/wangshuo/zhaox"
} else {
    root = "/home/tongxueqing/zhao"
}

root = file.path(root, "ImageProcessing/mr_clinic_model/_data")

data = read.csv(file.path(root, "mr.merged.csv"), header = T, stringsAsFactors = F, row.names = 1)

features = colnames(data)[1:(ncol(data) - 5)]
infos = colnames(data)[(ncol(data) - 4):ncol(data)]
subfeatures = lapply(unique(data$series), function(serie) {
    reps = data[data$series == serie & data$isrep == 1,]
    orig = data[data$series == serie & data$isrep != 1,]
    orig = data[match(reps$name, orig$name),]
    icc.filter = sapply(features, function(f) {
        suppressMessages({
            result = ICC(cbind(reps[,f], orig[,f]))
            icc = result$results$ICC[3]
        })
        if (is.na(icc)) {
            icc = 1; p = 0
        } else {
            p = result$results$p[3]
        }
        c('icc' = icc, 'p' = p)
    })
    colnames(icc.filter)[icc.filter['icc',] > 0.75]
})

names = unique(data$name)
newdata = data.frame(name = names)
newdata[,infos] = data[match(newdata$name, data$name), infos]
for (serie in unique(data$series)) {
    sub = subfeatures[[serie]]
    seriesub = paste(sub, "_", serie, sep = "")
    newdata[,seriesub] = data[data$series == serie & data$isrep == 0, sub]
}

for (f in colnames(newdata)[!colnames(newdata) %in% infos]) {
    newdata[,f] = (newdata[,f] - mean(newdata[,f])) / sd(newdata[,f])
}
newdata[,c("series", "hosi", "isrep")] = NULL
write.csv(newdata, file.path(root, "mr.iccfiltered.csv"))