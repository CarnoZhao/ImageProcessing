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
for (serie in unique(data$series)) {
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
    features = colnames(icc.filter)[icc.filter['icc',] > 0.75]
    print(length(features))
}

subdata = data[data$isrep == 0, c(features, infos)]
for (f in features) {
    subdata[,f] = (subdata[,f] - mean(subdata[,f])) / sd(subdata[,f])
}
write.csv(subdata, file.path(root, "mr.iccfiltered.csv"))