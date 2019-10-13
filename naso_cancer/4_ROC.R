library(ROCR)
library(ggplot2)

sapply(c('train', 'test'), function(name) {
    d = read.csv(paste('/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_data/roc/121model.', name, '.roc.csv', sep = ''), header = F)
    pred = prediction(d$V1, d$V2)
    perf = performance(pred, "tpr", "fpr")
    auc = performance(pred, "auc")@y.values[[1]]
    png(paste('/home/tongxueqing/zhaox/ImageProcessing/naso_cancer/_plots/', name, '.roc.png', sep = ''))
    plot(perf, colorize = T)
    abline(a = 0, b = 1)
    dev.off()
    auc
})