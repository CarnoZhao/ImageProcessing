suppressPackageStartupMessages(library(ROCR))
suppressPackageStartupMessages(library(ggplot2))

args = commandArgs(T)
rocfile = args[1]
plotfile = args[2]

auc = sapply(c('train', 'test'), function(name) {
    d = read.csv(paste(rocfile, name, 'csv', sep = '.'), header = F)
    pred = prediction(d$V1, d$V2)
    perf = performance(pred, "tpr", "fpr")
    auc = performance(pred, "auc")@y.values[[1]]
    png(paste(plotfile, name, 'png', sep = '.'))
    plot(perf, colorize = T)
    abline(a = 0, b = 1)
    dev.off()
    auc
})

cat("\t\tAUC\n")
cat(paste("train\t", round(auc['train'], digits = 3), '\n', sep = ''))
cat(paste("test\t", round(auc['test'], digits = 3), '\n', sep = ''))