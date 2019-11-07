suppressPackageStartupMessages(library(survcomp))
if (dir.exists("/wangshuo")) {
    root = "/wangshuo/zhaox"
} else {
    root = "/home/tongxueqing/zhao"
}
root = file.path(root, "ImageProcessing/mr_clinic_model/_data")

data = read.csv(file.path(root, "mr.iccfiltered.csv"), header = T, row.names = 1, stringsAsFactors = F)
labeldata = read.csv(file.path(root, "info.csv"), header = T, row.names = 1, stringsAsFactors = F)
data[,c("time", "event")] = labeldata[match(data$name, labeldata$name), c('time', 'event')]

common.name = intersect(data$name, labeldata$name)
data = data[data$name %in% common.name,]

data$time = labeldata$time[match(data$name, labeldata$name)]
data$event = labeldata$event[match(data$name, labeldata$name)]

features = colnames(data)[1:(ncol(data) - 7)]

