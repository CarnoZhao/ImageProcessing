suppressPackageStartupMessages({
    library(survcomp)
    library(glmnet)
    library(survival)
    library(caret)
    library(randomForestSRC)
    library(ggplot2)
    library(pheatmap)
})
if (dir.exists("/wangshuo")) {
    root = "/wangshuo/zhaox"
} else {
    root = "/home/tongxueqing/zhao"
}
root = file.path(root, "ImageProcessing/mr_clinic_model/_data")
set.seed(1)

data = read.csv(file.path(root, "mr.iccfiltered.csv"), header = T, row.names = 1, stringsAsFactors = F)
labeldata = read.csv(file.path(root, "clinic/info.csv"), header = T, row.names = 1, stringsAsFactors = F)
data[,c("time", "event")] = labeldata[match(data$name, labeldata$name), c('time', 'event')]
common.name = intersect(data$name, labeldata$name)
data = data[data$name %in% common.name,]
data$time = labeldata$time[match(data$name, labeldata$name)]
data$event = labeldata$event[match(data$name, labeldata$name)]

features = colnames(data)[1:(ncol(data) - 7)]

D = data
train = D[D$set == 0,]
test = D[D$set == 1,]
L = list()
cis = c()
for (serie in unique(data$series)) {
    data = train[train$series == serie,]

    ## mrmr ##
    if (T) {
        mrmr.result = mrmr.cindex(data[,features], data$time, data$event, method = "noether")
        mrmr.result = sort(mrmr.result, decreasing = T)
        features.mrmr = names(mrmr.result)[1:round(length(mrmr.result) / 3)]
    }

    ## hazard ratio ##
    if (T) {
        hrs = as.data.frame(t(sapply(features.mrmr, function(f) {
            hr = hazard.ratio(x = data[,f], surv.time = data$time, surv.event = data$event)
            p = hr$p.value
            cox = coxph(Surv(data$time, data$event) ~ data[,f], data = data)
            pred = predict(cox, newdata = data, type = 'lp')
            ci = concordance.index(x = pred, surv.time = data$time, surv.event = data$event, method = "noether")
            ci = ci$c.index
            c("p" = p, "ci" = ci)
        })))
        features.hr = rownames(hrs)[hrs$p < 0.05]
    }

    ## random forest ##
    if (T) {
        subdata = data[,c(features.hr, "time", "event")]
        rf = rfsrc(Surv(time, event) ~ . , data = subdata)
        rf = max.subtree(rf)
        features.rf = rf$topvars.1se
    }
    
    ## cox model ##
    if (T) {
        subdata = data = data[,c(features.rf, "time", "event")]
        cox = coxph(Surv(time, event) ~ . , data = subdata)
        sink("/dev/null")
        cox.new = step(cox, direction = c("both"))
        L[[serie]] = cox.new
        sink()
    }

    ## test ##
    testdata = test[test$series == serie,]
    pred = predict(cox.new, newdata = testdata, type = 'lp')
    ci = concordance.index(x = pred, surv.time = testdata$time, surv.event = testdata$event, method = "noether")
    ci = ci$c.index
    cis = c(cis, ci)
}

## train 3 series ##

to_preds = function(set) {
    setnames = unique(set$name)
    preds = sapply(1:3, function(serie) {
        setdata = set[set$series == serie,]
        setdata = setdata[match(setnames, setdata$name),]
        pred = predict(L[[serie]], newdata = setdata, type = 'lp')
    })
    preds
}

to_ci = function(set, weight = c(1, 1, 1)) {
    preds = to_preds(set)
    preds = preds * weight / sum(weight)
    preds = rowMeans(preds)
    setnames = unique(set$name)
    settime = set[match(setnames, set$name), "time"]
    setevent = set[match(setnames, set$name), "event"]
    ci = concordance.index(x = preds, surv.time = settime, surv.event = setevent, method = "noether")
    ci$c.index
}

citr = to_ci(train)
cits = to_ci(test)

print(citr)
print(cits)

if (F) {
    n = 100
    w1 = rep(1:n, n) / n
    w2 = rep(1:n, each = n) / n
    w3 = 1 - w1 - w2
    W = cbind(w1, w2, w3)
    res = as.data.frame(t(sapply(1:nrow(W), function(i) {
        weight = W[i,]
        citr = to_ci(train, weight)
        cits = to_ci(test, weight)
        c(weight, "citr" = citr, "cits" = cits)
    })))
    # colnames(res)[1:3] = c('w1', 'w2', 'w3')

    res$mean = (res$citr + res$cits) / 2
    for (name in c("citr", "cits", "mean", "bound")) {
        mat = matrix(res[,name], c(n, n))
        p = pheatmap(mat, cluster_cols = F, cluster_rows = F)
        ggsave(paste("/home/tongxueqing/zhao/ImageProcessing/mr_clinic_model/_plots/", name, ".heat.png", sep = ""), p)
    }
}
