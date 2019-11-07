suppressPackageStartupMessages({
    library(survcomp)
    library(glmnet)
    library(survival)
    library(caret)
    library(randomForestSRC)
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
    y = Surv(data$time, data$event)

    ## mrmr ##
    {
        mrmr.result = mrmr.cindex(data[,features.hr.filter], data$time, data$event, method = "noether")
        mrmr.result = sort(mrmr.result, decreasing = T)
        features.mrmr = names(mrmr.result)[1:round(length(mrmr.result) / 2)]
    }

    ## hazard ratio ##
    {
        hrs = as.data.frame(t(sapply(features, function(f) {
            hr = hazard.ratio(x = data[,f], surv.time = data$time, surv.event = data$event)
            p = hr$p.value
            cox = coxph(y ~ data[,f], data = data)
            pred = predict(cox, newdata = data, type = 'lp')
            ci = concordance.index(x = pred, surv.time = data$time, surv.event = data$event, method = "noether")
            ci = ci$c.index
            c("p" = p, "ci" = ci)
        })))
        features.hr.filter = rownames(hrs)[hrs$p < 0.05]
    }

    ## random forest ##
    {
        subdata = data[,c(features.mrmr, "time", "event")]
        rf = rfsrc(Surv(time, event) ~ . , data = subdata)
        rf = max.subtree(rf)
        features.rf = rf$topvars.1se
    }
    
    ## cox model ##
    {
        cox = coxph(y ~ ., data = data[,features.rf])
        sink("/dev/null")
        cox.new = step(cox, direction = c("both"))
        sink()
        L[[serie]] = cox.new
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
    preds = sapply(1:3, function(seire) {
        setdata = set[set$series == serie,]
        setdata = setdata[match(setnames, setdata$name),]
        pred = predict(L[[seire]], newdata = setdata, type = 'lp')
    })
    # preds = rowMeans(preds)、
    preds
}

preds = to_preds(test)
preds = rowMeans(preds)
testnames = unique(test$name)
testtime = test[match(testnames, test$name), "time"]
testevent = test[match(testnames, test$name), "event"]
ci = concordance.index(x = preds, surv.time = testtime, surv.event = testevent, method = "noether")
ci = ci$c.index