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
set.seed(0)

data = read.csv(file.path(root, "ImageProcessing/mr_clinic_model/_data/mr.iccfiltered.csv"), header = T, row.names = 1, stringsAsFactors = F)
labeldata = read.csv(file.path(root, "ImageProcessing/mr_clinic_model/_data/clinic/info.csv"), header = T, row.names = 1, stringsAsFactors = F)
data[,c("time", "event")] = labeldata[match(data$name, labeldata$name), c('time', 'event')]
common.name = intersect(data$name, labeldata$name)
data = data[data$name %in% common.name,]
data$time = labeldata$time[match(data$name, labeldata$name)]
data$event = labeldata$event[match(data$name, labeldata$name)]

to_ci = function(set, cox = cox.new) {
    pred = predict(cox, newdata = set, type = "lp")
    ci = concordance.index(x = pred, surv.time = set$time, surv.event = set$event, method = "noether")  
    ci$c.index  
    # paste(round(ci$c.index, 3), " (", round(ci$lower, 3), "-", round(ci$upper, 3), ") ", signif(ci$p.value, 3), sep = "")
}

combine_pred = function(set, models = L) {
    names = set[set$series == 1,]$name
    time = set[set$series == 1,]$time
    event = set[set$series == 1,]$event
    preds = sapply(1:3, function(s) {
        s.set = set[set$series == s,]
        s.set = s.set[match(s.set$name, names),]
        pred = predict(models[[s]], newdata = s.set, type = "lp")
    })
    # pred = rowMeans(preds)
    pred = apply(preds, 1, max)
    ci = concordance.index(x = pred, surv.time = time, surv.event = event, method = "noether")
    ci$c.index
}


pat_set = read.table(file.path(root, "ImageProcessing/combine_model/_data/new_set.txt"))$V1

trainAll = data[data$set == 0 & data$name %in% pat_set,]
valAll = data[data$set == 0 & !data$name %in% pat_set,]
test = data[data$set == 1,]

n = 20
models.all = lapply(1:3, function(s) {
    train = trainAll[trainAll$series == s,]
    val = valAll[valAll$series == s,]
    features = colnames(train)[complete.cases(t(train))]
    features = features[!grepl("(name|set|time|event|series)", features)]

    # selection
    methods = sapply(1:15, function(x) as.numeric(intToBits(x)[1:4]))
    subfeatures.list = apply(methods, 2, function(method) {
        subfeatures = features
        ## hazard ratio ##
        if (method[1]) try({
            hrs = as.data.frame(t(sapply(subfeatures, function(f) {
                hr = hazard.ratio(x = train[,f], surv.time = train$time, surv.event = train$event)
                p = hr$p.value
                cox = coxph(Surv(train$time, train$event) ~ train[,f], data = train)
                pred = predict(cox, data = train, type = 'lp')
                ci = concordance.index(x = pred, surv.time = train$time, surv.event = train$event, method = "noether")
                ci = ci$c.index
                c("p" = p, "ci" = ci)
            })))
            subfeatures = rownames(hrs)[hrs$p < 0.05]
        })

        ## correlate ##
        if (method[2]) try({
            cors = cor(train[,subfeatures])
            notcors = c()
            for (f in colnames(cors)) {
                if (!f %in% notcors & all(!rownames(cors)[cors[,f] > 0.9] %in% notcors)) {
                    notcors = c(notcors, f)
                }
            }
            subfeatures = notcors
        })

        ## mrmr ##
        if (method[3]) try({
            mrmr.result = mrmr.cindex(train[,subfeatures], train$time, train$event, method = "noether")
            mrmr.result = sort(mrmr.result, decreasing = T)
            subfeatures = names(mrmr.result)[1:min(20, length(mrmr.result))]
        })

        ## random forest ##
        if (method[4]) try({
            subtrain = train[,c(subfeatures, "time", "event")]
            rf = rfsrc(Surv(time, event) ~ . , data = subtrain)
            rf = max.subtree(rf)
            subfeatures = rf$topvars.1se
        })
        subfeatures
    })

    # val
    models = lapply(subfeatures.list, function(subfeatures) {
        model.list = list()
        while (length(subfeatures) != 0 && length(subfeatures) <= 20) {
            subtrain = train[,c(subfeatures, "time", "event")]
            cox = coxph(Surv(time, event) ~ . , data = subtrain)
            sink("/dev/null")
            cox = step(cox, direction = c("both"))
            sink()
            model.list[[length(model.list) + 1]] = cox
            summ = summary(cox)
            summ = as.data.frame(summ$coefficients)
            if (all(summ[,"Pr(>|z|)"] < 0.05)) {
                break
            } else {
                subfeatures = rownames(summ)[summ[,"Pr(>|z|)"] < 0.05]
            }
        }
        if (length(model.list) == 0) {NULL}
        else {model.list}
    })
    do.call(c, models)
})

n = 20
result = matrix(rep(0, 4 * n), c(n, 4))
cits = 0
Ls = list()
for(w in 1:n / n) {
    L = lapply(1:3, function(s) {
        models = models.all[[s]]
        train = trainAll[trainAll$series == s,]
        val = valAll[valAll$series == s,]
        models = Filter(Negate(is.null), models)
        cis = sapply(models, function(cox) {
            citr = to_ci(train, cox)
            civl = to_ci(val, cox)
            citr * w + (1 - w) * civl
        })
        model = models[[match(max(cis), cis)]]
        model
    })
    citr = combine_pred(trainAll, L)
    civl = combine_pred(valAll, L)
    cits = combine_pred(test, L)
    result[w * n,] = c(w, citr, civl, cits)
    Ls[[w * n]] = L
}
result
saveRDS(Ls, file.path(root, "ImageProcessing/mr_clinic_model/_data/Ls_mr.rds"))

Ls = readRDS(file.path(root, "ImageProcessing/mr_clinic_model/_data/Ls_mr.rds"))
result = t(sapply(1:n / n, function(w) {
    L = Ls[[w * n]]
    citr = combine_pred(trainAll, L)
    civl = combine_pred(valAll, L)
    cits = combine_pred(test, L)
    c(w, citr, civl, cits)
}))
result
result[order(result[,3]),]
L = Ls[[match(max(result[,3]), result[,3])]]

split.result = t(sapply(1:n / n, function(w) {
    L = Ls[[w * n]]
    citr = sapply(3:3, function(s) to_ci(trainAll[trainAll$series == s,], L[[s]]))
    civl = sapply(3:3, function(s) to_ci(valAll[valAll$series == s,], L[[s]]))
    cits = sapply(3:3, function(s) to_ci(test[test$series == s,], L[[s]]))
    c(w, citr, civl, cits)
}))
split.result

newpreds = sapply(1:3, function(s) {
    cox = L[[s]]
    pred = predict(cox, newdata = data[data$series == s,], type = "lp")
})

cname = paste("mr_serie", 1:3, sep = '')
preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
preds[,grepl("(mr|cli)_fold", colnames(preds))] = NULL
# preds[,cname] = newpreds[match(data$name, preds$name),]
preds[,"sig_mr"] = apply(newpreds, 1, max)
preds$set = ifelse(preds$name %in% test$name, 1, 0)
write.csv(preds, file.path(root, "ImageProcessing/combine_model/_data/preds.csv"))


### 
m1 = coxph(Surv(time, event) ~ log.sigma.4.0.mm.3D_glszm_GrayLevelNonUniformity + original_shape_Flatness + wavelet.LH_glcm_ClusterShade, data = trainAll[trainAll$series == 1,])
m2 = coxph(Surv(time, event) ~ original_shape_Flatness + log.sigma.2.0.mm.3D_gldm_DependenceNonUniformityNormalized + wavelet.LL_gldm_DependenceNonUniformity + log.sigma.2.0.mm.3D_glrlm_RunLengthNonUniformityNormalized
, data = trainAll[trainAll$series == 2,])
m3 = coxph(Surv(time, event) ~ original_shape_Flatness + original_shape_Maximum2DDiameterColumn+ original_shape_Maximum2DDiameterRow + original_shape_SurfaceVolumeRatio + original_glszm_SizeZoneNonUniformity
, data = trainAll[trainAll$series == 3,])
L = list(m1, m2, m3)