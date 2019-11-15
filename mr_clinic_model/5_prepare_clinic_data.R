# load packages
if (T) {
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
}

# laod data
if (T) {
    data = read.csv(file.path(root, "ImageProcessing/combine_model/_data/clinicinfo.csv"), stringsAsFactors = F, row.names = 1)
    data = data[,complete.cases(t(data))]
    k_fold = read.csv(file.path(root, "ImageProcessing/combine_model/_data/k_fold_name.csv"), row.names = 1)
    features = colnames(data)[!grepl("(name|set|event|time)", colnames(data))]
}


to_ci = function(set, cox = cox.new) {
    pred = predict(cox, newdata = set, type = "lp")
    ci = concordance.index(x = pred, surv.time = set$time, surv.event = set$event, method = "noether")    
    ci$c.index
}

combine_pred = function(set, models = L) {
    preds = sapply(models, function(cox) {
        pred = predict(cox, newdata = set, type = "lp")
    })
    pred = rowMeans(preds)
    ci = concordance.index(x = pred, surv.time = set$time, surv.event = set$event, method = "noether")
    ci$c.index
}

train = data[data$set == 0 & !data$name %in% k_fold[,2],]
val = data[data$set == 0 & data$name %in% k_fold[,2],]
test = data[data$set == 1,]


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
        subfeatures = names(mrmr.result)[1:min(10, length(mrmr.result))]
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
    while (length(subfeatures) != 0 && length(subfeatures) <= 10) {
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
Ls = do.call(c, models)

Ls = readRDS("/home/tongxueqing/zhao/ImageProcessing/mr_clinic_model/_data/Ls_clinic_biHB.rds")
result = t(sapply(Ls, function(cox) {
    citr = to_ci(train, cox)
    civl = to_ci(val, cox)
    cits = to_ci(test, cox)
    c(citr, civl, cits)
}))
result[order(result[,2]),]
saveRDS(Ls, "/home/tongxueqing/zhao/ImageProcessing/mr_clinic_model/_data/Ls_clinic_biHB.rds")
write.csv(result, "/home/tongxueqing/zhao/ImageProcessing/mr_clinic_model/_outs/weight_choose_clinic_biHB.out")

L = Ls[[match(max(result[,2]), result[,2])]]
cits = to_ci(test, L)
newpred = predict(L, newdata = data, type = "lp")

cname = "sig_cli"
preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
preds[,grepl("cli_fold", colnames(preds))] = NULL
preds[,cname] = newpred[match(data$name, preds$name)]
preds$set = data$set[match(data$name, preds$name)]
write.csv(preds, file.path(root, "ImageProcessing/combine_model/_data/preds.csv"))