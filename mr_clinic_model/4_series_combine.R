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
# data[,c("time", "event")] = labeldata[match(data$name, labeldata$name), c('time', 'event')]
common.name = intersect(data$name, labeldata$name)
data = data[data$name %in% common.name,]
data$time = labeldata$time[match(data$name, labeldata$name)]
data$event = labeldata$event[match(data$name, labeldata$name)]

k_fold = read.csv(file.path(root, "ImageProcessing/combine_model/_data/k_fold_name.csv"), row.names = 1)
features = colnames(data)[!colnames(data) %in% c("name", "set", "time", "event")]

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

n = 20
models.all = lapply(1:4, function(k) {
    train = data[data$set == 0 & !data$name %in% k_fold[,k],]
    val = data[data$set == 0 & data$name %in% k_fold[,k],]

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
    models.k = list()
    for (model.list in models) {
        for (model in model.list) {
            models.k[[length(models.k) + 1]] = model
        }
    }
    models.k
})

result = matrix(rep(0, 3 * n), c(n, 3))
Ls = list()
for(w in 1:n / n) {
    L = lapply(models.all, function(models) {
        train = data[data$set == 0 & !data$name %in% k_fold[,k],]
        val = data[data$set == 0 & data$name %in% k_fold[,k],]
        models = Filter(Negate(is.null), models)
        cis = sapply(models, function(cox) {
            citr = to_ci(train, cox)
            civl = to_ci(val, cox)
            citr * w + (1 - w) * civl
        })
        model = models[[match(max(cis), cis)]]
        model
    })
    citr = mean(sapply(1:4, function(k) {
        tr = data[data$set == 0 & !data$name %in% k_fold[,k],]
        to_ci(tr, L[[k]])
    }))
    civl = mean(sapply(1:4, function(k) {
        vl = data[data$set == 0 & data$name %in% k_fold[,k],]
        to_ci(vl, L[[k]])
    }))
    result[w * n,] = c(w, citr, civl)
    Ls[[w * n]] = L
}


saveRDS(Ls, file.path(root, "ImageProcessing/mr_clinic_model/_data/Ls_mr.rds"))
write.csv(result, file.path(root, "ImageProcessing/mr_clinic_model/_outs/weight_choose_mr.out"))

L = Ls[[match(max(result[,3]), result[,3])]]
cits = combine_pred(test, L)
newpreds = sapply(L, function(cox) {
    pred = predict(cox, newdata = data, type = "lp")
})

cname = paste("mr_fold", 1:4, sep = '')
preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
preds[,cname] = newpreds[match(data$name, preds$name),]
write.csv(preds, file.path(root, "ImageProcessing/combine_model/_data/preds.csv"))