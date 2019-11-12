# load packages and root path
if (T) {
    if (T) { 
        suppressPackageStartupMessages({
            library(survcomp)
            library(glmnet)
            library(survival)
            library(caret)
            library(randomForestSRC)
            library(ggplot2)
            library(pheatmap)
            library(survMisc)
        })
        if (dir.exists("/wangshuo")) {
            root = "/wangshuo/zhaox"
        } else {
            root = "/home/tongxueqing/zhao"
        }
        set.seed(1)
    }

    preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
    ZFinfo = read.csv(file.path(root, "ImageProcessing/combine_model/_data/ZFinfo.csv"), row.names = 1, stringsAsFactors = F)
    GXinfo = read.csv(file.path(root, "ImageProcessing/combine_model/_data/GXinfo.csv"), row.names = 1, stringsAsFactors = F)
    GXinfo[colnames(ZFinfo)[!colnames(ZFinfo) %in% colnames(GXinfo)]] = NA
    info = rbind(ZFinfo, GXinfo)
    colnames(info) = c(
            "smoke", "family.history", "gender", "age", "body.status", "neuron", "EVB", "HB", "LDH", "sarcoma", "necrosis", "lymphocyte", "T.read", "N.read", "N.cut.N3b", "total.cut", "total.cut.IVA" 
            )
    for (col in colnames(info)) {
        preds[,col] = info[match(preds$name, rownames(info)),col]
    }
    preds$sig_mr = apply(as.matrix(preds[,grepl('mr_fold', colnames(preds))]), 1, mean)
    preds$sig_cli = apply(as.matrix(preds[,grepl('cli_fold', colnames(preds))]), 1, mean)
    preds$set = ifelse(preds$name %in% rownames(ZFinfo), 0, 1)

    k_fold = read.csv("/home/tongxueqing/zhao/ImageProcessing/combine_model/_data/k_fold_name.csv", row.names = 1)

    train_val = preds[preds$set == 0,]
    test = preds[preds$set == 1,]
}

# single significant clinic feature
if (T) {
    ps = sapply(colnames(info)[colnames(info) != "EVB"], function(col) {
        x = train[,col]
        time = train$time[!is.na(x)]
        event = train$event[!is.na(x)]
        x = x[!is.na(x)]
        hr = hazard.ratio(x = x, surv.time = time, surv.event = event)
        p = hr$p.value
    })
    signi.cli = names(ps)[ps < 0.05]
    remove.cli = colnames(info)[!colnames(info) %in% c(signi.cli, "EVB")]
    train[,remove.cli] = NULL
    test[,remove.cli] = NULL
}

to_ci = function(data, cox) {
    features = names(cox$coefficients)
    subdata = data[,features]
    index = complete.cases(subdata)
    time = data$time[index]
    event = data$event[index]
    subdata = subdata[index,]
    pred = predict(cox, newdata = subdata, type = "lp")
    ci = concordance.index(x = pred, surv.time = time, surv.event = event, method = "noether")   
    ci$c.index
}

make_sig = function(data, features) {
    name = paste(features, collapse = ".")
    subdata = data[,features]
    index = complete.cases(subdata)
    time = data$time[index]
    event = data$event[index]
    subdata = subdata[index,]
    cox = coxph(Surv(time, event) ~ ., data = subdata)
    pred = predict(cox, newdata = subdata, type = 'lp')
    ci = concordance.index(x = pred, surv.time = time, surv.event = event, method = "noether")
    ci = ci$c.index
    return(list("model" = cox, "pred" = pred, "citr" = ci))
}

cut_off = function() {
    
}

features.list = list(
    "deep_mr" = c("sig_deep", "sig_mr"),
    "deep_mr_cli" = c("sig_deep", "sig_mr", "sig_cli"),
    "deep_mr_sigcli" = c("sig_deep", "sig_mr", signi.cli)
)

meancis = sapply(features.list, function(features) {
    cis = sapply(1:4, function(k) {
        train = train_val[!train_val$name %in% k_fold[,k],]
        val = train_val[train_val$name %in% k_fold[,k],]
        res = make_sig(train, features)
        citr = res$ci
        cox = res$model
        cits = to_ci(val, cox)
        c(citr, cits)
    })
    rowMeans(cis)
})
