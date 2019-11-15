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
        set.seed(0)
    }
    source(file.path(root, "ImageProcessing/combine_model/functions.R"))

    preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
    info = read.csv(file.path(root, "ImageProcessing/combine_model/_data/clinicinfo.csv"), stringsAsFactors = F, row.names = 1)
    info[,c("N.cut.N3b", "total.cut.IVA")] = NULL
    for (col in colnames(info)) {
        preds[,col] = info[match(preds$name, rownames(info)),col]
    }
    preds$sig_mr = apply(as.matrix(preds[,grepl('mr_fold', colnames(preds))]), 1, mean)
    preds$sig_cli = apply(as.matrix(preds[,grepl('cli_fold', colnames(preds))]), 1, mean)
    preds = preds[complete.cases(preds$sig_deep),]

    k_fold = read.csv("/home/tongxueqing/zhao/ImageProcessing/combine_model/_data/k_fold_name.csv", row.names = 1)

    train_val = preds[preds$set == 0,]
    test = preds[preds$set == 1,]
    
    data.dist = datadist(train_val[,!grepl("(EVB|set)", colnames(train_val))])
    options(datadist = 'data.dist')
}

# single significant clinic feature
if (T) {
    clis = colnames(info)[!colnames(info) %in% c("EVB", "name", "set", "time", "event")]
    ps = sapply(clis, function(col) {
        x = train_val[,col]
        time = train_val$time[!is.na(x)]
        event = train_val$event[!is.na(x)]
        x = x[!is.na(x)]
        hr = hazard.ratio(x = x, surv.time = time, surv.event = event)
        p = hr$p.value
    })
    signi.cli = names(ps)[ps < 0.05]
    remove.cli = clis[!clis %in% signi.cli]
    train_val[,remove.cli] = NULL
    test[,remove.cli] = NULL
}

if (T) {  
    features.list = list(
        "sigcli" = signi.cli,
        "deep_mr" = c("sig_deep", "sig_mr"),
        "deep_cli" = c("sig_deep", "sig_cli"),
        "deep_sigcli" = c("sig_deep", signi.cli),
        "mr_cli" = c("sig_mr", "sig_cli"),
        "mr_sigcli" = c("sig_mr", signi.cli),
        "deep_mr_cli" = c("sig_deep", "sig_mr", "sig_cli"),
        "deep_mr_sigcli" = c(signi.cli, "sig_deep", "sig_mr")
    )  
    result = matrix(rep(0, length(features.list) * 2), c(length(features.list), 2))
    row.names(result) = names(features.list)
    colnames(result) = c("citr", "cits")
    for (name in names(features.list)) {
        features = features.list[[name]]
        models = lapply(1:4, function(k) {
            train = train_val[!train_val$name %in% k_fold[,k],]
            val = train_val[train_val$name %in% k_fold[,k],]
            res = make_sig(train, features)
            cox = res$model
        })
        # ci
        citr = mean(sapply(models, to_ci, data = train_val))
        cits = mean(sapply(models, to_ci, data = test))
        result[match(name, names(features.list)),] = c(citr, cits)
        
        # add new signature
        train_val[,name] = rowMeans(sapply(models, function(cox) to_pred(train_val, cox)$pred))
        test[,name] = rowMeans(sapply(models, function(cox) to_pred(test, cox)$pred))
    }
    result

    for (name in names(features.list)) {   
        pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/', name, '.pdf')), width = 16, height = 8) 
        par(mfrow = c(1, 2))

        # risk plot
        cutoff = cut_off(train_val, name)
        risk_plot(train_val, cutoff, name)
        risk_plot(test, cutoff, name)

        # survival ROC
        roc_plot(train_val, name)
        roc_plot(test, name)

        # nomogram
        nomo_plot(train_val, name, features.list)

        # calibration 
        calibration_plot(train_val, name)
        calibration_plot(test, name)

        dev.off()
    }
}

if (T) {
    evb.features.list = list(
        "evb_deep" = c("sig_deep", "EVB"),
        "evb_mr" = c("sig_mr", "EVB"),
        "evb_cli" = c("sig_cli", "EVB"),
        "evb_sigcli" = c(signi.cli, "EVB"),
        "evb_deep_mr" = c("sig_deep", "sig_mr", "EVB"),
        "evb_deep_cli" = c("sig_deep", "sig_cli", "EVB"),
        "evb_deep_sigcli" = c("sig_deep", signi.cli, "EVB"),
        "evb_mr_cli" = c("sig_mr", "sig_cli", "EVB"),
        "evb_mr_sigcli" = c("sig_mr", signi.cli, "EVB"),
        "evb_deep_mr_cli" = c("sig_deep", "sig_mr", "sig_cli", "EVB"),
        "evb_deep_mr_sigcli" = c(signi.cli, "sig_deep", "sig_mr", "EVB")
    )
    evb.features.list = c(evb.features.list, features.list)
    EVB.median = median(as.numeric(ifelse(na.omit(train_val$EVB) == "<500", "0", na.omit(train_val$EVB))))
    sub.train_val = train_val[complete.cases(train_val$EVB),]
    sub.train_val$EVB = ifelse(sub.train_val$EVB == "<500", "0", sub.train_val$EVB)
    sub.train_val$EVB = ifelse(as.numeric(sub.train_val$EVB) < EVB.median, 0, 1)
    
    evb.result = matrix(rep(0, length(evb.features.list) * 2), c(length(evb.features.list), 2))
    row.names(evb.result) = names(evb.features.list)
    colnames(evb.result) = c("citr", "civl")
    for (name in names(evb.features.list)) {
        features = evb.features.list[[name]]
        models = lapply(1:4, function(k) {
            train = sub.train_val[!sub.train_val$name %in% k_fold[,k],]
            val = sub.train_val[sub.train_val$name %in% k_fold[,k],]
            res = make_sig(train, features)
            cox = res$model
        })
        # ci
        citr = mean(sapply(1:4, function(k) {
            tr = sub.train_val[!sub.train_val$name %in% k_fold[,k],]
            model = models[[k]]
            to_ci(tr, model)
        }))
        civl = mean(sapply(1:4, function(k) {
            vl = sub.train_val[sub.train_val$name %in% k_fold[,k],]
            model = models[[k]]
            to_ci(vl, model)
        }))
        evb.result[match(name, names(evb.features.list)),] = c(citr, civl)
        
        # add new signature
        sub.train_val[,name] = rowMeans(sapply(models, function(cox) to_pred(sub.train_val, cox)$pred))
    }
}

round(result, 3)
round(evb.result, 3)[order(rowMeans(evb.result)),]
signi.cli
