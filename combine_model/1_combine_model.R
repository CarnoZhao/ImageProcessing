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
    mrs = colnames(preds)[grepl("mr_serie", colnames(preds))]
    # preds$sig_mr = apply(as.matrix(preds[,grepl('mr_serie', colnames(preds))]), 1, mean)
    # preds$sig_cli = apply(as.matrix(preds[,grepl('cli_fold', colnames(preds))]), 1, mean)
    preds = preds[complete.cases(preds$sig_deep),]

    k_fold = read.csv("/home/tongxueqing/zhao/ImageProcessing/combine_model/_data/k_fold_name.csv", row.names = 1)

    train_val = preds[preds$set == 0,]
    test = preds[preds$set == 1,]
    
    data.dist = datadist(train_val[,!grepl("(EVB|set)", colnames(train_val))])
    options(datadist = 'data.dist')
}

# single significant clinic feature
if (T) {
    all.clis = colnames(info)[!colnames(info) %in% c("EVB", "name", "set", "time", "event")]
    ps = sapply(all.clis, function(col) {
        x = train_val[,col]
        time = train_val$time[!is.na(x)]
        event = train_val$event[!is.na(x)]
        x = x[!is.na(x)]
        hr = hazard.ratio(x = x, surv.time = time, surv.event = event)
        p = hr$p.value
    })
    clis = names(ps)[ps < 0.05]
    remove.cli = all.clis[!all.clis %in% clis]
    train_val[,remove.cli] = NULL
    test[,remove.cli] = NULL
}

if (T) {  
    features.list = list(
        "deep_mr" = c("sig_deep", "sig_mr"),
        "deep_mr_cli" = c("sig_deep", "sig_mr", clis),
        "deep_mr_sigcli" = c("sig_deep", "sig_mr", "sig_cli")
    )  
    train = train_val[!train_val$name %in% k_fold[,2],]
    val = train_val[train_val$name %in% k_fold[,2],]
    result = matrix(rep(0, length(features.list) * 3), c(length(features.list), 3))
    row.names(result) = names(features.list)
    colnames(result) = c("citr", "civl", "cits")
    for (name in names(features.list)) {
        features = features.list[[name]]
        res = make_sig(train, features)
        cox = res$model
        # ci
        citr = to_ci(train, cox)
        civl = to_ci(val, cox)
        cits = to_ci(test, cox)
        result[match(name, names(features.list)),] = c(citr, civl, cits)
        
        # add new signature
        train[,name] = to_pred(train, cox)$pred
        val[,name] = to_pred(val, cox)$pred
        test[,name] = to_pred(test, cox)$pred
    }
    result[order(rowMeans(result[,1:2])),]

    for (name in names(features.list)) {   
        pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/', name, '.pdf')), width = 16, height = 8) 
        par(mfrow = c(1, 2))

        # risk plot
        cutoff = cut_off(train, name)
        risk_plot(train, cutoff, name)
        risk_plot(rbind(val, test), cutoff, name)

        # survival ROC
        roc_plot(train, name)
        roc_plot(rbind(val, test), name)

        # nomogram
        nomo_plot(train, name, features.list)

        # calibration 
        calibration_plot(train, name)
        calibration_plot(rbind(val, test), name)

        dev.off()
    }
}

# EVB
if (F) {
    evb.features.list = lapply(names(features.list), function(name) {
        c("EVB", features.list[[name]])
    })
    names(evb.features.list) = paste0("evb_", names(features.list))
    evb.features.list = c(evb.features.list, features.list)
    EVB.median = median(as.numeric(ifelse(na.omit(train_val$EVB) == "<500", "0", na.omit(train_val$EVB))))
    evb.data = train_val[complete.cases(train_val$EVB),]
    evb.data$EVB = ifelse(evb.data$EVB == "<500", "0", evb.data$EVB)
    evb.data$EVB = ifelse(as.numeric(evb.data$EVB) < EVB.median, 0, 1)
    
    evb.result = matrix(rep(0, length(evb.features.list) * 2), c(length(evb.features.list), 2))
    row.names(evb.result) = names(evb.features.list)
    colnames(evb.result) = c("citr", "civl")
    for (name in names(evb.features.list)) {
        features = evb.features.list[[name]]
        models = lapply(1:4, function(k) {
            train = evb.data[!evb.data$name %in% k_fold[,k],]
            val = evb.data[evb.data$name %in% k_fold[,k],]
            res = make_sig(train, features)
            cox = res$model
        })
        # ci
        citr = mean(sapply(1:4, function(k) {
            tr = evb.data[!evb.data$name %in% k_fold[,k],]
            model = models[[k]]
            to_ci(tr, model)
        }))
        civl = mean(sapply(1:4, function(k) {
            vl = evb.data[evb.data$name %in% k_fold[,k],]
            model = models[[k]]
            to_ci(vl, model)
        }))
        evb.result[match(name, names(evb.features.list)),] = c(citr, civl)
        
        # add new signature
        evb.data[,name] = rowMeans(sapply(models, function(cox) to_pred(evb.data, cox)$pred))
    }
    evb.result
}


