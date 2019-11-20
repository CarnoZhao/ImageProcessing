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
    info$N.read = ifelse(info$N.read %in% c(0, 1), 0, 1)
    info$T.read = ifelse(info$T.read %in% c(1, 2), 0, 1)
    info$total.cut = ifelse(info$total.cut %in% c(1, 2, 3), 0, 1)
    # info$age = ifelse(info$age < 50, 0, 1)
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
    train = train_val[!train_val$name %in% k_fold[,2],]
    val = train_val[train_val$name %in% k_fold[,2],]
    val = rbind(test, val)
    options(datadist = 'data.dist')
}

# single significant clinic feature
if (T) {
    all.clis = colnames(info)[!colnames(info) %in% c("EVB", "name", "set", "time", "event")]
    ps = sapply(all.clis, function(col) {
        time = train$time
        event = train$event
        cox = coxph(Surv(time, event) ~ train[,col],)
        summary(cox)$coefficients[,"Pr(>|z|)"]
    })
    clis = names(ps)[ps < 0.05]
    clis = clis[clis != "T.read"]
    # cli.cox = coxph(as.formula(paste0("Surv(time, event) ~ ", paste(clis, collapse = " + "))), data = train)
    remove.cli = all.clis[!all.clis %in% clis]
    train[,remove.cli] = NULL
    val[,remove.cli] = NULL
}

if (T) {  
    features.list = list(
        "deep" = "sig_deep",
        "mr" = "sig_mr",
        "cli" = clis,
        # "sigcli" = "sig_cli",
        "deep_mr" = c("sig_deep", "sig_mr"),
        "deep_mr_cli" = c("sig_deep", "sig_mr", clis)
        # "deep_mr_sigcli" = c("sig_deep", "sig_mr", "sig_cli")
    )
    result = matrix(rep(0, length(features.list) * 2), c(length(features.list), 2))
    row.names(result) = names(features.list)
    colnames(result) = c("citr", "civl")
    models = list()
    for (name in names(features.list)) {
        features = features.list[[name]]
        res = make_sig(train, features)
        cox = res$model
        ci = res$ci
        models[[length(models) + 1]] = cox

        # ci
        citr = to_ci(train, cox)
        civl = to_ci(val, cox)
        result[match(name, names(features.list)),] = c(citr, civl)
        
        # add new signature
        train[,name] = to_pred(train, cox)$pred
        val[,name] = to_pred(val, cox)$pred
    }
    names(models) = names(features.list)
    result[order(rowMeans(result[,1:2])),]

    # c-index confidence interval and p-value
    if (T) {        
        summary.cis = lapply(names(features.list), function(name) {
            cox = models[[match(name, names(features.list))]]
            d = list('train' = train, 'val' = val)
            summs = sapply(d, function(data) {
                ci = to_raw_ci(data, cox)
                sapply(ci[c('c.index', "lower", 'upper', 'p.value')], signif, digits = 3)
            })
            summs = as.data.frame(summs)
            summs$X = rownames(summs)
            summs$name = name
            summs = summs[,c(4, 3, 1, 2)]
        })
        names(summary.cis) = names(features.list)
        summary.cis = do.call(rbind, summary.cis)
        rownames(summary.cis) = NULL
        write.csv(summary.cis, "/home/tongxueqing/zhao/ImageProcessing/combine_model/_outs/summary.cis.csv")
    }

    # model compare: cindex.comp
    if (T) {    
        tmp = train
        comp = sapply(models, function(cox1) {
            sapply(models, function(cox2) {
                cindex.comp(to_raw_ci(tmp, cox2), to_raw_ci(tmp, cox1))$p.value
            })
        })
        # comp = ifelse(comp < 0.05, comp, NA)
        train.sig = signif(comp, 3)
        tmp = val
        comp = sapply(models, function(cox1) {
            sapply(models, function(cox2) {
                cindex.comp(to_raw_ci(tmp, cox2), to_raw_ci(tmp, cox1))$p.value
            })
        })
        # comp = ifelse(comp < 0.05, comp, NA)
        val.sig = signif(comp, 3)
    }


    # plots
    dd.features = colnames(train)[!grepl("(EVB|set)", colnames(train))]
    for (name in names(features.list)) {   
        pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/', name, '.pdf')), width = 16, height = 8) 
        par(mfrow = c(1, 2))

        # risk plot
        cutoff = cut_off(train, name)
        risk_plot(train, cutoff, name)
        risk_plot(val, cutoff, name)

        cutby = "gender"; cutby.names = c("male", "female")
        risk_plot_strat(train, cutoff, name, info, cutby, cutby.names)
        risk_plot_strat(val, cutoff, name, info, cutby, cutby.names)

        # survival ROC
        roc_plot(train, name)
        roc_plot(val, name)

        # nomogram
        data.dist = datadist(train[,dd.features])
        nomo_plot(train, name, features.list)

        # calibration 
        opar = par(no.readonly = T)
        par(mfrow = c(1, 2), lwd = 2, pch = 20)
        data.dist = datadist(train[,dd.features])
        calibration_plot(train, name, features.list, npoints = 3)
        data.dist = datadist(val[,dd.features])
        calibration_plot(val, name, features.list, npoints = 3)
        par(opar)

        dev.off()
    }
}

# EVB
if (T) {
    # EVB preload
    if (T) {    
        evb.features.list = lapply(names(features.list), function(name) {
            c("EVB", features.list[[name]])
        })
        names(evb.features.list) = paste0("evb_", names(features.list))
        evb.features.list = c(evb.features.list, features.list, "evb" = "EVB")
        EVB.median = mean(as.numeric(ifelse(na.omit(train_val$EVB) == "<500", "0", na.omit(train_val$EVB))))
        evb.data = train_val[complete.cases(train_val$EVB),]
        evb.data$EVB = ifelse(evb.data$EVB == "<500", "0", evb.data$EVB)
        evb.data$EVB = ifelse(as.numeric(evb.data$EVB) < EVB.median, 0, 1)
        
        evb.result = matrix(rep(0, length(evb.features.list) * 2), c(length(evb.features.list), 2))
        row.names(evb.result) = names(evb.features.list)
        colnames(evb.result) = c("citr", "civl")
        evb.models = list()
        evb.fold = t(matrix(c(sample(evb.data$name), rep(NA, 2)), c(3, 33)))
    }

    for (name in names(evb.features.list)) {
        features = evb.features.list[[name]]
        # evb.models = lapply(1:3, function(k) {
        #     train = evb.data[!evb.data$name %in% evb.fold[,k],]
        #     val = evb.data[evb.data$name %in% evb.fold[,k],]
        #     res = make_sig(train, features)
        #     cox = res$model
        # })
        # k_fold = as.matrix(k_fold)
        res = make_sig(evb.data, features)
        cox = res$model
        evb.models[[length(evb.models) + 1]] = cox


        # ci
        # citr = mean(sapply(1:3, function(k) {
        #     tr = evb.data[!evb.data$name %in% evb.fold[,k],]
        #     model = evb.models[[k]]
        #     to_ci(tr, model)
        # }))
        # civl = mean(na.omit(sapply(1:3, function(k) {
        #     vl = evb.data[evb.data$name %in% evb.fold[,k],]
        #     model = evb.models[[k]]
        #     to_ci(vl, model)
        # })))
        citr = to_ci(evb.data, cox)
        evb.result[match(name, names(evb.features.list)),] = c(citr, NA)
        
        # add new signature
        # evb.data[,name] = rowMeans(sapply(models, function(cox) to_pred(evb.data, cox)$pred))
    }
    names(evb.models) = names(evb.features.list)
    evb.result[order(evb.result[,1]),]

    # c-index comparison
    if (T) {        
        tmp = evb.data
        names(evb.models) = names(evb.features.list)
        comp = sapply(names(features.list), function(name) {
            cox2 = evb.models[[paste0("evb_", name)]]
            cox1 = evb.models[[name]]
            cindex.comp(to_raw_ci(tmp, cox2), to_raw_ci(tmp, cox1))$p.value
        })
        signif(comp, 3)
    }
}


