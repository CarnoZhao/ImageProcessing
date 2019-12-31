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
            root = "/wangshuo/zhaox/"
        } else {
            root = "/home/tongxueqing/zhao/"
        }
        set.seed(0)
    }
    source(file.path(root, "ImageProcessing/combine_model/functions.R"))

    preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
    info = read.csv(file.path(root, "ImageProcessing/combine_model/_data/ClinicMessageForAnalysis.csv"), stringsAsFactors = F, row.names = 1)

    info$age = ifelse(info$age < 52, 0, 1)
    for (col in colnames(info)) {
        preds[,col] = info[match(preds$name, as.numeric(rownames(info))),col]
    }
    mrs = colnames(preds)[grepl("mr_serie", colnames(preds))]
    preds = preds[complete.cases(preds$sig_deep),]

    k_fold = read.csv(paste0(root, "ImageProcessing/combine_model/_data/k_fold_name.csv"), row.names = 1)

    train_val = preds[preds$set == 0,]
    test = preds[preds$set == 1,]
    train = train_val[!train_val$name %in% k_fold[,2],]
    val = train_val[train_val$name %in% k_fold[,2],]
    # val = rbind(test, val)
    options(datadist = 'data.dist')
}

# single significant clinic feature
if (F) {
    all.clis = colnames(info)[!colnames(info) %in% c("EVB", "name", "set", "time", "event")]
    ps = sapply(all.clis, function(col) {
        subtrain = train[complete.cases(train[,col]),]
        time = subtrain$time
        event = subtrain$event
        cox = coxph(Surv(time, event) ~ subtrain[,col],)
        summary(cox)$coefficients[,"Pr(>|z|)"]
        hazard.ratio(subtrain[,col], surv.time = time, surv.event = event)[1:6]
    })
    clis = names(ps)[ps < 0.05]
    clis = clis[clis != "T.read"]
    # cli.cox = coxph(as.formula(paste0("Surv(time, event) ~ ", paste(clis, collapse = " + "))), data = train)
    remove.cli = all.clis[!all.clis %in% clis]
    train[,remove.cli] = NULL
    val[,remove.cli] = NULL
} else {
    clis = c("age", "N.read", "lymphocyte")
}

if (T) {  
    features.list = list(
        "deep" = "sig_deep",
        "mr" = "sig_mr",
        "cli" = clis,
        "deep_mr" = c("sig_deep", "sig_mr"),
        "deep_mr_cli" = c("sig_deep", "sig_mr", clis),
        "doctor" = c("necrosis", "lymphocyte")
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
        citr = to_ci(train, cox)
        civl = to_ci(val, cox)
        result[match(name, names(features.list)),] = c(citr, civl)
        train[,name] = to_pred(train, cox)$pred
        val[,name] = to_pred(val, cox)$pred
        test[,name] = to_pred(test, cox)$pred
    }
    names(models) = names(features.list)
    result[order(rowMeans(result[,1:2])),]

    # c-index confidence interval and p-value
    if (T) {        
        summary.cis = lapply(names(features.list), function(name) {
            cox = models[[match(name, names(features.list))]]
            d = list('train' = train, 'val' = val, "test" = test)
            summs = sapply(d, function(data) {
                ci = to_raw_ci(data, cox)
                x = sapply(ci[c('c.index', "lower", 'upper', 'p.value')], signif, digits = 3)
                paste(x[1], " (", x[2], "-", x[3], "),", x[4], sep = "")
            })
            paste(summs, collapse = ",")
            # summs = as.data.frame(summs)
            # summs$X = rownames(summs)
            # summs$name = name
            # summs = summs[,c(3, 2, 1)]
        })
        names(summary.cis) = names(features.list)
        summary.cis = do.call(rbind, summary.cis)
        rownames(summary.cis) = NULL
        write.csv(summary.cis, paste0(root, "ImageProcessing/combine_model/_outs/summary.cis.csv"), quote = F)
    }

    # model compare: cindex.comp
    # if (T) {    
    #     tmp = train
    #     comp = sapply(models, function(cox1) {
    #         sapply(models, function(cox2) {
    #             cindex.comp(to_raw_ci(tmp, cox2), to_raw_ci(tmp, cox1))$p.value
    #         })
    #     })
    #     # comp = ifelse(comp < 0.05, comp, NA)
    #     train.sig = signif(comp, 3)
    #     tmp = val
    #     comp = sapply(models, function(cox1) {
    #         sapply(models, function(cox2) {
    #             cindex.comp(to_raw_ci(tmp, cox2), to_raw_ci(tmp, cox1))$p.value
    #         })
    #     })
    #     # comp = ifelse(comp < 0.05, comp, NA)
    #     val.sig = signif(comp, 3)
    # }


    # plots
    dd.features = do.call(c, features.list)
    name = "deep_mr_cli"
    pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/risk.pdf')), width = 24, height = 8) 
    par(mfrow = c(1, 3))
    cutoff = cut_off(train, name)
    risk_plot(train, cutoff, name)
    risk_plot(val, cutoff, name)
    risk_plot(test, cutoff, name)
    dev.off()

    pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/strat.pdf')), width = 16, height = 30) 
    par(mfrow = c(6, 2))
    all = rbind(train, val, test)
    risk_plot(all[all$gender == 1,], cutoff, name, main = "Male")
    risk_plot(all[all$gender == 2,], cutoff, name, main = "Female")
    risk_plot(all[all$T.read == 0,], cutoff, name, main = "T-stage: I-II")
    risk_plot(all[all$T.read == 1,], cutoff, name, main = "T-stage: III-IV")
    risk_plot(all[all$smoke == 0,], cutoff, name, main = "no smoke")
    risk_plot(all[all$smoke == 1,], cutoff, name, main = "smoke")
    risk_plot(all[all$HB == 0,], cutoff, name, main = "HB low")
    risk_plot(all[all$HB == 1,], cutoff, name, main = "HB high")
    risk_plot(all[all$total.cut == 0,], cutoff, name, main = "total.cut I-III")
    risk_plot(all[all$total.cut == 1,], cutoff, name, main = "total.cut IV")
    risk_plot(all[all$family == 0,], cutoff, name, main = "no family")
    risk_plot(all[all$family == 1,], cutoff, name, main = "family")
    # risk_plot(all[all$smoke == 0,], cutoff, name, main = "Smoke")
    # risk_plot(all[all$smoke == 1,], cutoff, name, main = "No smoke")
    dev.off()

    pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/second.pdf')), width = 24, height = 8) 
    par(mfrow = c(1, 3))
    all = rbind(train, val, test)
    risk_plot(all, cutoff, name, x = "death", main = "All data Death")
    risk_plot(all, cutoff, name, x = "trans", main = "All data Trans")
    risk_plot(all, cutoff, name, x = "re", main = "All data Re")
    dev.off()

    pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/ROC.pdf')), width = 24, height = 8)
    par(mfrow = c(1, 3))
    roc_plot(train, name)
    roc_plot(val, name)
    roc_plot(test, name)
    dev.off()

        # nomogram
    pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/nomo.pdf')), width = 16, height = 8)
    data.dist = datadist(train[,dd.features])
    nomo_plot(train, name, features.list)
    dev.off()

        # calibration 
    pdf(file.path(root, paste0('ImageProcessing/combine_model/_plots/calibration.pdf')), width = 24, height = 8)
    opar = par(no.readonly = T)
    par(mfrow = c(1, 3), lwd = 2, pch = 20)
    data.dist = datadist(train[,dd.features])
    calibration_plot(train, name, features.list, npoints = 3)
    data.dist = datadist(val[,dd.features])
    calibration_plot(val, name, features.list, npoints = 3)
    data.dist = datadist(test[,dd.features])
    calibration_plot(test, name, features.list, npoints = 3)
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


