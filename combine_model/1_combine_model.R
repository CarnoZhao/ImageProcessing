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

    preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
    info = read.csv(file.path(root, "ImageProcessing/combine_model/_data/clinicinfo.csv"), stringsAsFactors = F, row.names = 1)
    for (col in colnames(info)) {
        preds[,col] = info[match(preds$name, rownames(info)),col]
    }
    preds$sig_mr = apply(as.matrix(preds[,grepl('mr_fold', colnames(preds))]), 1, mean)
    preds$sig_cli = apply(as.matrix(preds[,grepl('cli_fold', colnames(preds))]), 1, mean)
    preds = preds[complete.cases(preds$sig_deep),]

    k_fold = read.csv("/home/tongxueqing/zhao/ImageProcessing/combine_model/_data/k_fold_name.csv", row.names = 1)

    train_val = preds[preds$set == 0,]
    test = preds[preds$set == 1,]
}

# single significant clinic feature
if (T) {
    ps = sapply(colnames(info)[colnames(info) != "EVB"], function(col) {
        x = train_val[,col]
        time = train_val$time[!is.na(x)]
        event = train_val$event[!is.na(x)]
        x = x[!is.na(x)]
        hr = hazard.ratio(x = x, surv.time = time, surv.event = event)
        p = hr$p.value
    })
    signi.cli = names(ps)[ps < 0.05]
    remove.cli = colnames(info)[!colnames(info) %in% c(signi.cli, "EVB")]
    train_val[,remove.cli] = NULL
    test[,remove.cli] = NULL
}

to_pred = function(data, cox) {
    features = names(cox$coefficients)
    subdata = data[,features]
    time = data$time
    event = data$event
    pred = predict(cox, newdata = subdata, type = "lp")
    data.frame(pred, time, event)
}

to_ci = function(data, cox) {
    df = to_pred(data, cox)
    ci = concordance.index(x = df$pred, surv.time = df$time, surv.event = df$event, method = "noether")   
    ci$c.index
}

make_sig = function(data, features) {
    subdata = data[,features]
    time = data$time
    event = data$event
    cox = coxph(Surv(time, event) ~ ., data = subdata)
    pred = predict(cox, newdata = subdata, type = 'lp')
    ci = concordance.index(x = pred, surv.time = time, surv.event = event, method = "noether")
    ci = ci$c.index
    return(list("model" = cox, "pred" = pred, "citr" = ci))
}

cut_off = function(data, name) {
    z1 = data[,name]
    b2 = data.frame(z1)
    b2$t2 = data$time
    b2$d3 = data$event
    cox.2 = coxph(Surv(t2, d3) ~ z1, data = b2)
    cox.2 = cutp(cox.2)$z1
    round(cox.2$z1[1], digits = 3)
}

risk_plot = function(data, cutoff, name) {
    stra.df = data.frame("time" = data$time, "event" = data$event, "stra" = ifelse(data[,name] < cutoff, 0, 1))
    set = ifelse(all(data$set == 0), 'train', 'test')
    png(file.path(root, paste0('ImageProcessing/combine_model/_plots/', name, '.', set, '.risk.png')))
    sink("/dev/null")
    km.coxph.plot(
        formula.s = Surv(time, event) ~ stra, 
        data.s = stra.df, 
        leg.inset = 0.02,
        .lwd = c(2,2),
        x.label = "Time (months)",
        y.label = "Probability of Survival",
        main = "",
        .col = c('black','red'),
        leg.text = paste(c("Low risk", "High risk"), " ", sep = ""),
        leg.pos = "bottomright",
        show.n.risk = TRUE,
        n.risk.step = 12,
        n.risk.cex = 1, 
        mark.time = T, 
        v.line = c(38,60)
        )
    sink()
    dev.off()
}

rowMax = function(df) {
    apply(as.matrix(df), 1, max)
}

features.list = list(
    "deep_mr" = c("sig_deep", "sig_mr"),
    "deep_mr_cli" = c("sig_deep", "sig_mr", "sig_cli"),
    "deep_mr_sigcli" = c(signi.cli, "sig_deep", "sig_mr")
)

for (name in names(features.list)[1:1]) {
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
    
    # add new signature
    train_val[,name] = rowMeans(sapply(models, function(cox) to_pred(train_val, cox)$pred))
    test[,name] = rowMeans(sapply(models, function(cox) to_pred(test, cox)$pred))
    
    # risk plot
    cutoff = cut_off(train_val, name)
    risk_plot(train_val, cutoff, name)
    risk_plot(test, cutoff, name)
}
