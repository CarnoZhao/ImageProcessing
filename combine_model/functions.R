suppressPackageStartupMessages({
    library(survivalROC)
    library(lattice)
    library(Formula)
    library(ggplot2)
    library(Hmisc)
    library(rms)
    library(survminer)
})
CUT1 = 36

to_pred = function(data, cox) {
    features = names(cox$coefficients)
    time = data$time
    event = data$event
    pred = predict(cox, newdata = data, type = "lp")
    data.frame(pred, time, event)
}

to_raw_ci = function(data, cox) {
    df = to_pred(data, cox)
    ci = concordance.index(x = df$pred, surv.time = df$time, surv.event = df$event, method = "noether")
    ci
}

to_ci = function(data, cox) {
    ci = to_raw_ci(data, cox)  
    ci$c.index
}

make_sig = function(data, features) {
    if (length(features) == 1) {
        subdata = data.frame(data[,features])
        colnames(subdata) = features
    } else {
        subdata = data[,features]
    }
    time = data$time
    event = data$event
    cox = coxph(Surv(time, event) ~ ., data = subdata)
    pred = predict(cox, newdata = subdata, type = 'lp')
    ci = concordance.index(x = pred, surv.time = time, surv.event = event, method = "noether")
    cix = ci$c.index
    return(list("model" = cox, "pred" = pred, "citr" = cix, "ci" = ci))
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

risk_plot = function(data, cutoff, name, x = NULL, main = NULL) {
    if (all(data$set == 0) & all(!is.na(data$set))) {set = 'train'}
    else {set = 'test'}
    if (is.null(main)) {
        set = set
    } else {
        set = main
    }
    if (is.null(x)) {
        timename = "time"
        eventname = "event"
    } else {
        timename = paste(x, "time", sep = ".")
        eventname = paste(x, "event", sep = ".")
    }
    stra.df = data.frame("time" = data[,timename], "event" = data[,eventname], "stra" = ifelse(data[,name] < cutoff, 0, 1))
    km.coxph.plot(
        formula.s = Surv(time, event) ~ stra, 
        data.s = stra.df, leg.inset = 0.02,
        .lwd = c(2,2), .col = c('black','red'),
        x.label = "Time (months)",
        y.label = "Probability of Survival",
        leg.text = paste(c("Low risk", "High risk"), " ", sep = ""),
        leg.pos = "bottomright",
        show.n.risk = TRUE,
        n.risk.step = 12, n.risk.cex = 1, 
        mark.time = T,  v.line = c(CUT1),
        main.title = set, verbose = F)
}

if (F) {
    data = train
    data = val
    data = test
    data$stra = ifelse(data$deep_mr_cli < cutoff, 0, 1)
    hazard.ratio(data$deep_mr_cli, surv.time = data$time, surv.event = data$event, strat = data$stra)
}

# risk_plot_strat = function(data, cutoff, name, info, cutby, cutby.names) {
#     stra.df = data.frame("time" = data$time, "event" = data$event, "stra" = ifelse(data[,name] < cutoff, 0, 1))
#     stra.df$cutby = info[match(data$name, info$name), cutby]
#     stra.df$stra.cut = paste(stra.df$stra, stra.df$cutby, sep = ".")
#     ps = sapply(unique(stra.df$stra), function(s) {
#         tmp = stra.df[stra.df$stra == s,]
#         p = summary(coxph(Surv(time, event) ~ cutby, data = tmp))$sctest["pvalue"]
#         signif(p, 3)
#     })
#     names(ps) = unique(stra.df$stra)
#     if (all(data$set == 0)) {
#         set = 'train'
#     } else {set = 'test'}
#     km.coxph.plot(
#         formula.s = Surv(time, event) ~ stra.cut, 
#         data.s = stra.df, leg.inset = 0.02,
#         .lwd = c(2,2), .col = rep(c('black','red'), each = 2), .lty = rep(c(1, 2), 2),
#         x.label = "Time (months)",
#         y.label = "Probability of Survival",
#         leg.text = paste(
#             rep(c("Low", "High"), each = 2), 
#             "risk",
#             rep(cutby.names, 2),
#             rep(c("", " P =")),
#             c("", ps[1], "", ps[2]), sep = " "),
#         leg.pos = "bottomleft", leg.bty = "n",
#         show.n.risk = F,
#         o.text = "",
#         mark.time = T,  v.line = c(CUT1),
#         main.title = set, verbose = F)
# }

# png("/home/tongxueqing/zhao/ImageProcessing/combine_model/_plots/test.png")
# risk_plot_strat(rbind(val, test), cutoff, name, info, cutby, cutby.names)
# dev.off()

rowMax = function(df) {
    apply(as.matrix(df), 1, max)
}

roc_plot = function(data, name, main = NULL) {
    names = c("cli", "mr", "deep", "deep_mr_cli")
    if (all(data$set == 0)) {set = 'train'}
    else {set = 'test'}
    ts = lapply(names, function(name) {
        n = length(data[,name])
        t = survivalROC(
            Stime = data$time, 
            status = data$event, 
            data[,name], 
            predict.time = CUT1, 
            span = 0.001 * n ^ (-0.2))
    })
    t1 = ts[[1]]
    plot(
        t1$FP, 
        t1$TP, 
        type = "l",
        xlim = c(0, 1),
        ylim = c(0, 1),
        xlab = "False Positive Rate (%)",
        ylab = "True Positive Rate (%)",
        col = "blue",
        lwd = 2,
        cex.lab = 1.5,
        main = set)
    legend(
        "bottomright",
        legend = paste0(names, " ", CUT1 / 12, '-year: AUC = ', round(sapply(ts, function(t)t$AUC), 3)),
        col = c("blue", "black", "green", "red"),
        lwd = 2,
        cex = 1.5,
        lty = c(1, 1))
    lines(c(0, 1), c(0, 1), lty = 6, col = rgb(113 / 255, 150 / 255, 159 / 255), lwd = 2.0)
    lines(ts[[2]]$FP, ts[[2]]$TP, lty = 1, lwd = 2, col = "black")    
    lines(ts[[3]]$FP, ts[[3]]$TP, lty = 1, lwd = 2, col = "green")    
    lines(ts[[4]]$FP, ts[[4]]$TP, lty = 1, lwd = 2, col = "red")    
}

nomo_plot = function(data, name, features.list) {
    opar = par(no.readonly = T)
    par(mfrow = c(1, 1))
    features = features.list[[name]]
    sink("/dev/null")
    f = cph(
        as.formula(
            paste0("Surv(time, event) ~ ", 
            paste(features, collapse = " + "))),
        surv = T, x = T, y = T, data = data)
    sink()
    surv.prob = Survival(f)
    nom = nomogram(
        f, 
        fun = list(function(x) surv.prob(CUT1, x)),
        funlabel = c(paste0(CUT1 / 12, "-year DFS rate")),
        fun.at = 10:0 / 10,
        lp = F)
    plot(nom, xfrac = 0.3, cex.axis = 1.3, cex.var = 1.3)
    par(opar)
}

HLtest = function(cal){
    len = cal[,'n']
    meanp = cal[,'mean.predicted']
    sump = meanp * len
    sumy = cal[,'KM'] * len
    contr = ((sumy - sump) ^ 2) / (len * meanp * (1 - meanp))
    chisqr = sum(contr)
    pval = 1 - pchisq(chisqr, length(len) - 2)
    return(pval)
}

calibration_plot = function(data, name, features.list, npoints = 3) {
    if (all(data$set == 0)) {set = 'train'}
    else {set = 'test'}
    features = features.list[[name]]
    pos = 0
    sink("/dev/null")
    f3 = cph(
        as.formula(
            paste0("Surv(time, event) ~ ", #name)),
             paste(features, collapse = " + "))),
        surv = T, x = T, y = T, data = data, time.inc = CUT1)
    cal3 = calibrate(
        f3, u = CUT1, cmethod = "KM", method = "boot",
        B = 30, m = floor(nrow(data) / npoints), surv = T)
    p3 = round(HLtest(cal3), 3)
    color3 = c(rgb(0, 112, 255, maxColorValue = 255))
    plot(
        cal3, lty = 1, lwd = 2,
        errbar.col = color3, 
        par.corrected = list(col = color3), 
        conf.int = T, 
        xlim = c(pos, 1), ylim = c(pos, 1), 
        riskdist = F, col = "red", axes = T,
        xlab="Nomogram-Predicted Probability DFS",
        ylab="Observed Actual DFS (Proportion)", main = set)

    legend(
        "bottomright", 
        legend = c(paste0(CUT1 / 12, '-year DFS: p = ', p3)), 
        col = c('red'), 
        lwd = 2, cex = 1.5, lty = c(1, 1))
    sink()
}