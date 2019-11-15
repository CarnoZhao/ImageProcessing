suppressPackageStartupMessages({
    library(survivalROC)
    library(lattice)
    library(Formula)
    library(ggplot2)
    library(Hmisc)
    library(rms)
})
CUT1 = 30
CUT2 = 60

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
    if (length(features) == 1) {
        subdata = data.frame(a = data[,features])
    } else {
        subdata = data[,features]
    }
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
    if (all(data$set == 0)) {set = 'train'}
    else {set = 'test'}
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
        v.line = c(CUT1, CUT2),
        main.title = set)
    sink()
}

rowMax = function(df) {
    apply(as.matrix(df), 1, max)
}

roc_plot = function(data, name) {
    if (all(data$set == 0)) {set = 'train'}
    else {set = 'test'}
    n = length(data[,name])
    t1 = survivalROC(
        Stime = data$time, 
        status = data$event, 
        data[,name], 
        predict.time = CUT1, 
        span = 0.001 * n ^ (-0.2))
    t2 = survivalROC(
        Stime = data$time, 
        status = data$event, 
        data[,name], 
        predict.time = CUT2, 
        span = 0.001 * n ^ (-0.2))
    plot(
        t1$FP, 
        t2$TP, 
        type = "l",
        xlim = c(0, 1),
        ylim = c(0, 1),
        xlab = "False Positive Rate (%)",
        ylab = "True Positive Rate (%)",
        col = rgb(254 / 255, 67 / 255, 101 / 255),
        lwd = 2,
        cex.lab = 1.5,
        main = set)
    legend(
        "bottomright",
        legend = c(
            paste0(CUT1 / 12, '-year: AUC = ', round(t1$AUC, 3)), 
            paste0(CUT2 / 12, '-year: AUC = ', round(t2$AUC, 3))),
        col = c(
            rgb(254 / 255, 67 / 255, 101 / 255),
            rgb(0, 0, 0)),
        lwd = 2,
        cex = 1.5,
        lty = c(1, 1))
    lines(c(0, 1), c(0, 1), lty = 6, col = rgb(113 / 255, 150 / 255, 159 / 255), lwd = 2.0)
    lines(t2$FP, t2$TP, lty = 1, lwd = 2, col = rgb(0, 0, 0))    
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
        fun = list(
            function(x) surv.prob(CUT1, x),
            function(x) surv.prob(CUT2, x)),
        funlabel = c(paste0(CUT1 / 12, "-year DFS rate"), paste0(CUT2 / 12, "-year DFS rate")),
        fun.at = 10:0 / 10,
        lp = F)
    plot(nom, xfrac = 0.3, cex.axis = 1.3, cex.var = 1.3)
    par(opar)
}

HLtest = function(cal){
    len = cal[,'n']
    meanp = cal[,'mean.predicted']
    sump = meanp * len
    sumy = cal[, 'KM'] * len
    contr = ((sumy - sump) ^ 2) / (len * meanp * (1 - meanp))
    chisqr = sum(contr)
    pval = 1 - pchisq(chisqr, length(len) - 2)
    return(pval)
}

calibration_plot = function(data, name, features.list) {
    pos = 0
    sink("/dev/null")
    f3 = cph(
        as.formula(
            paste0("Surv(time, event) ~ ", #name)),
             paste(features, collapse = " + "))),
        surv = T, x = T, y = T, data = data, time.inc = CUT1)
    cal3 = calibrate(
        f3, u = CUT1, cmethod = "KM", method = "boot", 
        m = floor(nrow(data) / 3), surv = T)
    p3 = round(HLtest(cal3), 3)
    x1 = c(rgb(0, 112, 255, maxColorValue = 255))
    plot(
        cal3, lty = 1, pch = 16, 
        errbar.col = x1, 
        par.corrected = list(col = x1), 
        conf.int = T, lwd = 1.2, 
        xlim = c(pos, 1), ylim = c(pos, 1), 
        riskdist = F, col = "blue", axes = T,
        xlab="Nomogram-Predicted Probability DFS",
        ylab="Observed Actual DFS (Proportion)")

    try({
        f5 = cph(
            as.formula(
            paste0("Surv(time, event) ~ ", #name)),
            paste(features, collapse = " + "))),
            surv = T, x = T, y = T, data = data, time.inc = CUT2)
        cal5 = calibrate(
            f5, u = CUT2, cmethod = "KM", method = "boot", 
            m = floor(nrow(data) / 3), surv = T)
        p5 = round(HLtest(cal5), 3)
        x3 = c(rgb(209, 73, 85, maxColorValue = 255))
            plot(
                cal5, lty = 1, pch = 16, 
                errbar.col = x3, 
                par.corrected = list(col = x3), 
                conf.int = T, lwd = 1.2,
                xlim = c(pos, 1), ylim = c(pos, 1), 
                riskdist = F, col = "red", add = T)
    })

    x2 = c(rgb(220, 220, 220, maxColorValue = 255))
    abline(pos, 1, lty = 3, col = x2, lwd = 1)

    legend(
        "bottomright", 
        legend = c(paste0(CUT1 / 12, '-year DFS: p = ', p3), paste0(CUT2 / 12, '-year DFS: p = ', p5)), 
        col = c('blue', 'red'), 
        lwd = 2, cex = 1.5, lty = c(1, 1))
    sink()
}