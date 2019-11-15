suppressPackageStartupMessages({
    library(survivalROC)
    library(lattice)
    library(Formula)
    library(ggplot2)
    library(Hmisc)
    library(rms)
})

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
        v.line = c(38,60),
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
        predict.time = 36, 
        span = 0.001 * n ^ (-0.2))
    t2 = survivalROC(
        Stime = data$time, 
        status = data$event, 
        data[,name], 
        predict.time = 60, 
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
            paste0('3-year: AUC = ', round(t1$AUC, 3)), 
            paste0('5-year: AUC = ', round(t2$AUC, 3))),
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
            function(x) surv.prob(36, x),
            function(x) surv.prob(60, x)),
        funlabel = c("3-year DFS rate", "5-year DFS rate"),
        fun.at = 10:0 / 10,
        lp = F)
    plot(nom, xfrac = 0.3, cex.axis = 1.3, cex.var = 1.3)
    par(mfrow = c(1, 2))
}

calibration_plot = function(data, name) {
    pos = 0
    sink("/dev/null")
    f3 = cph(
        as.formula(paste0("Surv(time, event) ~ ", name)), 
        surv = T, x = T, y = T, data = data, time.inc = 36)
    f5 = cph(
        as.formula(paste0("Surv(time, event) ~ ", name)), 
        surv = T, x = T, y = T, data = data, time.inc = 60)
    cal3 = calibrate(
        f3, u = 36, cmethod = "KM", method = "boot", 
        B = nrow(data) * 3, m = floor(nrow(data) / 3), surv = T, time.inc = 36)
    cal5 = calibrate(
        f5, u = 60, cmethod = "KM", method = "boot", 
        B = nrow(data) * 3, m = floor(nrow(data) / 3), surv = T, time.inc = 60)

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

    x2 = c(rgb(220, 220, 220, maxColorValue = 255))
    abline(pos, 1, lty = 3, col = x2, lwd = 1)

    x3 = c(rgb(209, 73, 85, maxColorValue = 255))
    plot(
        cal5, lty = 1, pch = 16, 
        errbar.col = x3, 
        par.corrected = list(col = x3), 
        conf.int = T, lwd = 1.2,
        xlim = c(pos, 1), ylim = c(pos, 1), 
        riskdist = F, col = "red", add = T)

    # axis(1, at = seq(pos, 1, 0.1), labels = seq(pos, 1, 0.1), pos = pos)
    # axis(2, at = seq(pos, 1, 0.1), labels = seq(pos, 1, 0.1), pos = pos)

    legend(
        "bottomright", 
        legend = c('3-year DFS: p = ','5-year DFS: p = '), 
        col = c('blue', 'red'), 
        lwd = 2, cex = 1.5, lty = c(1, 1))
    sink()
}