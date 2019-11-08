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
root = file.path(root, "ImageProcessing/mr_clinic_model/_data")
set.seed(0)

data = read.csv(file.path(root, "mr.iccfiltered.csv"), header = T, row.names = 1, stringsAsFactors = F)
labeldata = read.csv(file.path(root, "clinic/info.csv"), header = T, row.names = 1, stringsAsFactors = F)
data[,c("time", "event")] = labeldata[match(data$name, labeldata$name), c('time', 'event')]
common.name = intersect(data$name, labeldata$name)
data = data[data$name %in% common.name,]
data$time = labeldata$time[match(data$name, labeldata$name)]
data$event = labeldata$event[match(data$name, labeldata$name)]

features = colnames(data)[1:(ncol(data) - 7)]

names = unique(data$name)
newdata = data.frame(name = names)
for (serie in unique(data$series)) {
    subdata = data[data$series == serie,]
    subdata = subdata[match(subdata$name, names), features]
    colnames(subdata) = paste(colnames(subdata), serie, sep = "_")
    for (col in colnames(subdata)) {
        newdata[,col] = subdata[,col]
    }
}
newdata$time = data[match(names, data$name), "time"]
newdata$event = data[match(names, data$name), "event"]
newdata$set = data[match(names, data$name), "set"]
newdata$name = NULL
newdata$name = names

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

# k-fold prepare
if (T) {
    valname = matrix(rep(0, 44 * 4), c(44, 4))
    v1 = c(1510261, 1706442, 1503005, 1614914, 1504357, 1510123, 1604846,
        1410116, 1511382, 1602296, 1501049, 1512396, 1503058, 1703032,
        1501050, 1407606, 1603591, 1602748, 1700572, 1509612, 1405541,
        1504545, 1401438, 1501684, 1404116, 1411030, 1409660, 1408401,
        1401150, 1500695, 1510358, 1507884, 1511384, 1600294, 1401942,
        1400974, 1407348, 1501417, 1700462, 1612055, 1613361, 1506707,
        1604426, 1602751)
    v2 = c(1610256, 1403311, 1404316, 1501611, 1702583, 1615309, 1605686,
        1401920, 1501820, 1404637, 1406199, 1405036, 1505902, 1701138,
        1508191, 1511058, 1601789, 1501010, 1615495, 1605918, 1502357,
        1604547, 1502675, 1608527, 1700618, 1409419, 1607972, 1404924,
        1601239, 1609874, 1613107, 1604176, 1604103, 1614786, 1505470,
        1701811, 1508384, 1601095, 1408018, 1702593, 1604060, 1510827,
        1512377, 1602586)
    v3 = c(1403602, 1402805, 1607174, 1509038, 1701668, 1508005, 1500694,
        1700457, 1403098, 1406407, 1601350, 1404315, 1607688, 1601181,
        1600296, 1404309, 1607236, 1406322, 1405481, 1602238, 1504186,
        1403257, 1508466, 1505057, 1503175, 1602926, 1702746, 1700342,
        1608980, 1402299, 1505414, 1402355, 1508163, 1501416, 1402909,
        1509570, 1505796, 1406489, 1701663, 1614913, 1507284, 1603444,
        1405152, 1509767)
    v4 = c(1605606, 1701475, 1610465, 1507248, 1608839, 1608463, 1512384,
        1401317, 1605283, 1508362, 1400860, 1407127, 1405035, 1700177,
        1505679, 1614456, 1510392, 1602186, 1504406, 1501415, 1605530,
        1702589, 1704471, 1507472, 1510442, 1410317, 1600991, 1602137,
        1503718, 1602620, 1615043, 1512101, 1605385, 1510941, 1403144,
        1507217, 1600870, 1602453, 1607437, 1510385, 1408272, 1410575,
        1510039, 1604044)
    valname[,1] = labeldata$name[match(v1, labeldata$number)]
    valname[,2] = labeldata$name[match(v2, labeldata$number)]
    valname[,3] = labeldata$name[match(v3, labeldata$number)]
    valname[,4] = labeldata$name[match(v4, labeldata$number)]
}

n = 20
test = newdata[newdata$set == 1,]
result = matrix(rep(0, 3 * n), c(n, 3))
for (w in (1:n) / n) {
    L = lapply(1:4, function(k) {
        train = newdata[newdata$set == 0 & !newdata$name %in% valname[,k],]
        val = newdata[newdata$set == 0 & newdata$name %in% valname[,k],]
        features = colnames(train)[1:153]

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
                    pred = predict(cox, newdata = train, type = 'lp')
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
        models = sapply(subfeatures.list, function(subfeatures) {
            ci.mat = c()
            while (length(subfeatures) != 0 && length(subfeatures) <= 10) {
                subtrain = train[,c(subfeatures, "time", "event")]
                cox = coxph(Surv(time, event) ~ . , data = subtrain)
                sink("/dev/null")
                cox = step(cox, direction = c("both"))
                sink()
                citr = to_ci(train, cox)
                civl = to_ci(val, cox)
                ci.mat = rbind(ci.mat, c(citr, civl))
                model.list[[length(model.list) + 1]] = cox
                summ = summary(cox)
                summ = as.data.frame(summ$coefficients)
                if (all(summ[,"Pr(>|z|)"] < 0.05)) {
                    break
                } else {
                    subfeatures = rownames(summ)[summ[,"Pr(>|z|)"] < 0.05]
                }
            }
            if (length(ci.mat) != 0){
                ci.mat = rowSums(ci.mat * c(w, 1 - w))
                idx = match(max(ci.mat), ci.mat)
                list("model" = model.list[[idx]], "ci.mat" = ci.mat[idx])
            } else{
                NULL
            }
        })
        models = Filter(Negate(is.null), models)
        cis = sapply(models, function(x) x$ci.mat)
        model = models[[match(max(cis), cis)]]$model
        model
    })

    result[w * n,] = c(w, combine_pred(rbind(train, val), L), combine_pred(test, L))
}

result