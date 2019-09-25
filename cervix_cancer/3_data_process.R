#======
# @Author: Xun Zhao
# @Email: zhaoxun16@mails.ucas.ac.cn
# @Date: 2019-09-20 08:49:17
# @LastEditors: Xun Zhao
# @LastEditTime: 2019-09-20 14:37:09
# @Description: basic data processing of cervix cancer data
#======

library(ggplot2)
library(dplyr)
library(Hmisc)
library(caret)
library(survcomp)
library(pROC)

d = read.csv('/home/tongxueqing/zhaox/cervix_cancer/2_filter_feature/Features_multimode_noNA_noRe_withClinicTime.csv', row.names = 1)
trainidx = d$ctTime < 20161200
testidx = d$ctTime >= 20161200
data = d[, !colnames(d) %in% c('ctTime')]
X = data[, colnames(data) != 'label']
Y = data$label
name = colnames(X)
Xnorm = as.data.frame(apply(X, 2, function(x) {(x - mean(x)) / sd(x)}))
colnames(Xnorm) = name

trainX = Xnorm[trainidx, ]
testX = Xnorm[testidx, ]
trainY = Y[trainidx]
testY = Y[testidx]

### Q: mrmr.cindex ?
mrmr.seq = mrmr.cindex(x = trainX, cl = as.numeric(trainY), method = 'norther')
mrmr.sort = mrmr.seq[order(mrmr.seq, decreasing = T)]
mfeatures = names(mrmr.sort)[1:30]
trainXsub = trainX[mfeatures]

pvalues = sapply(mfeatures, function(feature) {
    wt = wilcox.test(unlist(trainXsub[feature]) ~ trainY)
    wt$p.value
})
pvalues = pvalues[pvalues < 0.05]
pfeatures = names(pvalues)
trainXsubsub = trainXsub[pfeatures]

ctrl = rfeControl(functions = rfFuncs,
                   method = "cv",
                   verbose = F
                   )

result = rfe(trainXsubsub, trainY, 
    sizes = 1:10,
    rfeControl = ctrl
    )
rfefeatures = result$optVariables
png('/home/tongxueqing/zhaox/ImageProcessing/cervix_cancer/rfeplots.png')
plot(result, type = c('g', 'o'), xlim = 0:11, pch = 1, lwd = 2)
dev.off()

trainXlr = trainXsubsub[rfefeatures]
testXlr = testX[colnames(trainXlr)]
trainXlr$label = trainY
fit = glm(label ~ ., data = trainXlr, family = binomial(link = 'logit'))
trainXlr$p = predict(fit, newdata = trainXlr, type = 'response')
testXlr$label = testY
testXlr$p = predict(fit, newdata = testXlr, type = 'response')

trainROC = roc(label ~ p, trainXlr, col = "1", print.auc = T)
testROC = roc(label ~ p, testXlr, col = "2", print.auc = T)
png('/home/tongxueqing/zhaox/ImageProcessing/cervix_cancer/rocplots.png')
plot(testROC)
dev.off()

logistic_regression = function(data, label, testdata, testlabel, iter = 10000, lr = 0.001) {
    data = as.matrix(data) # m n
    m = dim(data)[1]
    n = dim(data)[2]
    W = matrix(rnorm(n), n, 1) # n 1
    b = 0 # 1 1
    for (i in 1:iter) {
        Z = data %*% W + b # m 1
        A = 1 / (1 + exp(-Z))
        dZ = A - label
        dW = t(data) %*% dZ / m
        db = mean(dZ)
        W = W - lr * dW
        b = b - lr * db
        if (i %% 200 == 0) {
            testA = 1 / (1 + exp(-(testdata %*% W + b)))
            loss = -mean(label * log(A) + (1 - label) * log(1 - A))
            testloss = -mean(testlabel * log(testA) + (1 - testlabel) * log(1 - testA))
            print(paste('after', i, 'iterations, the cost is', loss, '/', testloss, sep = ' '))
        }
    }
    print(mean((A > 0.5) == label))
    print(mean((testA > 0.5) == testlabel))
    list(W = W, b = b)
}