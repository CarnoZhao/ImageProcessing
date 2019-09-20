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

d = read.csv('/home/tongxueqing/zhaox/cervix_cancer/2_filter_feature/Features_multimode_noNA_noRe_withClinicTime.csv', row.names = 1)
trainidx = d$ctTime < 20161200
testidx = d$ctTime >= 20161200
data = d[, !colnames(d) %in% c('HE4', 'ctTime')]
X = as.matrix(data[, colnames(data) != 'label'])
Y = as.matrix(data$label)
Xnorm = apply(X, 2, function(x) {(x - mean(x)) / sd(x)})

trainX = Xnorm[trainidx, ]
testX = Xnorm[testidx, ]
trainY = Y[trainidx, ]
testY = Y[testidx, ]

pvalue = function(X, Y) {
    sapply(colnames(X), function(name) {
        wilcox.test(X[Y == 1, name], X[Y == 0, name])$p.value
    })
}

ptrain = pvalue(trainX, trainY)
features1 = names(ptrain[ptrain < 0.05])
trainXsub = trainX[,features1]
testXsub = testX[,features1]

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

# lrout = logistic_regression(trainXsub, trainY, testXsub, testY, iter = 10000, lr = 0.01)
# W = lrout$W
# b = lrout$b
# rownames(W)[order(abs(W), decreasing = T)[1:10]]

ctrl = rfeControl(functions = rfFuncs,
                   method = "cv",
                   verbose = F)

result = rfe(trainXsub, trainY, 
    sizes = 1:10,
    rfeControl = ctrl)