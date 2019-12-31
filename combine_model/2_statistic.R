train_val = read.csv("/home/tongxueqing/zhao/ImageProcessing/combine_model/_data/ZForigcsv.csv")
train = train_val[train_val$集合 == "Train",]
val = train_val[train_val$集合 == "Val",]
test = read.csv("/home/tongxueqing/zhao/ImageProcessing/combine_model/_data/gxorig.csv")
test = test[test$病案号 != 1418116,]

for (data in list(train, val, test)){
    a = table(data$HB & data$HB)
    b = round(100 * a / nrow(data), 2)
    a
    c = paste(a, " (", b, "%)\n", sep = "")
    cat(paste(c, collapse = ""))
    print("")
}


test$HB = c(105, 129, 99, 135, 146, 160, 121, 119, 106, 114, 124, 121, 112, 101, 136, 141, 86, 114, 176, 133, 117, 125, 133, 128, 134, 105, 114, 73, 114, 120, 132, 124, 145, 139, 111, 134, 148, 80, 131, 112, 113, 146, 146, 83)
name = "发病年龄"
df = data.frame(name = c(train[,name], val[,name], test[,name]), set = c(rep(1, nrow(train)), rep(2, nrow(val)), rep(3, nrow(test))))
kruskal.test(df$name ~ df$set)

name = "淋巴结坏死"
train$淋巴结坏死 = train$颈部坏死 | train$咽后坏死
val$淋巴结坏死 = val$颈部坏死 | val$咽后坏死
test$淋巴结坏死 = test$颈部坏死 | test$咽后坏死
a = unique(c(train[,name], val[,name], test[,name])); a = a[order(a)]
x = sapply(list(train, val, test), function(data) {
    sapply(a, function(ai) sum(data[,name] == ai))
})
chisq.test(x)