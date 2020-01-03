train_val = read.csv("/wangshuo/zhaox/ImageProcessing/combine_model/_data/ZForigcsv.csv")
trainname = read.table("/wangshuo/zhaox/ImageProcessing/combine_model/_data/new_set.txt")

cat(paste(paste(train$病案号, train$病理号), collapse = "\n"))
# train = train_val[!train_val$病案号 %in% trainname$V1 & train_val,]
test = read.csv("/wangshuo/zhaox/ImageProcessing/combine_model/_data/gxorig.csv")
test = test[test$病案号 != 1418116,]
train_val$淋巴结坏死 = train_val$颈部坏死 | train_val$咽后坏死
test$淋巴结坏死 = test$颈部坏死 | test$咽后坏死
train_val$WHO病理类型 = ifelse(train_val$WHO病理类型 == "III", 3, 2)
test$WHO病理类型 = 3
test$HB = c(105, 129, 99, 135, 146, 160, 121, 119, 106, 114, 124, 121, 112, 101, 136, 141, 86, 114, 176, 133, 117, 125, 133, 128, 134, 105, 114, 73, 114, 120, 132, 124, 145, 139, 111, 134, 148, 80, 131, 112, 113, 146, 146, 83)
train = train_val[train_val$病案号 %in% trainname$V1,]
val = train_val[!train_val$病案号 %in% trainname$V1 & train_val$集合 != "None",]
# for (data in list(train, val, test)){
#     a = table(data$HB & data$HB)
#     b = round(100 * a / nrow(data), 2)
#     a
#     c = paste(a, " (", b, "%)\n", sep = "")
#     cat(paste(c, collapse = ""))
# }

# name = "发病年龄"
# df = data.frame(name = c(train[,name], val[,name], test[,name]), set = c(rep(1, nrow(train)), rep(2, nrow(val)), rep(3, nrow(test))))
# kruskal.test(df$name ~ df$set)$p.value
# wilcox.test(train[,name], val[,name])$p.value

ps = 0
cnt = 0
while (any(ps < 0.9)) {
    cnt = cnt + 1
    train_val$集合[train_val$集合 %in% c("Train", "Val")] = sample(c(rep("Train", 132), rep("Val", 44)))
    train = train_val[train_val$集合 == "Train",]
    val = train_val[train_val$集合 == "Val",]

    names = c("性别", "总分期", "WHO病理类型", "T读片", "N读片", "瘤内坏死", "淋巴细胞浸润总", "肉瘤样细胞", "淋巴结坏死", "发生治疗失败")
    ps = c()
    for(name in names) {
        a = unique(c(train[,name], val[,name], test[,name])); a = a[order(a)]
        x = sapply(list(train, val, test), function(data) {
            sapply(a, function(ai) sum(data[,name] == ai))
        })
        suppressWarnings({p = chisq.test(x)$p.value})
        ps = c(ps, p)
    }
    # for(name in names) {
    #     a = unique(c(train[,name], val[,name])); a = a[order(a)]
    #     x = sapply(list(train, val), function(data) {
    #         sapply(a, function(ai) sum(data[,name] == ai))
    #     })
    #     suppressWarnings({p = chisq.test(x)$p.value})
    #     ps = c(ps, p)
    # }
    ps = c(ps, wilcox.test(train[,"发病年龄"], val[,"发病年龄"])$p.value)
    ps = c(ps, wilcox.test(train[,"HB"], val[,"HB"])$p.value)
    names(ps) = c(names, "发病年龄", "HB")
    # print(ps[ps < 0.05])
    ps
}