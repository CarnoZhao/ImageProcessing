acnt = 1
root = "/wangshuo/zhaox"
while (acnt < 100) {  
    print(acnt)  
    try({        
        if(T) {    
            train_val = read.csv("/wangshuo/zhaox/ImageProcessing/combine_model/_data/ZForigcsv.csv")
            trainname = read.table("/wangshuo/zhaox/ImageProcessing/combine_model/_data/new_set.txt")

            test = read.csv("/wangshuo/zhaox/ImageProcessing/combine_model/_data/gxorig.csv")
            test = test[test$病案号 != 1418116,]
            train_val$淋巴结坏死 = train_val$颈部坏死 | train_val$咽后坏死
            test$淋巴结坏死 = test$颈部坏死 | test$咽后坏死
            train_val$WHO病理类型 = ifelse(train_val$WHO病理类型 == "III", 3, 2)
            test$WHO病理类型 = 3
            test$HB = c(105, 129, 99, 135, 146, 160, 121, 119, 106, 114, 124, 121, 112, 101, 136, 141, 86, 114, 176, 133, 117, 125, 133, 128, 134, 105, 114, 73, 114, 120, 132, 124, 145, 139, 111, 134, 148, 80, 131, 112, 113, 146, 146, 83)
            train = train_val[train_val$病案号 %in% trainname$V1,]
            val = train_val[!train_val$病案号 %in% trainname$V1 & train_val$集合 != "None",]

            ps = 0
            cnt = 0
            while (any(ps < 0.075)) {
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
                ps = c(ps, wilcox.test(train[,"发病年龄"], val[,"发病年龄"])$p.value)
                ps = c(ps, wilcox.test(train[,"HB"], val[,"HB"])$p.value)
                names(ps) = c(names, "发病年龄", "HB")
                ps
            }
            write.table(train[,c("病案号", "病理号")], "/wangshuo/zhaox/ImageProcessing/combine_model/_data/new_set copy.txt", sep = "\t", row.names = F, col.names = F, quote = F)
            system(paste0("mkdir ", file.path(root, "ImageProcessing/combine_model/_data/newsets", acnt)))
            write.table(train[,c("病案号", "病理号")], file.path(root, "ImageProcessing/combine_model/_data/newsets", acnt, "newset.txt"), sep = "\t", row.names = F, col.names = F, quote = F)
        }

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
                })
                if (dir.exists("/wangshuo")) {
                    root = "/wangshuo/zhaox"
                } else {
                    root = "/home/tongxueqing/zhao"
                }

                data = read.csv(file.path(root, "ImageProcessing/mr_clinic_model/_data/mr.iccfiltered.csv"), header = T, row.names = 1, stringsAsFactors = F)
                labeldata = read.csv(file.path(root, "ImageProcessing/mr_clinic_model/_data/clinic/info.csv"), header = T, row.names = 1, stringsAsFactors = F)
                data[,c("time", "event")] = labeldata[match(data$name, labeldata$name), c('time', 'event')]
                common.name = intersect(data$name, labeldata$name)
                data = data[data$name %in% common.name,]
                data$time = labeldata$time[match(data$name, labeldata$name)]
                data$event = labeldata$event[match(data$name, labeldata$name)]

                to_ci = function(set, cox = cox.new) {
                    pred = predict(cox, newdata = set, type = "lp")
                    ci = concordance.index(x = pred, surv.time = set$time, surv.event = set$event, method = "noether")  
                    ci$c.index  
                    # paste(round(ci$c.index, 3), " (", round(ci$lower, 3), "-", round(ci$upper, 3), ") ", signif(ci$p.value, 3), sep = "")
                }

                combine_pred = function(set, models = L, a = c(1, 1, 1)) {
                    names = set[set$series == 1,]$name
                    time = set[set$series == 1,]$time
                    event = set[set$series == 1,]$event
                    preds = sapply(1:3, function(s) {
                        s.set = set[set$series == s,]
                        s.set = s.set[match(s.set$name, names),]
                        pred = predict(models[[s]], newdata = s.set, type = "lp")
                    })
                    # pred = rowMeans(preds)
                    # pred = apply(preds, 1, function(x) sum(a * x) / sum(a))
                    pred = apply(preds, 1, max)
                    ci = concordance.index(x = pred, surv.time = time, surv.event = event, method = "noether")
                    ci$c.index
                }


                pat_set = read.table(file.path(root, "ImageProcessing/combine_model/_data/new_set copy.txt"))$V1

                trainAll = data[data$set == 0 & data$name %in% pat_set,]
                valAll = data[data$set == 0 & !data$name %in% pat_set,]
                test = data[data$set == 1,]
            }

            n = 20
            models.all = lapply(1:3, function(s) {
                train = trainAll[trainAll$series == s,]
                val = valAll[valAll$series == s,]
                features = colnames(train)[complete.cases(t(train))]
                features = features[!grepl("(name|set|time|event|series)", features)]

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
                            pred = predict(cox, data = train, type = 'lp')
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
                        subfeatures = names(mrmr.result)[1:min(20, length(mrmr.result))]
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
                models = lapply(subfeatures.list, function(subfeatures) {
                    model.list = list()
                    # if (length(subfeatures) > 20) {
                    #     NULL
                    # } else {
                    #     subtrain = train[,c(subfeatures, "time", "event")]
                    #     cox = coxph(Surv(time, event) ~ . , data = subtrain)
                    #     sink("/dev/null")
                    #     cox = step(cox, direction = "backward")
                    #     sink()
                    #     model.list[[length(model.list) + 1]] = cox
                    # }
                    while (length(subfeatures) != 0 && length(subfeatures) <= 20) {
                        subtrain = train[,c(subfeatures, "time", "event")]
                        cox = coxph(Surv(time, event) ~ . , data = subtrain)
                        sink("/dev/null")
                        cox = step(cox, direction = "backward")
                        sink()
                        model.list[[length(model.list) + 1]] = cox
                        summ = summary(cox)
                        summ = as.data.frame(summ$coefficients)
                        if (all(summ[,"Pr(>|z|)"] < 0.05)) {
                            break
                        } else {
                            subfeatures = rownames(summ)[summ[,"Pr(>|z|)"] < 0.05]
                        }
                    }
                    if (length(model.list) == 0) {NULL}
                    else {model.list}
                })
                do.call(c, models)
            })

            result = matrix(rep(0, 4 * n), c(n, 4))
            Ls = list()
            for(w in 1:n / n) {
                L = lapply(1:3, function(s) {
                    models = models.all[[s]]
                    train = trainAll[trainAll$series == s,]
                    val = valAll[valAll$series == s,]
                    models = Filter(Negate(is.null), models)
                    cis = sapply(models, function(cox) {
                        citr = to_ci(train, cox)
                        civl = to_ci(val, cox)
                        citr * w + (1 - w) * civl
                    })
                    model = models[[match(max(cis), cis)]]
                    model
                })
                citr = combine_pred(trainAll, L)
                civl = combine_pred(valAll, L)
                cits = combine_pred(test, L)
                result[w * n,] = c(w, citr, civl, cits)
                Ls[[w * n]] = L
            }

            result = t(sapply(1:n / n, function(w) {
                L = Ls[[w * n]]
                citr = combine_pred(trainAll, L)
                civl = combine_pred(valAll, L)
                cits = combine_pred(test, L)
                c(w, citr, civl, cits)
            }))
            result
            result[order(result[,3]),]
            L = Ls[[match(max(result[,3]), result[,3])]]
            saveRDS(L, file.path(root, "ImageProcessing/combine_model/_data/newsets", acnt, "L.rds"))

            newpreds = sapply(1:3, function(s) {
                cox = L[[s]]
                pred = predict(cox, newdata = data[data$series == s,], type = "lp")
            })

            cname = paste("mr_serie", 1:3, sep = '')
            preds = read.csv(file.path(root, "ImageProcessing/combine_model/_data/preds.csv"), row.names = 1, stringsAsFactors = F)
            preds[,grepl("(mr|cli)_fold", colnames(preds))] = NULL
            preds[,"sig_mr"] = apply(newpreds, 1, max)
            preds$set = ifelse(preds$name %in% test$name, 1, 0)
            write.csv(preds, file.path(root, "ImageProcessing/combine_model/_data/preds.csv"))
            write.csv(preds, file.path(root, "ImageProcessing/combine_model/_data/newsets", acnt, "preds.csv"))
        }

        if(T) {
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
                        # library(xlsx)
                    })
                    if (dir.exists("/wangshuo")) {
                        root = "/wangshuo/zhaox/"
                    } else {
                        root = "/home/tongxueqing/zhao/"
                    }
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
                pat_set = read.table(file.path(root, "ImageProcessing/combine_model/_data/new_set copy.txt"))$V1
                train_val = preds[preds$set == 0,]
                test = preds[preds$set == 1,]
                train = train_val[train_val$name %in% pat_set,]
                val = train_val[!train_val$name %in% pat_set,]
                options(datadist = 'data.dist')
            }
            if (T) {
                all.clis = colnames(info)[!grepl("(EVB|name|number|set|time|event)", colnames(info))]
                ps = sapply(all.clis, function(col) {
                    subtrain = train[complete.cases(train[,col]),]
                    time = subtrain$time
                    event = subtrain$event
                    hazard.ratio(subtrain[,col], surv.time = time, surv.event = event)$p.value
                })
                clis = names(ps)[ps < 0.05 & !is.na(ps)]
                clis = clis[clis != "lympho_crosis"]
                cli.cox = coxph(as.formula(paste0("Surv(time, event) ~ ", paste(clis, collapse = " + "))), data = train)
                sink("/dev/null")
                cox = step(cli.cox, directions = "both")
                sink()
                clis = names(cox$coefficients)
            } else {
                clis = c("age", "N.read", "lymphocyte")
            }
            features.list = list(
                "deep" = "sig_deep",
                "mr" = "sig_mr",
                "cli" = clis,
                "deep_mr" = c("sig_deep", "sig_mr"),
                "deep_mr_cli" = c("sig_deep", "sig_mr", clis),
                "doctor" = c("necrosis", "lymphocyte")
            )
            result = matrix(rep(0, length(features.list) * 3), c(length(features.list), 3))
            row.names(result) = names(features.list)
            colnames(result) = c("citr", "civl", "cits")
            models = list()
            for (name in names(features.list)) {
                features = features.list[[name]]
                res = make_sig(train, features)
                cox = res$model
                ci = res$ci
                models[[length(models) + 1]] = cox
                citr = to_ci(train, cox)
                civl = to_ci(val, cox)
                cits = to_ci(test, cox)
                result[match(name, names(features.list)),] = c(citr, civl, cits)
                train[,name] = to_pred(train, cox)$pred
                val[,name] = to_pred(val, cox)$pred
                test[,name] = to_pred(test, cox)$pred
            }
            saveRDS(models, file.path(root, "ImageProcessing/combine_model/_data/newsets", acnt, "models.rds"))
            names(models) = names(features.list)
            print(result[order(rowMeans(result[,1:2])),])
        }
    })
    acnt = acnt + 1
}