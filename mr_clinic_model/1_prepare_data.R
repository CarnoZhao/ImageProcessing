library(xlsx)
if (dir.exists("/wangshuo")) {
    root = "/wangshuo/zhaox"
} else {
    root = "/home/tongxueqing/zhao"
}

root = file.path(root, "ImageProcessing/mr_clinic_model/_data")
files = list.files(root, recursive = T, full.names = T)

data = data.frame()
for (file in files) { if (grepl("(xlsx|csv)$", file)) {
    split = strsplit(file, '/')[[1]]
    hosi = ifelse(grepl("ZF", split[8]), 0, 1)
    isrepeat = ifelse(grepl("^Re", split[9], ignore.case = T), 1, 0)
    series = ifelse(grepl("T2", split[10]),  2, 
             ifelse(grepl("T1C", split[10]), 3, 
             ifelse(grepl("T1", split[10]),  1, -1)))
    H_L = ifelse(grepl("HH$", split[9]), 0, 
          ifelse(grepl("LL$", split[9]), 1, -1))
    if (grepl("xlsx$", file)) {
        d = read.xlsx(file, 1)
        rownames(d) = d$name
        d$name = NULL
    } else {
        d = read.csv(file, row.names = 1)
    }
    if (H_L == 0) {
        features = colnames(d)[grepl("HH", colnames(d))], function(feature) {
            data[d$name, features] = d[,feature]
        });
    }
    data = rbind(data, d)
}}
