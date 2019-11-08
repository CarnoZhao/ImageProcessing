data = read.csv("/home/tongxueqing/zhao/ImageProcessing/mr_clinic_model/_data/clinic/ClinicMessageForAnalysis.csv", stringsAsFactors = F)
colnames(data) = c(
    "name", "realname", "smoke", "family.history", "number", 
    "T.read", "N.read", "total.cut", "gender", "age", "body.status", 
    "neuron", "EVB", "HB", "LDH", "sarcoma", "necrosis", "lymphocyte", 
    "N.cut.N3b", "total.cut.IVA", "event", "time"
    )
