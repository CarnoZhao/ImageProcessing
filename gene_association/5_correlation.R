gene = read.csv("/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/gene.csv")
g = 'KMT2D'
not = gene[gene[,g] == 0,]
yes = gene[gene[,g] == 1,]

notID = not$ID
yesID = yes$ID

dpsig = read.csv('/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/DP.sig.csv')
dpsig$realname = 
notdp = dpsig$dp_sig[dpsig$name %in% notID]
yesdp = dpsig$dp_sig[dpsig$name %in% yesID]

mr = read.csv("/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/mr.merged.csv")
mr1 = mr[mr$series == 1,]
mr1yes = mr1[mr1$name %in% yesID,]
mr1not = mr1[mr1$name %in% notID,]

for (col in colnames(mr1)[1:790]) {
    
}

p.values = sapply(colnames(mr1)[1:789], function(col) {
    t.test(x = mr1yes[,col], y = mr1not[,col])$p.value
})