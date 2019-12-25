library(pheatmap)
library(ComplexHeatmap)

gene = read.csv("/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/gene.csv", stringsAsFactors = F, row.names = 1)
nfkb = c("CYLD", "NFKBIA")
pik = c("MAP2K1", "MAPK8IP1", "MTOR", "PIK3C2G", "FGF19", "FGF3", "FGF4", "FGFR3", "PIK3CA", "TEK", "HGF", "IRS2", "HSP90AA1", "VEGFA", "SH2B3")
remdl = c("ARID1A", "KMT2C", "KMT2D", "KDM5C", "KDM6A", "DNMT3A") 
dnarp = c("BACH1", "BRCA2", "NBN", "MSH5", "RAD52")
cycle = c("TP53", "EP300", "CCND1", "MYC", "STAG2")
grps = list(nfkb = nfkb, pik = pik, remdl = remdl, dnarp = dnarp, cycle = cycle)
gs = colnames(gene)[9:66]
gene = cbind(gene[,!colnames(gene) %in% gs], sapply(grps, function(clst) {
    as.numeric(suppressWarnings(apply(gene[,clst], 1, any)))
}))
colSums(gene[,names(grps)])

mr = read.csv("/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/mr.merged.csv")

have = list(
    'T1' = c(
        "log.sigma.4.0.mm.3D_glszm_GrayLevelNonUniformity", 
        "original_shape_Flatness", 
        "wavelet.LH_glcm_ClusterShade"
    ),
    'T2' = c(
        "original_shape_Flatness",
        "log.sigma.2.0.mm.3D_gldm_DependenceNonUniformityNormalized",
        "wavelet.LL_gldm_DependenceNonUniformity",
        "log.sigma.2.0.mm.3D_glrlm_RunLengthNonUniformityNormalized"
    ),
    "T1C" = c(
        "original_shape_Flatness",
        "original_shape_Maximum2DDiameterColumn",
        "original_shape_Maximum2DDiameterRow",
        "original_shape_SurfaceVolumeRatio",
        "original_glszm_SizeZoneNonUniformity"
    )
)

dp.sig = read.csv("/home/tongxueqing/zhao/ImageProcessing/gene_association/_data/DP.sig.csv")

main = function(p) {
    features = lapply(names(grps), function(grp) {
        notID = gene$ID[gene[,grp] == 0]
        yesID = gene$ID[gene[,grp] == 1]
        tmp = lapply(c("T1", "T2", "T1C"), function(seq) {
            submr = mr[mr$series == match(seq, c("T1", "T2", "T1C")),]
            submryes = submr[submr$name %in% yesID,]
            submrnot = submr[submr$name %in% notID,]

            p.values = sapply(colnames(submr)[1:789], function(col) {
                tryCatch({t.test(x = submryes[,col], y = submrnot[,col])$p.value}, 
                error = function(cond) {1})
            })
            nm = names(p.values)[p.values < p]
            nm[nm %in% have[[seq]]]
        }); names(tmp) = c("T1", "T2", "T1C")
        dp.p = t.test(x = dp.sig$dp_sig[dp.sig$name %in% notID], y = dp.sig$dp_sig[dp.sig$name %in% yesID])$p.value
        if (dp.p < p) {
            tmp[['deep']] = 'yes'
        } else {
            tmp[['deep']] = list()
        }
        tmp
    }); names(features) = names(grps)
    print(paste0(p, ":"))
    print(sapply(features, function(a) sapply(a, length)))
}

for (p in c(0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001)) {
    main(p)
}

pss = lapply(names(have), function(seq) {
    x = sapply(names(grps), function(grp) {
        notID = gene$ID[gene[,grp] == 0]
        yesID = gene$ID[gene[,grp] == 1]
        submr = mr[mr$series == match(seq, c("T1", "T2", "T1C")),]
        submryes = submr[submr$name %in% yesID,]
        submrnot = submr[submr$name %in% notID,]

        p.values = sapply(colnames(submr)[1:789], function(col) {
            tryCatch({t.test(x = submryes[,col], y = submrnot[,col])$p.value}, 
            error = function(cond) {1})
        })
        p.values[have[[seq]]]
    })
})
pss = do.call(rbind, pss)

pheatmap(-log10(pss), filename = "/home/tongxueqing/zhao/ImageProcessing/gene_association/_plots/p.values.png", treeheight_row = 0, treeheight_col = 0, width = 10, height = 6, main = "-log10(p)")

library(ggplot2)
png("/home/tongxueqing/zhao/ImageProcessing/gene_association/_plots/p.values.comp.png", width = 1000, height = 600)
Heatmap(
    -log10(pss), 
    cell_fun = function(i, j, x, y, width, height, fill) {
        v = signif(pss[j, i], 3)
        if (pss[j, i] < 0.05) {
            v = paste0(v, ' *')
        }
        grid.text(v, x = x, y = y)
    },
    show_heatmap_legend = F,
    show_column_dend = F,
    show_row_dend = F,
    column_title_rot = 0
)
dev.off()