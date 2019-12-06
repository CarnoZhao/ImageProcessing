from Bio import Entrez
from bs4 import BeautifulSoup

Entrez.email = 'zhaoxun16@mails.ucas.ac.cn'

genes = "ARHGAP35,ARID1A,BACH1,BCORL1,BRCA2,CASP8,CBFB,CCND1,CYLD,CYP2C8,DNMT3A,EP300,FGF19,FGF3,FGF4,FGFR3,HGF,HSP90AA1,INHBA,IRS2,KDM5C,KDM6A,KIAA1549,KIF1B,KMT2C,KMT2D,MAP2K1,MAPK8IP1,MED12,MSH5,MTOR,MYC,NAV3,NBN,NFKBIA,PCBP1,PIK3C2G,PIK3CA,PTPRD,RAD52,ROBO1,ROBO2,RPL5,SETD2,SH2B3,SIN3A,SLAMF7,SMARCA4,STAG2,TAF1,TEK,TFG,TOP2B,TP53,TShz2,UPF1,VEGFA,ZNF703".split(',')

results = {}
for gene in genes:
    x = Entrez.esearch('gene', term = '(%s[Gene Name]) AND Homo Sapien[Organism]' % gene)
    x = Entrez.read(x)
    x = Entrez.efetch(db = "gene", id = x['IdList'][0], retmode = 'XML')
    x = x.read()

    soup = BeautifulSoup(x)

    prop = soup.find("entrezgene_properties")
    goall = prop.findChildren("gene-commentary", recursive = False)[2]
    go = goall.findChild("gene-commentary_comment", recursive = False)
    print(gene)
    result = {key.text: key for key in go.find_all("gene-commentary_label")}
    for k in result:
        sub = result[k].findNextSibling("gene-commentary_comment")
        subs = sub.findChildren("other-source_anchor")
        txts = [s.text for s in subs]
        result[k] = txts
    results[gene] = result

fcs, prs, cos = [], [], []
for g in results:
    if 'Function' in results[g]:
        fcs.extend(results[g]['Function'])
    if 'Process' in results[g]:
        prs.extend(results[g]["Process"])
    if 'Component' in results[g]:
        cos.extend(results[g]['Component'])

from collections import Counter
highfc = sorted(list(Counter(fcs).items()), key = lambda x: x[1], reverse = True)[:5]
highpr = sorted(list(Counter(prs).items()), key = lambda x: x[1], reverse = True)[:5]
highco = sorted(list(Counter(cos).items()), key = lambda x: x[1], reverse = True)[:5]




