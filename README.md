# MAHN_Liuwenzhi
Predicting disease-metabolite associations based on the metapath aggregation of tripartite heterogeneous networks
# MAHN for disease-metabolite associations prediction

## Dependecies
- Python 3.9
- pytorch 1.12.1
- dgl 1.1.1
- numpy 1.22.4
- pandas 1.4.4

## Dataset
disease-metabolite associations:association_DME.xlsx
disease-microbe associations:association_DMI.xlsx
microbe-metabolite associations:association_MIME.xlsx
disease semantic networks based on metapath DMED and DMID:A_DMED.xlsx and A_DMID.xlsx
metabolite semantic networks based on metapath MEDME and MEMIME: A_MEDME.xlsx and A_MEMIME.xlsx 
Disease Gaussian kernel similarity:disease_Gaussian_Simi.xlsx
Disease semantic similarity:disease_Semantic_simi.xlsx
Metabolite functional similarity:metabolite_func_simi.xlsx
Metabolite Gaussian kernel similarity:metabolite_Gaussian_Simi.xlsx
microbe Gaussian kernel similarities:microbe_Gaussian_Simi_1.xlsx and microbe_Gaussian_Simi_2.xlsx 

###### Model options
```
--epochs           int     Number of training epochs.                 Default is 1000.
--attn_size        int     Dimension of attention.                    Default is 64.
--attn_heads       int     Number of attention heads.                 Default is 6.
--out_dim          int     Output dimension after feature extraction  Default is 64.
--dropout          float   Dropout rate                               Default is 0.1.
--slope            float   Slope                                      Default is 0.2.
--lr               float   Learning rate                              Default is 0.001.
--wd               float   weight decay                               Default is 5e-3.

```

###### How to run?
```
Run main.py

```
