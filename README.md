# MAHN
Predicting disease-metabolite associations based on the metapath aggregation of tripartite heterogeneous networks

## 🏠 Overview
![image](https://github.com/Lwz1225/MAHN/assets/127914409/ddd7ad49-8a8c-4f67-8287-d5900db5f0c7)


## 🛠️ Dependecies
```
- Python == 3.9
- pytorch == 1.12.1
- dgl == 1.1.1
- numpy == 1.22.4+mkl
- pandas == 1.4.4
```

## 🗓️ Dataset
```
- disease-metabolite associations: association_DME.xlsx
- disease-microbe associations: association_DMI.xlsx
- microbe-metabolite associations: association_MIME.xlsx
- disease semantic networks based on metapath DMED and DMID: A_DME_D.xlsx and A_DMI_D.xlsx
- metabolite semantic networks based on metapath MEDME and MEMIME: A_DME_ME.xlsx and A_MIME_ME.xlsx 
- disease Gaussian kernel similarity: disease_Gaussian_Simi.xlsx
- disease semantic similarity: disease_Semantic_simi.xlsx
- metabolite functional similarity: metabolite_func_simi.xlsx
- metabolite Gaussian kernel similarity: metabolite_Gaussian_Simi.xlsx
- microbe Gaussian kernel similarities: microbe_Gaussian_Simi_1.xlsx and microbe_Gaussian_Simi_2.xlsx 
```

## 🛠️ Model options
```
--epochs           int     Number of training epochs.                 Default is 1000.
--attn_size        int     Dimension of attention.                    Default is 64.
--attn_heads       int     Number of attention heads.                 Default is 6.
--out_dim          int     Output dimension after feature extraction  Default is 64.
--sampling number  int     enhanced GraphSAGE sampling number         Default is 50.
--dropout          float   Dropout rate                               Default is 0.2.
--slope            float   Slope                                      Default is 0.2.
--lr               float   Learning rate                              Default is 0.001.
--wd               float   weight decay                               Default is 5e-3.

```

## 🎯 How to run?
```
1、Loading various associations and similarities in the data folder
2、Running main.py in the my_code folder calls train.py, model.py, layers.py and utils.py to get the experimental results

```
