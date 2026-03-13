# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:23:18 2021

@author: Jing Li, Small steps make changes. dnt_seq@163.com
"""

from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

amino_table = ['I', 'D', 'M', 'H', 'E', 'W', 'R', 'L', 'Y', 'Q', 'G', 'A', 'S', 'P', 'C', 'T', 'V', 'F', 'N', 'K']
nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
codon_table = [nt1+nt2+nt3 for nt1 in nt_table for nt2 in nt_table for nt3 in nt_table]
codon_table1 = codon_table.copy()
codon_table1.remove('taa')
codon_table1.remove('tag')
codon_table1.remove('tga')
codonpair_table = [codon0 + codon1 for codon0 in codon_table1 for codon1 in codon_table1]
nt_category = ['n1','n2','n3']
dnt_category = ['n12', 'n23','n31']
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
nts_cols_list = ['Freq_'+ nt + '_' + nt_cat for nt_cat in nt_category for nt in nt_table]
dnts_cols_list = ['Freq_'+ dnt + '_' + dnt_cat for dnt_cat in dnt_category for dnt in dnt_table]
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]
codon_cols_list = ['Freq_'+ codon for codon in codon_table]
codonpair_cols_list = ['Freq_'+ codonpair for codonpair in codonpair_table]
amino_cols_list = ['Freq_'+ amino for amino in amino_table]
# full_cols_list0 = dnts_cols_list + dntpair_cols_list + codon_cols_list + codonpair_cols_list + amino_cols_list

dcr_set_name_list = ['nt','dnts', 'codons', 'aminos', 'DCR','codonpair']
dcr_set_list = [nts_cols_list,dnts_cols_list,codon_cols_list, amino_cols_list,  dntpair_cols_list, codonpair_cols_list]

cnames = {
'lightblue':            '#ADD8E6',
'deepskyblue':          '#00BFFF',
'cadetblue':            '#5F9EA0',
'cyan':                 '#00FFFF',
'purple':               '#800080',
'orchid':               '#DA70D6',
'lightgreen':           '#90EE90',
'darkgreen':            '#006400',
'red':                  '#FF0000',
'darkred':              '#8B0000',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32',
'deeppink':             '#FF1493',
'burlywood':            '#DEB887',
'indianred':            '#CD5C5C',
    }

cnames = {
'red':                  '#FF0000',
'deepskyblue':          '#00BFFF',
'darkgreen':            '#006400',
'lightgreen':           '#90EE90',
'purple':               '#800080',
'orchid':               '#DA70D6',
'deepskyblue':          '#00BFFF',
'darkgreen':            '#006400',
'lightblue':            '#ADD8E6',
'darkred':              '#8B0000',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32',
'deeppink':             '#FF1493',
'burlywood':            '#DEB887',
'indianred':            '#CD5C5C',
    }




# color_num_list = list (range(1,16,1))
color_num_list = list (range(1,16,1))

# print (len(color_num_list))
color_dict = dict(zip(color_num_list,cnames.values()))
# print (color_dict)
color_list0 = list(color_dict.values())
color_list_all = []
for i in color_list0:
    color_list_all.append(i)


gene_list = ['ORF1ab']
df0_CoVs = pd.read_csv ('../data/df_Coronaviridae_host_labels.csv')
df0_CoVs1 = pd.read_csv ('../data/df_CHI_CoVs_SARS0_1.csv')


df0_CoVs_Chiroptera0 = df0_CoVs1 [(df0_CoVs1 ['host']=='Chiroptera')].sample (frac = 0.45, random_state = 10)
print (df0_CoVs_Chiroptera0.shape)
df0_CoVs_Chiroptera0['adaptation'] = [0]*df0_CoVs_Chiroptera0.shape[0]
print (df0_CoVs_Chiroptera0.shape)

df0_CoVs_Chiroptera1 = df0_CoVs1 [(df0_CoVs1 ['host']=='Chiroptera_1')].sample (frac = 0.26, random_state = 10)
print (df0_CoVs_Chiroptera1.shape)
df0_CoVs_Chiroptera1['adaptation'] = [0]*df0_CoVs_Chiroptera1.shape[0]
print (df0_CoVs_Chiroptera1.shape)


df0_CoVs_Chiroptera=pd.concat ([df0_CoVs_Chiroptera0,df0_CoVs_Chiroptera1],axis = 0)



df0_CoVs_Artiodactyla = df0_CoVs [(df0_CoVs ['host']=='Artiodactyla')].sample (frac = 0.17, random_state = 10)
print (df0_CoVs_Artiodactyla.shape)
df0_CoVs_Artiodactyla['adaptation'] = [1]*df0_CoVs_Artiodactyla.shape[0]
print (df0_CoVs_Artiodactyla.shape)

df0_CoVs_Rodent_Lagomorpha = df0_CoVs [(df0_CoVs ['host']=='Rodent_Lagomorpha')]
print (df0_CoVs_Rodent_Lagomorpha.shape)
df0_CoVs_Rodent_Lagomorpha['adaptation'] = [2]*df0_CoVs_Rodent_Lagomorpha.shape[0]
print (df0_CoVs_Rodent_Lagomorpha.shape)

df0_CoVs_Carnivora = df0_CoVs [(df0_CoVs ['host']=='Carnivora')]
print (df0_CoVs_Carnivora.shape)
df0_CoVs_Carnivora['adaptation'] = [3]*df0_CoVs_Carnivora.shape[0]
print (df0_CoVs_Carnivora.shape)

df0_CoVs_Primates = df0_CoVs [(df0_CoVs ['host']=='Primates')].sample (frac = 0.17, random_state = 10)
print (df0_CoVs_Primates.shape)
df0_CoVs_Primates['adaptation'] = [4]*df0_CoVs_Primates.shape[0]
print (df0_CoVs_Primates.shape)


df0_CoVs_all = pd.concat ([df0_CoVs_Chiroptera, df0_CoVs_Artiodactyla, df0_CoVs_Rodent_Lagomorpha, df0_CoVs_Carnivora, df0_CoVs_Primates], axis = 0)#连接参数产生的字符串
print (df0_CoVs_all.shape)

id_list = df0_CoVs_all['id'].tolist()
df0_CoVs_all['id'] = df0_CoVs_all['id'].astype('category')#改变数据类型？
df0_CoVs_all['id'].cat.reorder_categories(id_list, inplace=True)#？？？
df0_CoVs_all.sort_values('id', inplace=True)#将数据集依照某个字段的数据排序，上三行在对accession排序，确保后面加adaptation列时顺序一致


df_CoVs_ORF = pd.read_csv ('../data/df_full_DCR_counting_ORF1ab.csv')
print (df_CoVs_ORF.shape)
df0_CoVs_ORF_training = df_CoVs_ORF [df_CoVs_ORF['accession'].isin(id_list)]#筛选accession列中值在id_list中的行
print (df0_CoVs_ORF_training.shape)

df0_CoVs_ORF_training['accession'] = df0_CoVs_ORF_training['accession'].astype('category')
df0_CoVs_ORF_training['accession'].cat.reorder_categories(id_list, inplace=True)
df0_CoVs_ORF_training.sort_values('accession', inplace=True)
print (df0_CoVs_ORF_training.shape)
df0_CoVs_ORF_training['adaptation'] = df0_CoVs_all['adaptation'].tolist()#加一列adaptation
print (df0_CoVs_ORF_training.shape)
df0_CoVs_ORF_training = shuffle(df0_CoVs_ORF_training)
df0_CoVs_ORF_training.to_csv ('shuffle_save_virus.csv')
df0_CoVs_ORF_training = pd.read_csv ('shuffle_save_virus.csv')
X = df0_CoVs_ORF_training.iloc[:, 2:-4]#全部组成性信息（去掉后面的密码子和标签列）
y = df0_CoVs_ORF_training.iloc[:, -1]#adaptation列，即0，1，2标签
print (X.shape, y.shape)
print (df0_CoVs_ORF_training['adaptation'].value_counts())

 ########################################################    prepair data
 ########################################################    prepair data


 ########################################## data split and SMOTE resampling
 ########################################## data split and SMOTE resampling

train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.01, random_state=4)#X：要划分的样本特征集，y：要划分的样本结果，testsize：样本占比，种子
print (train_x.shape, valid_x.shape)
print (train_x.head(2))
smo = SMOTE(random_state = 4)#随机选取少数类样本用以合成新样本？
SMOTE_train_x, SMOTE_train_y = smo.fit_resample(train_x, train_y)

# print(type(SMOTE_train_x))
# print(SMOTE_train_y.head())
SMOTE_train_x.to_csv ('SMOTE_train_x_1.csv')
SMOTE_train_y.to_csv ('SMOTE_train_y_1.csv')
# df0_smo = pd.concat([SMOTE_train_x,SMOTE_train_y],ignore_index=True)
# df0_smo.to_csv ('df_smo.csv')

