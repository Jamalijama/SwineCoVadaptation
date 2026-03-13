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



df0_CoVs2 = pd.read_csv ('../data/df_Coronaviridae_host_labels.csv')

df0_CoVs_SADS = df0_CoVs2 [(df0_CoVs2['host']=='SADS')].sample (frac = 0.3171, random_state = 10)#41
print (df0_CoVs_SADS.shape)
df0_CoVs_SADS['adaptation'] = [5]*df0_CoVs_SADS.shape[0]
print (df0_CoVs_SADS.shape)

df0_CoVs_PEDV = df0_CoVs2 [(df0_CoVs2['host']=='PEDV')].sample (frac = 0.0245, random_state = 10)#530
print (df0_CoVs_PEDV.shape)
df0_CoVs_PEDV['adaptation'] = [6]*df0_CoVs_PEDV.shape[0]
print (df0_CoVs_PEDV.shape)


df0_CoVs_PHEV = df0_CoVs2 [(df0_CoVs2['host']=='PHEV')]#13
print (df0_CoVs_PHEV.shape)
df0_CoVs_PHEV['adaptation'] = [7]*df0_CoVs_PHEV.shape[0]
print (df0_CoVs_PHEV.shape)


df0_CoVs_TGEV = df0_CoVs2 [(df0_CoVs2['host']=='TGEV')].sample (frac = 0.2241, random_state = 10)#58
print (df0_CoVs_TGEV.shape)
df0_CoVs_TGEV['adaptation'] = [8]*df0_CoVs_TGEV.shape[0]
print (df0_CoVs_TGEV.shape)


df0_CoVs_PDCoV = df0_CoVs2 [(df0_CoVs2['host']=='PDCoV')].sample (frac = 0.3250, random_state = 10)#40
print (df0_CoVs_PDCoV.shape)
df0_CoVs_PDCoV['adaptation'] = [9]*df0_CoVs_PDCoV.shape[0]
print (df0_CoVs_PDCoV.shape)

df0_CoVs_sui = pd.concat ([df0_CoVs_SADS, df0_CoVs_PEDV, df0_CoVs_PHEV, df0_CoVs_TGEV, df0_CoVs_PDCoV], axis = 0)#连接参数产生的字符串
print (df0_CoVs_sui.shape)


id_list = df0_CoVs_sui['id'].tolist()
df0_CoVs_sui['id'] = df0_CoVs_sui['id'].astype('category')#改变数据类型？
df0_CoVs_sui['id'].cat.reorder_categories(id_list, inplace=True)#？？？
df0_CoVs_sui.sort_values('id', inplace=True)#将数据集依照某个字段的数据排序，上三行在对accession排序，确保后面加adaptation列时顺序一致


df_CoVs_ORF_sui = pd.read_csv ('../data/df_full_DCR_counting_Spike.csv')
print (df_CoVs_ORF_sui.shape)
df0_CoVs_ORF_training_sui = df_CoVs_ORF_sui [df_CoVs_ORF_sui['accession'].isin(id_list)]#筛选accession列中值在id_list中的行
print (df0_CoVs_ORF_training_sui.shape)

df0_CoVs_ORF_training_sui['accession'] = df0_CoVs_ORF_training_sui['accession'].astype('category')
df0_CoVs_ORF_training_sui['accession'].cat.reorder_categories(id_list, inplace=True)
df0_CoVs_ORF_training_sui.sort_values('accession', inplace=True)
print (df0_CoVs_ORF_training_sui.shape)
df0_CoVs_ORF_training_sui['adaptation'] = df0_CoVs_sui['adaptation'].tolist()#加一列adaptation
print (df0_CoVs_ORF_training_sui.shape)
df0_CoVs_ORF_training_sui = shuffle(df0_CoVs_ORF_training_sui)
df0_CoVs_ORF_training_sui.to_csv ('shuffle_save_sui_PDCoV_spike.csv')
df0_CoVs_ORF_training_sui = pd.read_csv ('shuffle_save_sui_PDCoV_spike.csv')
X_sui = df0_CoVs_ORF_training_sui.iloc[:, 2:-4]#全部组成性信息（去掉后面的密码子和标签列）
y_sui = df0_CoVs_ORF_training_sui.iloc[:, -1]#adaptation列，即0，1，2标签
print (X_sui.shape, y_sui.shape)
print (df0_CoVs_ORF_training_sui['adaptation'].value_counts())