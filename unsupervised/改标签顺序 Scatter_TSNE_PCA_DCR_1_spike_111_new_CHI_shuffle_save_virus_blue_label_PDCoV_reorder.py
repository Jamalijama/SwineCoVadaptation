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

dcr_set_name_list = ['nt', 'nt','dnts', 'codons', 'aminos', 'DCR','codonpair']
dcr_set_list = [nts_cols_list, nts_cols_list,dnts_cols_list,codon_cols_list, amino_cols_list,  dntpair_cols_list, codonpair_cols_list]

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
'blue1':                '#1663A9',
'purple':               '#800080',
'darkgreen':            '#006400',
'darkyellow':           '#996600',
'gray':                 '#666666',
# 'darkred':              '#8B0000',
'deeppink':             '#FF1493',
'green':                '#66CC00',
'deepskyblue':          '#00BFFF',
'orange1':              '#FF6600',
'orchid':               '#DA70D6',
'yellow':               '#FFCC33',
'blue2':                '#6EB1DE',
'lightgreen':           '#90EE90',
# 'blue3':                '#1383C2',
# 'blue4':                '#20C2L1',
# 'deepskyblue':          '#00BFFF',
'darkgreen':            '#006400',
# 'lightblue':            '#ADD8E6',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32',
'burlywood':            '#DEB887',
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
# df0_CoVs = pd.read_csv ('../data/df_full_CDS_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_1_SARS2_virus.csv')
# df0_CoVs1 = pd.read_csv ('../data/df_PCA_SDCR.csv')
#
#
# df0_CoVs_Chiroptera0 = df0_CoVs1 [(df0_CoVs1 ['host']=='Chiroptera')].sample (frac = 0.45, random_state = 10)
# print (df0_CoVs_Chiroptera0.shape)
# df0_CoVs_Chiroptera0['adaptation'] = [0]*df0_CoVs_Chiroptera0.shape[0]
# print (df0_CoVs_Chiroptera0.shape)
#
# df0_CoVs_Chiroptera1 = df0_CoVs1 [(df0_CoVs1 ['host']=='Chiroptera_1')].sample (frac = 0.26, random_state = 10)
# print (df0_CoVs_Chiroptera1.shape)
# df0_CoVs_Chiroptera1['adaptation'] = [0]*df0_CoVs_Chiroptera1.shape[0]
# print (df0_CoVs_Chiroptera1.shape)
#
#
# df0_CoVs_Chiroptera=pd.concat ([df0_CoVs_Chiroptera0,df0_CoVs_Chiroptera1],axis = 0)
#
#
#
# df0_CoVs_Artiodactyla = df0_CoVs [(df0_CoVs ['host']=='Artiodactyla')].sample (frac = 0.17, random_state = 10)
# print (df0_CoVs_Artiodactyla.shape)
# df0_CoVs_Artiodactyla['adaptation'] = [1]*df0_CoVs_Artiodactyla.shape[0]
# print (df0_CoVs_Artiodactyla.shape)
#
# df0_CoVs_Rodent_Lagomorpha = df0_CoVs [(df0_CoVs ['host']=='Rodent_Lagomorpha')]
# print (df0_CoVs_Rodent_Lagomorpha.shape)
# df0_CoVs_Rodent_Lagomorpha['adaptation'] = [2]*df0_CoVs_Rodent_Lagomorpha.shape[0]
# print (df0_CoVs_Rodent_Lagomorpha.shape)
#
# df0_CoVs_Carnivora = df0_CoVs [(df0_CoVs ['host']=='Carnivora')]
# print (df0_CoVs_Carnivora.shape)
# df0_CoVs_Carnivora['adaptation'] = [3]*df0_CoVs_Carnivora.shape[0]
# print (df0_CoVs_Carnivora.shape)
#
# df0_CoVs_Primates = df0_CoVs [(df0_CoVs ['host']=='Primates')].sample (frac = 0.17, random_state = 10)
# print (df0_CoVs_Primates.shape)
# df0_CoVs_Primates['adaptation'] = [4]*df0_CoVs_Primates.shape[0]
# print (df0_CoVs_Primates.shape)
#
#
# df0_CoVs_all = pd.concat ([df0_CoVs_Chiroptera, df0_CoVs_Artiodactyla, df0_CoVs_Rodent_Lagomorpha, df0_CoVs_Carnivora, df0_CoVs_Primates], axis = 0)#连接参数产生的字符串
# print (df0_CoVs_all.shape)
#
# id_list = df0_CoVs_all['id'].tolist()
# df0_CoVs_all['id'] = df0_CoVs_all['id'].astype('category')#改变数据类型？
# df0_CoVs_all['id'].cat.reorder_categories(id_list, inplace=True)#？？？
# df0_CoVs_all.sort_values('id', inplace=True)#将数据集依照某个字段的数据排序，上三行在对accession排序，确保后面加adaptation列时顺序一致
#
#
# df_CoVs_ORF = pd.read_csv ('../data/df_full_DCR_counting_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_CDS_ORF1ab_host_2_SARS2.csv')
# print (df_CoVs_ORF.shape)
# df0_CoVs_ORF_training = df_CoVs_ORF [df_CoVs_ORF['accession'].isin(id_list)]#筛选accession列中值在id_list中的行
# print (df0_CoVs_ORF_training.shape)
#
# df0_CoVs_ORF_training['accession'] = df0_CoVs_ORF_training['accession'].astype('category')
# df0_CoVs_ORF_training['accession'].cat.reorder_categories(id_list, inplace=True)
# df0_CoVs_ORF_training.sort_values('accession', inplace=True)
# print (df0_CoVs_ORF_training.shape)
# df0_CoVs_ORF_training['adaptation'] = df0_CoVs_all['adaptation'].tolist()#加一列adaptation
# print (df0_CoVs_ORF_training.shape)
# df0_CoVs_ORF_training = shuffle(df0_CoVs_ORF_training)
# df0_CoVs_ORF_training.to_csv ('shuffle_save_virus.csv')
# df0_CoVs_ORF_training = pd.read_csv ('shuffle_save_virus.csv')
# X = df0_CoVs_ORF_training.iloc[:, 2:-4]#全部组成性信息（去掉后面的密码子和标签列）
# y = df0_CoVs_ORF_training.iloc[:, -1]#adaptation列，即0，1，2标签
# print (X.shape, y.shape)
# print (df0_CoVs_ORF_training['adaptation'].value_counts())
#
#  ########################################################    prepair data
#  ########################################################    prepair data
#
#
#  ########################################## data split and SMOTE resampling
#  ########################################## data split and SMOTE resampling
#
# train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.01, random_state=4)#X：要划分的样本特征集，y：要划分的样本结果，testsize：样本占比，种子
# print (train_x.shape, valid_x.shape)
# print (train_x.head(2))
# smo = SMOTE(random_state = 4)#随机选取少数类样本用以合成新样本？
# SMOTE_train_x, SMOTE_train_y = smo.fit_resample(train_x, train_y)
#
# # print(type(SMOTE_train_x))
# # print(SMOTE_train_y.head())
# SMOTE_train_x.to_csv ('SMOTE_train_x_1.csv')
# SMOTE_train_y.to_csv ('SMOTE_train_y_1.csv')
# # df0_smo = pd.concat([SMOTE_train_x,SMOTE_train_y],ignore_index=True)
# # df0_smo.to_csv ('df_smo.csv')
#
# # df0_CoVs2 = pd.read_csv ('../data/df_full_CDS_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_1_SARS2_virus.csv')
# #
# #
# #
# df0_CoVs_SADS = df0_CoVs2 [(df0_CoVs2['host']=='SADS')].sample (frac = 0.36, random_state = 10)
# print (df0_CoVs_SADS.shape)
# df0_CoVs_SADS['adaptation'] = [5]*df0_CoVs_SADS.shape[0]
# print (df0_CoVs_SADS.shape)
#
# df0_CoVs_PEDV = df0_CoVs2 [(df0_CoVs2['host']=='PEDV')].sample (frac = 0.028, random_state = 10)
# print (df0_CoVs_PEDV.shape)
# df0_CoVs_PEDV['adaptation'] = [6]*df0_CoVs_PEDV.shape[0]
# print (df0_CoVs_PEDV.shape)
#
#
# df0_CoVs_PHEV = df0_CoVs2 [(df0_CoVs2['host']=='PHEV')]
# print (df0_CoVs_PHEV.shape)
# df0_CoVs_PHEV['adaptation'] = [7]*df0_CoVs_PHEV.shape[0]
# print (df0_CoVs_PHEV.shape)
#
#
# df0_CoVs_TGEV = df0_CoVs2 [(df0_CoVs2['host']=='TGEV')]
# print (df0_CoVs_TGEV.shape)
# df0_CoVs_TGEV['adaptation'] = [8]*df0_CoVs_TGEV.shape[0]
# print (df0_CoVs_TGEV.shape)
#
#
#
# df0_CoVs_sui = pd.concat ([df0_CoVs_SADS, df0_CoVs_PEDV, df0_CoVs_PHEV, df0_CoVs_TGEV], axis = 0)#连接参数产生的字符串
# print (df0_CoVs_sui.shape)
#
#
#
#
# id_list = df0_CoVs_sui['id'].tolist()
# df0_CoVs_sui['id'] = df0_CoVs_sui['id'].astype('category')#改变数据类型？
# df0_CoVs_sui['id'].cat.reorder_categories(id_list, inplace=True)#？？？
# df0_CoVs_sui.sort_values('id', inplace=True)#将数据集依照某个字段的数据排序，上三行在对accession排序，确保后面加adaptation列时顺序一致
#
#
# df_CoVs_ORF_sui = pd.read_csv ('../data/df_full_DCR_counting_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_CDS_ORF1ab_host_2_SARS2.csv')
# print (df_CoVs_ORF_sui.shape)
# df0_CoVs_ORF_training_sui = df_CoVs_ORF_sui [df_CoVs_ORF_sui['accession'].isin(id_list)]#筛选accession列中值在id_list中的行
# print (df0_CoVs_ORF_training_sui.shape)
#
# df0_CoVs_ORF_training_sui['accession'] = df0_CoVs_ORF_training_sui['accession'].astype('category')
# df0_CoVs_ORF_training_sui['accession'].cat.reorder_categories(id_list, inplace=True)
# df0_CoVs_ORF_training_sui.sort_values('accession', inplace=True)
# print (df0_CoVs_ORF_training_sui.shape)
# df0_CoVs_ORF_training_sui['adaptation'] = df0_CoVs_sui['adaptation'].tolist()#加一列adaptation
# print (df0_CoVs_ORF_training_sui.shape)
# # df0_CoVs_ORF_training_sui = shuffle(df0_CoVs_ORF_training)
# # df0_CoVs_ORF_training.to_csv ('shuffle_save_virus.csv')
# # df0_CoVs_ORF_training = pd.read_csv ('shuffle_save_virus.csv')
# X_sui = df0_CoVs_ORF_training_sui.iloc[:, 1:-4]#全部组成性信息（去掉后面的密码子和标签列）
# y_sui = df0_CoVs_ORF_training_sui.iloc[:, -1]#adaptation列，即0，1，2标签
# print (X_sui.shape, y_sui.shape)
# print (df0_CoVs_ORF_training_sui['adaptation'].value_counts())
#
#
#
#
#
# # SMOTE_test_x, SMOTE_test_y = smo.fit_resample(valid_x, valid_y)
#
# print ('SMOTE',SMOTE_train_x.shape)
# id_list_smote = SMOTE_train_x ['id'].tolist()
# #SMOTE_train_x.to_csv ('df_full_DCR_counting_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_CDS_ORF1ab_host_1_smote.csv')





df0_CoVs_ORF_training_sui = pd.read_csv ('data_TSNE_PCA_post_sampling_Spike_reorder.csv')
SMOTE_train_x = df0_CoVs_ORF_training_sui.iloc[:, 2:-1]#全部组成性信息（去掉后面的密码子和标签列）
SMOTE_train_y = df0_CoVs_ORF_training_sui.iloc[:, -1]#adaptation列，即0，1，2标签
id_list_smote = df0_CoVs_ORF_training_sui.iloc[:, 1]
# print(SMOTE_train_x.head())
# print(SMOTE_train_y.head())
# print(id_list_smote.head())
lst = []
for i in SMOTE_train_y:
    if i == 0:
        x = 'CHI'
    elif i == 1:
        x = 'PRI'
    elif i == 2:
        x = 'ROD_LAG'
    elif i == 3:
        x =  'CAR'
    elif i == 4:
        x = 'ART'
    elif i == 5:
        x = 'SADS'
    elif i == 6:
        x = 'PEDV'
    elif i == 7:
        x = 'PHEV'
    elif i == 8:
        x = 'TGEV'
    else:
        x = 'PDCoV'
    lst.append(x)

SMOTE_train_x['host1']=lst
#for gene in gene_list:
#print (gene)
#df_DCR1 = df_DCR0 [df_DCR0['Gene']==gene]
y_types =list (set(SMOTE_train_x['host1'].tolist()))
y_types = ['CHI','PRI','ROD_LAG','CAR','ART','SADS','PEDV','PHEV','TGEV','PDCoV']
#df_DCR=df_DCR0 [(df_DCR0 ['host2']=='Galliformes_avians')|\
#                            (df_DCR0 ['host2']=='Chiroptera')|\
#                            (df_DCR0 ['host2']=='Rodent_Lagomorpha')|\
#                            (df_DCR0 ['host2']=='Artiodactyla')|\
#                            (df_DCR0['host2']=='Carnivora')|\
#                            (df_DCR0 ['host2']=='Primates')]

print(y_types)
#y_types = ['GAL_AVI', 'CHI', 'ROD_LAG','ART', 'CAR', 'PRI']
#print (y_types)
y_num = len(y_types)
print(y_num)


label_list2 = SMOTE_train_x['host1'].tolist()
label_cate_list = list(set(label_list2))
print((label_cate_list))
print(len(label_cate_list))
label_c_list = list(range(1,11,1))

dict1 = dict(zip(label_cate_list,label_c_list))
dict2 = dict(zip(label_cate_list,color_dict))
label_list1 = [dict1[i] for i in label_list2]
color_list = [color_dict[i] for i in label_list1]

for set_i in range(7):
        set_name = dcr_set_name_list[set_i]
        dcr_set = dcr_set_list[set_i]
        data = np.array (SMOTE_train_x[dcr_set])
        print (data.shape)

        X_tsne = TSNE(learning_rate=100).fit_transform(data)
        X_pca = PCA(n_components = 2).fit_transform(data)

        df_tsne = pd.DataFrame (X_tsne,columns = ['t_SNE1','t_SNE2'])
        df_tsne = (df_tsne - df_tsne.min()) / (df_tsne.max() - df_tsne.min())
        df_tsne ['label'] = SMOTE_train_x['host1'].tolist()
        df_tsne ['id'] =id_list_smote
#        df_tsne.to_csv ('df_tSNE_S' + set_name + '.csv')
        # print (df_pca.head(2))

        df_pca = pd.DataFrame (X_pca,columns = ['PCA1','PCA2'])
        df_pca = (df_pca - df_pca.min()) / (df_pca.max() - df_pca.min())

        df_pca ['label'] = SMOTE_train_x['host1'].tolist()
        df_pca ['id'] =id_list_smote
        df_pca.to_csv ('df_PCA_ORF1ab' + set_name + '.csv')

        plt.figure(figsize=(8, 3))
        plt.subplot(121)
#        for y_i in range(y_num):
#            y_ = y_types[y_i]
#            print (y_i)
#            color = color_list0[y_i]
##            plt.xlim([-0.1,1.1])
##            plt.ylim([-0.1,1.1])
#
#            df_X_tsne_label = df_tsne[df_tsne['label'] == y_]
#            print (df_X_tsne_label.shape)
#            sns.scatterplot(x = df_X_tsne_label['t_SNE2'], y = df_X_tsne_label['t_SNE1'], color = color) # ,x_estimator=np.mean
#            sns.set( font_scale=0.2)
#            sns.set_style("white")
        sns.scatterplot(data = df_tsne, x = 't_SNE2', y = 't_SNE1', hue = 'label', palette = color_list0[:y_num],hue_order = y_types) #

        plt.legend(scatterpoints=1)
        plt.subplot(122)

        for y_i in range(y_num):
            y_ = y_types[y_i]
            print (y_i)
            color = color_list0[y_i]
#            plt.xlim([-0.1,1.1])
#            plt.ylim([-0.1,1.1])

            df_X_pca_label = df_pca[df_pca['label'] == y_]
            print (df_X_pca_label.shape)
            sns.scatterplot(x = df_X_pca_label['PCA2'], y = df_X_pca_label['PCA1'], color = color) # ,x_estimator=np.mean
            sns.set( font_scale=0.2)
            sns.set_style("white")
#             plt.legend(scatterpoints=1)
        plt.savefig('sns_scatterplot_tSNE_PCA_Spike_' + set_name + '_sampling200_test_sui_virus_blue_label_PDCoV_reorder.png', dpi = 300, bbox_inches = 'tight')
