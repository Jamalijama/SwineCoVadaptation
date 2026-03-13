# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 18:23:18 2021

@author: Jing Li, Small steps make changes. dnt_seq@163.com
"""

from sklearn.manifold import TSNE
# from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import seaborn as sns

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
nt_category = ['n1', 'n2','n3']
dnt_category = ['n12', 'n23','n31']
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
nts_cols_list = ['Freq_'+ nt + '_' + nt_cat for nt_cat in nt_category for nt in nt_table]
dnts_cols_list = ['Freq_'+ dnt + '_' + dnt_cat for dnt_cat in dnt_category for dnt in dnt_table]
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]
codon_cols_list = ['Freq_'+ codon for codon in codon_table]
codonpair_cols_list = ['Freq_'+ codonpair for codonpair in codonpair_table]
amino_cols_list = ['Freq_'+ amino for amino in amino_table]
full_cols_list0 = nts_cols_list + dnts_cols_list + dntpair_cols_list + codon_cols_list + codonpair_cols_list + amino_cols_list
# full_cols_list1 = [col[5:] for col in full_cols_list0]
# gene_list = ["CDS_1", "CDS_2", "CDS_3", "CDS_4", "CDS_5", "CDS_6", "CDS_7", "CDS_8"]
dcr_set_name_list = ['nts','dnts', 'DCR', 'codons','codonpair', 'aminos']
dcr_set_list = [nts_cols_list, dnts_cols_list, dntpair_cols_list, codon_cols_list, codonpair_cols_list, amino_cols_list]

# color_ = [ '#00FFFF', '#DC143C', '#A52A2A', '#DEB887', '#8FBC8F', '#DC143C', '#8B0000', '#008080']
# dict_c = dict(zip(['Artiodactyla', 'Chiroptera', 'Suiformes', 'Carnivora', 'Primates', 'Avian'],color_[:6]))
color_ = [ '#00FFFF', '#00008B', '#FF00FF', '#DEB887', '#DC143C', '#006400', '#00CED1', '#008080','#FF1493','#7FFF00', '#D2691E', '#000000']
# dict_c = dict(zip(['Chiroptera','Artiodactyla','Rodent_Lagomorpha', 'Carnivora', 'Primates', 'Suiformes', 'Galliformes_avians', 'Lagomorpha', 'Perissodactyla', 'Unclassified', 'Soricomorpha ', 'Erinaceomorpha'],color_[:12]))
dict_c = dict(zip(['CHI','ART','ROD_LAG', 'CAR', 'PRI', 'SADS', 'PEDV', 'PHEV', 'TGEV','PDCoV'],color_[:9]))

label_list = [0,1,2,3,4,5,6,7,8,9]

# color_label_list = [dict_c[label] for label in label_list]

path = './' 
path1 = '../Counting/' 
# target_host_list0 = ['##', 'Suiformes', 'Chiroptera','Soricomorpha_Erinaceomorpha',
#                      'Primates','Carnivora','Rodent_Lagomorpha', 'Artiodactyla',
#                       'Galliformes_avians']
# target_host_list2 = [-1,5, 0, -1, 4, 3, 2, 1, -1]
# dict_host = dict(zip(target_host_list0, target_host_list2))
# # print (dict_host)

# target_host_list1 = ['Chiroptera','Artiodactyla','Rodent_Lagomorpha', 'Carnivora', 'Primates', 'Suiformes']
target_host_list1 = ['CHI','ART','ROD_LAG', 'CAR', 'PRI', 'SADS-CoV', 'PEDV', 'PHEV', 'TGEV','PDCoV']

target_host_list3 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dict_host2 = dict(zip(target_host_list3, target_host_list1))

# print (dict_host)

# for file in os.listdir(path1):
#         if file.endswith('host_2.csv') & file.startswith('df_full_DCR_counting_CoVs_(SARS2_added)_CDS'):
#             print (file)
#             orf_name = file[-17:-11]
#             print (orf_name)
#             df0_CoVs = pd.read_csv (path1 + file)
#             print (df0_CoVs.shape)
#             df0_CoVs = df0_CoVs[df0_CoVs['host2']!='Galliformes_avians']
#             print (df0_CoVs.shape)
#
#             host2_list = df0_CoVs.loc[:,'host2'].tolist()
#             # print (list(set(host2_list)))
#             host_list = [dict_host[i] for i in host2_list]
#             df0_CoVs['host'] = host_list
#             df0_CoVs_all = df0_CoVs [df0_CoVs['host']>=0]
#             id_list = df0_CoVs_all['id'].tolist()
#             host_list_check = df0_CoVs_all.loc[:,'host'].tolist()
#             # print (list(set(host_list_check)))
#
#             df0_CoVs_all['id'] = df0_CoVs_all['id'].astype('category')
#             df0_CoVs_all['id'].cat.reorder_categories(id_list, inplace=True)
#             df0_CoVs_all.sort_values('id', inplace=True)
#             df_count = df0_CoVs_all.loc[:, full_cols_list0]
#             y = df0_CoVs_all.loc[:, 'host']
#             print (df_count.shape, y.shape)
for set_i in range(6):
    set_name = dcr_set_name_list[set_i]
    dcr_set = dcr_set_list[set_i]
    # X = df_count[dcr_set]
    # print(X.shape)


                # for split_i in range(1,2,1):
                #     train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=0.1, random_state = split_i)
                #     print (train_x.shape, valid_x.shape)
                #     for smote_i in range (1,2,1):
                #         smo = SMOTE(random_state = smote_i)
                #         SMOTE_train_x, SMOTE_train_y = smo.fit_resample(train_x, train_y)
                #         SMOTE_test_x, SMOTE_test_y = smo.fit_resample(valid_x, valid_y)
                #
    df0_CoVs_ORF_training_sui = pd.read_csv ('Hier_counting_post_sampling_ORF1ab.csv')
    id = df0_CoVs_ORF_training_sui.iloc[:, 1]
    SMOTE_test_x_0 = df0_CoVs_ORF_training_sui.iloc[:, 2:-1]#全部组成性信息（去掉后面的密码子和标签列）
    SMOTE_test_y = df0_CoVs_ORF_training_sui.iloc[:, -1]#adaptation列，即0，1，2标签
    SMOTE_test_x = SMOTE_test_x_0[dcr_set]

    index_i_list = [i for i in id]
    index_i_list_0 = [str(i) for i in range(SMOTE_test_x.shape[0])]
    index_h_list = [dict_host2[j] for j in SMOTE_test_y]
    index_list = []
    for k in range(len(index_i_list_0)):
        index_list.append(index_h_list[k] + '_' + index_i_list[k])
    SMOTE_test_x['index'] = index_list
    SMOTE_test_x = SMOTE_test_x.set_index(['index'])
    print ( SMOTE_test_x.shape)

    name_train = 'ORF1ab' + '_splitRand' + '_smoteRand' + '_sui_virus_' + set_name

    sns.clustermap(SMOTE_test_x, method ='ward', metric='euclidean',figsize=(10,22))
    plt.savefig('Hierarchical_cluster_PDCoV' + '_' + name_train + '.png',dpi = 1200)
