# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:06:06 2021

@author: Jing Li, Small steps make changes. dnt_seq@163.com
"""
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os

nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]
path = '../counting/'
path1 = '../data/'
model_name = './DCR_CNN_model_ORF1ab.txt'
# file_labels = 'df0_IAVs_8segs_unique_concat_intersection_NCBI,GISAID,IRD.csv'

class CNN (nn.Module):
    def __init__ (self):
        super (CNN, self).__init__()
        self.conv1 = nn.Sequential ( 
            nn.Conv3d ( in_channels = 1,
                        out_channels = 8,
                        kernel_size = (1, 3, 3), # kernel only for 2d data
                        stride =(1,1,1),
                        padding = (0,1,1),
                        bias = True
                        ),            # 
            nn.ReLU (),
            nn.AvgPool3d (kernel_size = (1,2,2)) 
        )
        self.conv2 = nn.Sequential ( 
            nn.Conv3d ( in_channels = 8,
                        out_channels = 16,
                        kernel_size = (1, 3, 3),# kernel only for 2d data
                        stride =(1,1,1),
                        padding = (0,1,1),
                        bias = True
                        ),            # 
            nn.ReLU (),
            nn.AvgPool3d (kernel_size = (1, 2, 2)) # Max or Avg
        )
        self.conv3 = nn.Sequential (
            nn.Conv3d ( in_channels = 16,
                        out_channels = 32,
                        kernel_size = (1, 3, 3),# kernel only for 2d data
                        stride =(1,1,1),
                        bias = True,
                        padding = (0,1,1)
                        ),
            nn.ReLU (),
            nn.AvgPool3d (kernel_size = (1, 2, 2)) #MaxPool3d
        )
        self.fc1 = nn.Linear (768, 192)  # adding this step, too slow
        # self.fc2 = nn.Linear (192, 48)  # adding this step, too slow
        self.fc3 = nn.Linear (192, 5)
    def forward (self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view (x.size(0), -1) # flat x, similar to reshape of numpy
        fc_full = x
        x = F.sigmoid (self.fc1(x))  # to activate x
        prob_ = F.softmax (self.fc3(x))
        pred_ = self.fc3(x)
        return prob_, pred_, fc_full

cnn = CNN()
if_use_gpu = 1
if if_use_gpu:
    cnn = cnn.cuda()

# df_labels = pd.read_csv (path1 + file_labels)
# print (df_labels.shape)

for file in os.listdir(path):
    if file.startswith ('df_full_DCR_counting_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_CDS_ORF1ab_host_2_SARS2.csv')&file.endswith('.csv'):
        print (file)
        df_DCR_counting = pd.read_csv (path + file)
        print (df_DCR_counting.columns)
        print (df_DCR_counting.shape)
        accession_list = df_DCR_counting.loc[:,'accession'].tolist()
        df_DCR = df_DCR_counting.loc[:,dntpair_cols_list]
        DCR_array = np.array(df_DCR)
        _num = DCR_array.shape[0]
        DCR_array2 = DCR_array.reshape(_num,1,6,16,16)
        print (DCR_array2.shape)
        DCR_tensor = torch.tensor(DCR_array2)
        DCR_tensor = DCR_tensor.to(torch.float32)

        cnn = torch.load(model_name)
        _pred = cnn (Variable(DCR_tensor))
        _pred_array = _pred[0].data.numpy()
        _prob_array = _pred[1].data.numpy()
        print (_pred_array.shape)

        res = pd.DataFrame ({'Strain_name': accession_list}) # _labels
        print (res.shape)
        res [['Predict_0','Predict_1', 'Predict_2','Predict_3','Predict_4']] = _pred_array
        res [['Score_0','Score_1','Score_2','Score_3','Score_4']] = _prob_array
        # res [df_labels.columns.tolist()[-7:]] = df_labels.iloc[:,-7:]
        print (res.shape)
        file_name = 'df_predicted_Swine_ORF1ab_' + model_name[2:-4] + '.csv'
        res.to_csv (file_name, index = False)