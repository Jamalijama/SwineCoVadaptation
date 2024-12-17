# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 17:06:06 2021

@author: Jing Li, Small steps make changes. dnt_seq@163.com
"""
import pandas as pd
import numpy as np
import torch
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.utils import shuffle

nt_table = ['t','c', 'a', 'g']
dnt_table = [nt1+nt2 for nt1 in nt_table for nt2 in nt_table]
dnts_table = [dnt1+dnt2 for dnt1 in dnt_table for dnt2 in dnt_table]
dntpair_category = ['n12m12', 'n23m23','n31m31','n12n31','n23m12','n31m23']
dntpair_cols_list = ['Freq_'+ dnts + '_' + dnts_cat for dnts_cat in dntpair_category for dnts in dnts_table]

#path = './' 
#path1 = '../Counting/' 
#target_host_list0 = ['##', 'Suiformes', 'Chiroptera','Soricomorpha_Erinaceomorpha',
#                     'Primates','Carnivora','Rodent_Lagomorpha', 'Artiodactyla', 
#                      'Galliformes_avians']
#target_host_list = [-1, 1, -1, -1, 4, 3, 2, 0, -1]
#dict_host = dict(zip(target_host_list0,target_host_list))
# print (dict_host)
y_dict = {0: 'CHI', 1:'ART', 2:'ROD', 3:'CAR',4: 'PRI'}
host_list = list(y_dict.values())
# split_size_list = [0.2, 0.25, 0.3]
epoch_num_list =  [10, 20, 30, 40, 50]
split_size_list = [0.2]
split_i = 1
smote_i = 5

font1 = {'family' : 'arial',  
        'color'  : 'darkblue',  
        'weight' : 'normal',  
        'size'   : 12,  
        } 
font2 = {'family' : 'arial',  
        'color'  : 'black',  
        'weight' : 'normal',  
        'size'   : 12,  
        }  
font3 = {'family' : 'arial',  
        'weight' : 'normal',  
        'size'   : 18,  
        }  


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Confusion matrix (%)'
        else:
            title = 'Confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]#np.newaxis:加维度
        print("Confusion matrix (%)")
    fig, ax = plt.subplots(dpi=300)#,figsize = (10,10)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels = ['CHI', 'ART', 'ROD', 'CAR', 'PRI'],
           yticklabels = ['CHI', 'ART', 'ROD', 'CAR', 'PRI'],
           title=title)

    ax.set_title(title, fontsize=16, color='darkblue')
    ax.set_ylabel('True label', fontdict=font1, rotation=90, loc='center') 
    ax.set_xlabel('Predicted label', fontdict=font1, loc='center') 
    ax.set_yticklabels(['CHI', 'ART', 'ROD', 'CAR', 'PRI'], fontdict=font1, rotation=90) 
    ax.set_xticklabels(['CHI', 'ART', 'ROD', 'CAR', 'PRI'], fontdict=font1) 

    plt.setp(ax.get_xticklabels(), rotation=0, ha="right",
             rotation_mode="anchor")
    ax.figure.colorbar(im, ax=ax)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    fontdict=font3,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

df0_CoVs = pd.read_csv ('../data/df_full_CDS_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_1_SARS2.csv')
df0_CoVs1 = pd.read_csv ('../data/df_PCA_SDCR.csv')

#df0_CoVs_Galliformes_avians = df0_CoVs [(df0_CoVs ['host']=='Galliformes_avians')].sample (frac = 0.33, random_state = 10)
#print (df0_CoVs_Galliformes_avians.shape)
#df0_CoVs_Galliformes_avians['adaptation'] = [0]*df0_CoVs_Galliformes_avians.shape[0]#添加标签
#print (df0_CoVs_Galliformes_avians.shape)
 
df0_CoVs_Chiroptera0 = df0_CoVs1 [(df0_CoVs1 ['host']=='Chiroptera')].sample (frac = 0.45, random_state = 10)
print (df0_CoVs_Chiroptera0.shape)
df0_CoVs_Chiroptera0['adaptation'] = [0]*df0_CoVs_Chiroptera0.shape[0]
print (df0_CoVs_Chiroptera0.shape)
 
df0_CoVs_Chiroptera1 = df0_CoVs1 [(df0_CoVs1 ['host']=='Chiroptera_1')].sample (frac = 0.26, random_state = 10)
print (df0_CoVs_Chiroptera1.shape)
df0_CoVs_Chiroptera1['adaptation'] = [0]*df0_CoVs_Chiroptera1.shape[0]
print (df0_CoVs_Chiroptera1.shape)

df0_CoVs_Chiroptera=pd.concat ([df0_CoVs_Chiroptera0,df0_CoVs_Chiroptera1],axis = 0)

# df0_CoVs_Suiformes = df0_CoVs [(df0_CoVs ['host']=='Suiformes')].sample (frac = 0.17, random_state = 10)
# print (df0_CoVs_Suiformes.shape)
# df0_CoVs_Suiformes['adaptation'] = [-1]*df0_CoVs_Suiformes.shape[0]
# print (df0_CoVs_Suiformes.shape)

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


df_CoVs_ORF = pd.read_csv ('../data/df_full_DCR_counting_S_ORF1ab_Coronaviridae(Organism)_26-32k_2564_CDS_ORF1ab_host_2_SARS2.csv')
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
#df0_CoVs_ORF_training.to_csv ('xxxxxxx.csv')
X = df0_CoVs_ORF_training.loc[:, dntpair_cols_list]
y = df0_CoVs_ORF_training.loc[:, 'adaptation']
print (X.shape, y.shape)
print (df0_CoVs_ORF_training['adaptation'].value_counts())

     
#        print (X.shape, y.shape)
#        print (df0_CoVs_all['host2'].value_counts())
#        
         ########################################################    prepair data

         ########################################## data split and SMOTE resampling
for split_i in range(split_i,split_i+1,1):
    for split_size in split_size_list:
        train_x, valid_x, train_y, valid_y = train_test_split(X, y, test_size=split_size, random_state = split_i)#X：要划分的样本特征集，y：要划分的样本结果，testsize：样本占比，种子
        print (train_x.shape, valid_x.shape)

        for smote_i in range (1,smote_i,1):
            smo = SMOTE(random_state = smote_i)#随机选取少数类样本用以合成新样本
            SMOTE_train_x, SMOTE_train_y = smo.fit_resample(train_x, train_y)
            SMOTE_test_x, SMOTE_test_y = smo.fit_resample(valid_x, valid_y)
            # SMOTE_train_x.to_csv('SMOTE_train_x.csv')
            # SMOTE_train_y.to_csv('SMOTE_train_y.csv')
            print ('SMOTE',SMOTE_train_x.shape,SMOTE_test_x.shape)
            print(pd.value_counts(SMOTE_train_y).sort_index())
            print(pd.value_counts(SMOTE_test_y).sort_index())

            ##############################    prepair data | data sampling
            ##############################    prepair data | data sampling

            for epoch_num in epoch_num_list:
                name_train = 'Spike_splitRand' + str(split_i) + '_smoteRand' + str(smote_i) + '_test_size'+ str(split_size) +'_epoch_num' + str(epoch_num) + '_predCHI'
                aucs = []
                pred_y_list=[]
                true_y_list=[]
                df_X_test=pd.DataFrame()
                mean_fpr = np.linspace(0, 1, 100)
                training_loss_list_all = []
                training_loss_list = []

                #############################################################     CNN training
                #############################################################     CNN training
                
                X_train_array = np.array(SMOTE_train_x)
                train_num = X_train_array.shape[0]
                y_train_list = SMOTE_train_y.tolist()
                X_valid_array = np.array(valid_x)
                y_valid_list = valid_y.tolist()
                valid_num = X_valid_array.shape[0]
                valid_size = len(y_valid_list)
                X_train_array2 = X_train_array.reshape(train_num,1,6,16,16)
                X_valid_array2 = X_valid_array.reshape(valid_num,1,6,16,16)
                
                X_train_tensor = torch.tensor(X_train_array2)
                y_train_tensor = torch.tensor(y_train_list)
                X_valid_tensor = torch.tensor(X_valid_array2)
                y_valid_tensor = torch.tensor(y_valid_list)
                X_train_tensor = X_train_tensor.to(torch.float32)
                X_valid_tensor = X_valid_tensor.to(torch.float32)
            
                # ###########################################################  data loader
                
                torch_train = Data.TensorDataset (X_train_tensor, y_train_tensor)
                print (torch_train)
                loader = Data.DataLoader (dataset = torch_train,
                                          batch_size = 20,
                                          shuffle = True,
                                          num_workers = 0)  # it defines Multiprocess, default ==0, causing error with more than 0, 

                #############################################################  data loader
                
                ###############################################################  build CNN
                
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
                
                    def forward (self, x):  # x is the a matrix, however, only lowcase is suggeted to use
                        x = self.conv1(x)
                        x = self.conv2(x)
                        x = self.conv3(x)
                        x = x.view (x.size(0), -1) # flat x, similar to reshape of numpy
                        fc_full = x
                        # return x.shape
                        x = F.sigmoid (self.fc1(x))        # to activate x
                        prob_ = F.softmax (self.fc3(x))  # for probability output
                        pred_ = self.fc3(x)                # for prediction output
                        return pred_, prob_, fc_full # also to output prob_ 
        
                cnn = CNN()
                if_use_gpu = 1
                if if_use_gpu:
                    cnn = cnn.cuda()
                    
                ###############################################################  buld CNN
                
                ##########################################  define optimizer and loss function
                
                optimizer = torch.optim.Adam(cnn.parameters(), lr = .001)#lr = .0001
                loss_func = nn.CrossEntropyLoss()
                
                ### train data
                for epoch in range (epoch_num):
                    for step, (x, y) in enumerate (loader):
                        b_x = Variable (x)
                # ###################################################
                        b_y = Variable(y)
                        if if_use_gpu:
                            b_x = b_x.cuda()
                            b_y = b_y.cuda()
                        pred = cnn (b_x)[0]
                        loss = loss_func (pred, b_y)
                        optimizer.zero_grad()                # 对loss求导
                        loss.backward()
                        optimizer.step()
                        training_loss_list.append(loss.cpu().detach().item())
                ##########################################  define optimizer and loss function

                print ('training finished!')


                df_training_loss = pd.DataFrame ()
                df_training_loss['Training_loss'] = training_loss_list
                df_name = 'df_training_loss_'+ name_train +'.csv'
                df_training_loss.to_csv (df_name)
        
                    ##########################################    predict sars2 adaptation
                # X_valid_tensor = X_valid_tensor.to(torch.float32)
                valid_pred_prob = cnn (Variable(X_valid_tensor).cuda())
                valid_pred_matrix = valid_pred_prob[0].cpu().detach().numpy()
                valid_pred_list = torch.max(valid_pred_prob[0],1)[1].data.cpu().detach().numpy()
                valid_pred_array = label_binarize (valid_pred_list, classes = [0, 1, 2, 3, 4])
                valid_true_array = label_binarize (y_valid_list, classes = [0, 1, 2, 3, 4])
                n_classes = valid_pred_array.shape[1]
                valid_prob_array = valid_pred_prob[1].cpu().detach().numpy()
        
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                plt.figure (figsize = (6,6))
                plt.grid(0)
                for i in range (n_classes):
                    fpr[i], tpr[i], thresholds = roc_curve(valid_true_array[:,i], valid_prob_array[:,i])
                    # print (tpr[i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], lw=1, alpha=0.3,\
                              label='ROC class %d (AUC = %0.2f)' % (i, roc_auc[i]))
                plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                          label='Chance', alpha=.8)
        
                plt.xlim([-0.05, 1.05])
                plt.ylim([-0.05, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend(loc="lower right")
                
                plt.savefig( 'ROC_AUC_prob_'+ name_train + '.png',dpi=600 ,bbox_inches='tight')
                plt.close()
        
                class_names = np.ravel(y_valid_list)
                class_names = class_names[unique_labels(y_valid_list, valid_pred_list)]

                plt.figure (figsize = (6,6))
                plt.grid(0)
                np.set_printoptions(precision=2)
                plot_confusion_matrix(y_valid_list, valid_pred_list, classes=class_names, normalize=True,
                                      title='Confusion matrix (%)')
                plt.grid(0)
                plt.savefig('Confusion matrix (%)_' + name_train + '.png',dpi=600,bbox_inches='tight')
                plt.close()
        
                fc_full_array = valid_pred_prob[2].cpu().detach().numpy()
                print (fc_full_array.shape)
                df_fc_full = pd.DataFrame (fc_full_array)
        
                pca = PCA(n_components=2)
                newX1 = pca.fit_transform(df_fc_full)
                newX11 = np.array (newX1)
                print (newX11.shape)

                y_valid_list = [y_dict[i] for i in y_valid_list]
                df_fc_full['True_label'] = y_valid_list
                cols = ['PCA1','PCA2']
                df_fc_full[cols] = newX11
                df_fc_full.to_csv ('df_fc_full_PCA_' + name_train + '.csv')

                plt.figure (figsize = (6,6))
                sns.set(font_scale=2)
                sns.set(style='white')
                sns.pairplot(data=df_fc_full[cols + ['True_label']],hue = 'True_label', kind="scatter")
                # sns.set(rc={'figure.figsize':(15,5)})
                # sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)
                plt.savefig('Pairplot_PCA_fc_full_' + name_train +'.png', dpi = 600,bbox_inches = 'tight')
                plt.close()

                model_name = 'DCR_CNN_model_' + name_train + '.txt'     ############   To save trained model
                torch.save(cnn.cpu(), model_name)