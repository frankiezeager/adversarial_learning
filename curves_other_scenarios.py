#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 19:52:03 2017

@author: frankiezeager
"""


import pandas as pd
import numpy as np
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from numpy import linspace,hstack
from pylab import plot,show,hist,savefig
import os
import copy
import pickle
from sklearn.externals import joblib
plt.switch_backend('agg')

#read in the data
#load model list 
adv_learning_models=joblib.load('adv_learning_models.pkl')

adv_learning_oot=[]

adv_trans_amount = []

for i in range(1,11):
    file=pd.read_csv('adv_learning_test_'+str(i)+'.csv')
    #remove index column
    file=file.drop(file.columns[0],axis=1)
    adv_learning_oot.append(file)


# define coverage_curve function
def coverage_curve(df, target_variable_col, predicted_prob_fraud_col, trxn_amount_col):
    df=df.copy(deep=True)
    df = df.sort_values(predicted_prob_fraud_col, ascending=False)
    df['Fraud_Cumulative'] = df[target_variable_col].cumsum()*1.0 / df[target_variable_col].sum( )
    df['TrxnCount'] = 1
    df['Trxn_Cumulative'] = df['TrxnCount'].cumsum()*1.0 / df['TrxnCount'].sum( )
    #df['Exposure'] = df[trxn_amount_col] * df[target_variable_col]
    #df['Exposure_Cumulative'] = df['Exposure'].cumsum()*1.0 / df['Exposure'].sum( )
    #df['FPrate_Cumulative'] = (df['TrxnCount'].cumsum()*1.0 - df[target_variable_col].cumsum()) / df[target_variable_col].cumsum()
  
    return df
##################
#new scenarios
##################

#################################################################################################################################################################################   
 #different scenario where testing the trained adversary's strategies over 10 rounds of playing the game back on the first model   
i_num = 0
fold_n=[1,4,7,10]
lw=2

colors = cycle(['cyan', 'indigo', 'seagreen','darkorange'])
folds_list11=[adv_learning_oot[0].copy(deep=True),adv_learning_oot[3].copy(deep=True),adv_learning_oot[6].copy(deep=True),adv_learning_oot[9].copy(deep=True)]
firstmod = adv_learning_models[0]
for l, color in zip(folds_list11, colors):
    syntheticdata_test7=l.drop('FRD_IND',axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    #mod_test2 = z.predict(syntheticdata_test)
    mod_test3 = firstmod.predict_proba(syntheticdata_test7)[:,1]
    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
    #print("The FNR is:", cmfull[0][1])
    print("The Outside of Time Sample AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
    #aucscore = roc_auc_score(l['FRD_IND'],mod_test2 )
    #getting predicted probablilites for fraud
    #mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
    #fpr1, tpr1, _ = roc_curve(l['FRD_IND'], mod_test3)
    aucscore = auc(fpr, tpr )
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC Round %d (area = %0.2f)' % (fold_n[i_num], aucscore))
    i_num += 1

#adding ROC curve code
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Adversarial Learning: Performance on First Model')
plt.legend(loc="lower right")
#plt.show()


#savefig('~/adversarial_learning/out_of_time_roc.png',bbox_inches='tight')
plt.savefig('out_of_time_roc_firstmodelstrategy.png',bbox_inches='tight')
plt.savefig('out_of_time_roc_firstmodelstrategy.svg',bbox_inches='tight')
#remove plot
plt.clf()


    
#print all AUCs 
for fold in adv_learning_oot:
   model=adv_learning_models[0] #first model
   syntheticdata_test9=fold.drop('FRD_IND',axis=1)
   #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
   mod_test3 = model.predict_proba(syntheticdata_test9)[:,1]
   fpr, tpr, _ = roc_curve(fold['FRD_IND'], mod_test3)
   print("The Outside of Time Sample AUC score (model 2 adv learning, first mod) is:", roc_auc_score(fold['FRD_IND'],mod_test3 )) 

####################################################################################################################################################################    
#the next scenario is testing the trained adversary's strategies over 10 rounds of playing the game on the last model 



i_num = 0
fold_n=[1,4,7,10]

colors = cycle(['cyan', 'indigo', 'seagreen','darkorange'])
folds_list10=[adv_learning_oot[0].copy(deep=True),adv_learning_oot[3].copy(deep=True),adv_learning_oot[6].copy(deep=True),adv_learning_oot[9].copy(deep=True)]
lastmod = adv_learning_models[9]
for l, color in zip(folds_list10, colors):
    syntheticdata_t=l.drop('FRD_IND',axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    #mod_test2 = z.predict(syntheticdata_test)
    mod_test3 = lastmod.predict_proba(syntheticdata_t)[:,1]
    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
    #print("The FNR is:", cmfull[0][1])
    print("The Outside of Time Sample AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
    #aucscore = roc_auc_score(l['FRD_IND'],mod_test2 )
    #getting predicted probablilites for fraud
    #mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
    #fpr1, tpr1, _ = roc_curve(l['FRD_IND'], mod_test3)
    aucscore = auc(fpr, tpr )
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC Round %d (area = %0.2f)' % (fold_n[i_num], aucscore))
    i_num += 1

#adding ROC curve code
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with Adversarial Learning: Performance on Last Model')
plt.legend(loc="lower right")
#plt.show()


#savefig('~/adversarial_learning/out_of_time_roc.png',bbox_inches='tight')
plt.savefig('out_of_time_roc_lastmodelstrategy.png',bbox_inches='tight')
plt.savefig('out_of_time_roc_lastmodelstrategy.svg',bbox_inches='tight')


#remove plot
plt.clf()


#print all AUCs
for fold in adv_learning_oot:
    model=adv_learning_models[9] #last model
    syntheticdata_t2=fold.drop('FRD_IND',axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    mod_test3 = model.predict_proba(syntheticdata_t2)[:,1]
    fpr, tpr, _ = roc_curve(fold['FRD_IND'], mod_test3)
    print("The Outside of Time Sample AUC score is:", roc_auc_score(fold['FRD_IND'],mod_test3 )) 

folds_list3=[adv_learning_oot[0].copy(deep=True),adv_learning_oot[3].copy(deep=True),adv_learning_oot[6].copy(deep=True),adv_learning_oot[9].copy(deep=True)]    

i_num = 0
fold_n=[1,4,7,10]


#run coverage curve:   
for fold,color in zip(folds_list3,colors):
    model=adv_learning_models[9]
    syntheticdata_test3=fold.drop('FRD_IND',axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    model_predictions=model.predict_proba(syntheticdata_test3)[:,1]
    new_fold=fold
    new_fold['model_pred']=model_predictions
    # create sorted df
    sorted_df = coverage_curve(new_fold, 'FRD_IND', 'model_pred', new_fold['AUTHZN_AMT'])
    #remove model_pred
    new_fold=new_fold.drop('model_pred',axis=1)
    # produce chart
    plt.plot(sorted_df['Trxn_Cumulative'], sorted_df['Fraud_Cumulative'], color=color, label='ROC Round %d' % (fold_n[i_num]))
    plt.title('Coverage Curve with Adversarial Learning')
    plt.legend(loc="lower right")
#save plot
plt.savefig('coverage_last_model_strategy.png',bbox_inches='tight')
plt.savefig('coverage_last_model_strategy.svg',bbox_inches='tight')

#remove plot
plt.clf()   
    


###################
