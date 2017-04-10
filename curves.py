#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:29:49 2017
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

############### Adversarial Learning #####################################################################################################################################
#read in the validation sets

#load model list 
adv_learning_models=joblib.load('adv_learning_models.pkl')

adv_learning_oot=[]

adv_trans_amount = []
for i in range(1,11):
    file=pd.read_csv('adv_learning_test_'+str(i)+'.csv')
    adv_learning_oot.append(file)
    
    fnr_file = file.copy(deep=True)
    #remove fraud indicator
    fnr_predict=fnr_file.drop('FRD_IND',axis=1)
    #remove index column
    fnr_predict=fnr_predict.drop(fnr_predict.columns[0],axis=1)
    
    model = adv_learning_models[(i-1)]
    fnr_mod = model.predict(fnr_predict)
    fnr_file['pred'] = fnr_mod
    
    trans_sum = 0
    fnr_index = fnr_file.where((fnr_file['pred'] == 0) & (fnr_file['FRD_IND'] == 1))
    trans_sum = fnr_index['AUTHZN_AMT'].sum()
            
    adv_trans_amount.append(trans_sum)
    
print("the money lost by adversarial learning by round: ",adv_trans_amount)  
    
#adv_learning_models=[]
#for mod in range(3):
#    x=joblib.load('adv_learning_model'+(str(mod))+'.pkl')
#    adv_learning_models.append(x)


#Finding False Negatives in Validation Sets





i_num = 0
fold_n=[1,4,7,10]


### ROC Curve (adversarial learning) ###

lw=2
colors = cycle(['cyan', 'indigo', 'seagreen','darkorange'])
folds_list=[adv_learning_oot[0].copy(deep=True),adv_learning_oot[3].copy(deep=True),adv_learning_oot[6].copy(deep=True),adv_learning_oot[9].copy(deep=True)]
model_list2=[adv_learning_models[0],adv_learning_models[3],adv_learning_models[6],adv_learning_models[9]]
#folds_list=adv_learning_oot
#model_list2=adv_learning_models
for l, color, z in zip(folds_list, colors, model_list2):
    l=l.copy(deep=True)
    #remove fraud indicator
    syntheticdata_test=l.drop('FRD_IND',axis=1)
    #remove index column
    syntheticdata_test=syntheticdata_test.drop(syntheticdata_test.columns[0],axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    #mod_test2 = z.predict(syntheticdata_test)
    mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
    #print("The FNR is:", cmfull[0][1])
    print("The Plot AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
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
plt.title('ROC Curve with Adversarial Learning')
plt.legend(loc="lower right")

#export plot
plt.savefig('adv_learning_roc.png',bbox_inches='tight')
plt.savefig('adv_learning_roc.svg',bbox_inches='tight')
#remove plot
plt.clf()
    
#print all AUCs
for fold, model in  zip(adv_learning_oot, adv_learning_models):
    fold=fold.copy(deep=True)
    #remove ground truth column
    syntheticdata_test=fold.drop('FRD_IND',axis=1)
    #remove index column
    syntheticdata_test=syntheticdata_test.drop(syntheticdata_test.columns[0],axis=1)
    mod_test3 = model.predict_proba(syntheticdata_test)[:,1]
    fpr, tpr, _ = roc_curve(fold['FRD_IND'], mod_test3)
    print("The All Rounds Adversarial Learning AUC score (model 1) is:", roc_auc_score(fold['FRD_IND'],mod_test3 ))





#### Coverage Curve for Adversarial Learning ####
  

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
    
ilist=[1,4,7,10]

folds_list2=[adv_learning_oot[0].copy(deep=True),adv_learning_oot[3].copy(deep=True),adv_learning_oot[6].copy(deep=True),adv_learning_oot[9].copy(deep=True)]  
model_list2=[adv_learning_models[0], adv_learning_models[3],adv_learning_models[6],adv_learning_models[9]]

val=1
#run coverage curve:   
for fold,model,color,i in zip(folds_list2,model_list2,colors,ilist):
    fold=fold.copy(deep=True)
    #remove ground truth column
    syntheticdata_test2=fold.drop('FRD_IND',axis=1)
    #remove index column
    syntheticdata_test2=syntheticdata_test2.drop(syntheticdata_test2.columns[0],axis=1)
    model_predictions=model.predict_proba(syntheticdata_test2)[:,1]
    fold['model_pred']=model_predictions
    # create sorted df
    sorted_df = coverage_curve(fold, 'FRD_IND', 'model_pred', fold['AUTHZN_AMT'])
    sorted_df.to_csv('adv_learning_coverage_'+str(val)+'.csv')
    #drop model_pred
    fold=fold.drop('model_pred',axis=1)
    # produce chart
    plt.plot(sorted_df['Trxn_Cumulative'], sorted_df['Fraud_Cumulative'], color=color, label='Round '+str(i))
    plt.xlabel('Cumulative Transactions Examined')
    plt.ylabel('Percent Fraud Caught')
    plt.title('Coverage Curve with Adversarial Learning')
    plt.legend(loc="lower right")
    val=val+1
    
#save plot
plt.savefig('adv_learn_coverage(1).png',bbox_inches='tight')
plt.savefig('adv_learn_coverage(1).svg',bbox_inches='tight')

#remove plot
plt.clf()   





##################### Adversary learns, classifier remains the same (no adv learning)############################################
#load model list 
no_learning_mod=joblib.load('no_learning_model.pkl')

no_learning_oot=[]

nolearn_trans_amount = []

for i in range(1,11):
    file=pd.read_csv('no_learning_test_'+str(i)+'.csv')
    no_learning_oot.append(file)
    
    fnr_file = file.copy(deep=True)
    #remove fraud indicator
    fnr_predict=fnr_file.drop('FRD_IND',axis=1)
    #remove index column
    fnr_predict=fnr_predict.drop(fnr_predict.columns[0],axis=1)
    
    model = no_learning_mod
    fnr_mod = model.predict(fnr_predict)
    fnr_file['pred'] = fnr_mod
    
    trans_sum = 0
    fnr_index = fnr_file.where((fnr_file['pred'] == 0) & (fnr_file['FRD_IND'] == 1))
    trans_sum = fnr_index['AUTHZN_AMT'].sum()
            
    nolearn_trans_amount.append(trans_sum)

print("the money lost without learning by round: ",nolearn_trans_amount)  

    
diff_list = [a_i - b_i for a_i, b_i in zip(adv_trans_amount, nolearn_trans_amount)]

i_num = 0
fold_n=[1,4,7,10]

colors = cycle(['cyan', 'indigo', 'seagreen','darkorange'])
folds_list=[no_learning_oot[0].copy(deep=True),no_learning_oot[3].copy(deep=True),no_learning_oot[6].copy(deep=True),no_learning_oot[9].copy(deep=True)]
firstmod=no_learning_mod


for l, color in zip(folds_list, colors):
    l=l.copy(deep=True)
    syntheticdata_test=l.drop('FRD_IND',axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    #mod_test2 = z.predict(syntheticdata_test)
    mod_test3 = firstmod.predict_proba(syntheticdata_test)[:,1]
    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
    #print("The FNR is:", cmfull[0][1])
    print("The Plot AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
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
plt.title('ROC Curve without Adversarial Learning')
plt.legend(loc="lower right")
#plt.show()

#savefig('~/adversarial_learning/out_of_time_roc.png',bbox_inches='tight')
plt.savefig('no_learning_roc.png',bbox_inches='tight')
plt.savefig('no_learning_roc.svg',bbox_inches='tight')

#remove plot
plt.clf()

#print all AUCs
for fold in no_learning_oot:
    fold=fold.copy(deep=True)
    model=firstmod
    syntheticdata_test=fold.drop('FRD_IND',axis=1)
    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
    mod_test3 = model.predict_proba(syntheticdata_test)[:,1]
    fpr, tpr, _ = roc_curve(fold['FRD_IND'], mod_test3)
    print("All Rounds No Learning AUC score is:", roc_auc_score(fold['FRD_IND'],mod_test3 ))  

### Coverage Curve ###

ilist=[1,4,7,10]
folds_list4=[no_learning_oot[0].copy(deep=True),no_learning_oot[3].copy(deep=True),no_learning_oot[6].copy(deep=True),no_learning_oot[9].copy(deep=True)]

val=1
#run coverage curve:   
for fold,color,i in zip(folds_list4,colors,ilist):
    fold=fold.copy(deep=True)
    model=firstmod
    syntheticdata_test4=fold.drop('FRD_IND',axis=1)
    model_predictions=model.predict_proba(syntheticdata_test4)[:,1]
    new_fold=fold
    new_fold['model_pred']=model_predictions
    # create sorted df
    sorted_df = coverage_curve(new_fold, 'FRD_IND', 'model_pred', new_fold['AUTHZN_AMT'])
    sorted_df.to_csv('no_learning_coverage_'+str(val)+'.csv')
    #drop model_pred
    new_fold=new_fold.drop('model_pred',axis=1)
    # produce chart
    plt.plot(sorted_df['Trxn_Cumulative'], sorted_df['Fraud_Cumulative'], color=color, label='Round '+str(i))
    plt.xlabel('Cumulative Transactions Examined')
    plt.ylabel('Percent Fraud Caught')
    plt.title('Coverage Curve without Adversarial Learning')
    plt.legend(loc="lower right")
    val=val+1
#save plot
plt.savefig('no_adv_coverage.png',bbox_inches='tight')
plt.savefig('no_adv_coverage.svg',bbox_inches='tight')

#remove plot
plt.clf()


##################


#try the following two scenarios later (use the adversar)
# #################################################################################################################################################################################   
# #different scenario where testing the trained adversary's strategies over 10 rounds of playing the game back on the first model   
#i_num = 0
#fold_n=[1,4,7,10]
#
#colors = cycle(['cyan', 'indigo', 'seagreen','darkorange'])
#folds_list=[all_fold_oot[0],all_fold_oot[3],all_fold_oot[6],all_fold_oot[9]]
#firstmod = model_list[0]
#for l, color in zip(folds_list, colors):
#    syntheticdata_test=l.drop('FRD_IND',axis=1)
#    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
#    #mod_test2 = z.predict(syntheticdata_test)
#    mod_test3 = firstmod.predict_proba(syntheticdata_test)[:,1]
#    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
#    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
#    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
#    #print("The FNR is:", cmfull[0][1])
#    print("The Outside of Time Sample AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
#    #aucscore = roc_auc_score(l['FRD_IND'],mod_test2 )
#    #getting predicted probablilites for fraud
#    #mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
#    #fpr1, tpr1, _ = roc_curve(l['FRD_IND'], mod_test3)
#    aucscore = auc(fpr, tpr )
#    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC Round %d (area = %0.2f)' % (fold_n[i_num], aucscore))
#    i_num += 1
#
##adding ROC curve code
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve with Adversarial Learning: Performance on First Model')
#plt.legend(loc="lower right")
##plt.show()
#
#
##savefig('~/adversarial_learning/out_of_time_roc.png',bbox_inches='tight')
#plt.savefig('out_of_time_roc_firstmodelstrategy_15pct.png',bbox_inches='tight')
#
##remove plot
#plt.clf()
#
#
#    
##print all AUCs 
#for fold in all_fold_oot:
#   model=model_list[0] #first model
#   syntheticdata_test=fold.drop('FRD_IND',axis=1)
#   #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
#   mod_test3 = model.predict_proba(syntheticdata_test)[:,1]
#   fpr, tpr, _ = roc_curve(fold['FRD_IND'], mod_test3)
#   print("The Outside of Time Sample AUC score (model 2) is:", roc_auc_score(fold['FRD_IND'],mod_test3 )) 
#
#####################################################################################################################################################################    
#the next scenario is testing the trained adversary's strategies over 10 rounds of playing the game on the last model 


####TRY THIS LATER####

#i_num = 0
#fold_n=[1,4,7,10]
#
#colors = cycle(['cyan', 'indigo', 'seagreen','darkorange'])
#folds_list=[all_fold_oot[0],all_fold_oot[3],all_fold_oot[6],all_fold_oot[9]]
#lastmod = model_list[9]
#for l, color in zip(folds_list, colors):
#    syntheticdata_t=l.drop('FRD_IND',axis=1)
#    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
#    #mod_test2 = z.predict(syntheticdata_test)
#    mod_test3 = lastmod.predict_proba(syntheticdata_t)[:,1]
#    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
#    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
#    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
#    #print("The FNR is:", cmfull[0][1])
#    print("The Outside of Time Sample AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
#    #aucscore = roc_auc_score(l['FRD_IND'],mod_test2 )
#    #getting predicted probablilites for fraud
#    #mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
#    #fpr1, tpr1, _ = roc_curve(l['FRD_IND'], mod_test3)
#    aucscore = auc(fpr, tpr )
#    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC Round %d (area = %0.2f)' % (fold_n[i_num], aucscore))
#    i_num += 1
#
##adding ROC curve code
#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve with Adversarial Learning: Performance on Last Model')
#plt.legend(loc="lower right")
##plt.show()
#
#
##savefig('~/adversarial_learning/out_of_time_roc.png',bbox_inches='tight')
#plt.savefig('out_of_time_roc_lastmodelstrategy_15pct.png',bbox_inches='tight')
#
##remove plot
#plt.clf()
#
#
##print all AUCs
#for fold in all_fold_oot:
#    model=model_list[9] #last model
#    syntheticdata_t2=fold.drop('FRD_IND',axis=1)
#    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
#    mod_test3 = model.predict_proba(syntheticdata_t2)[:,1]
#    fpr, tpr, _ = roc_curve(fold['FRD_IND'], mod_test3)
#    print("The Outside of Time Sample AUC score is:", roc_auc_score(fold['FRD_IND'],mod_test3 )) 
#
#folds_list3=[all_fold_oot[0],all_fold_oot[3],all_fold_oot[6],all_fold_oot[9]]    
# #run coverage curve:   
#for fold,color in zip(folds_list3,colors):
#    model=model_list[9]
#    syntheticdata_test3=fold.drop('FRD_IND',axis=1)
#    #syntheticdata_test=syntheticdata_test.drop('model_pred',axis=1)
#    model_predictions=model.predict_proba(syntheticdata_test3)[:,1]
#    new_fold=fold
#    new_fold['model_pred']=model_predictions
#    # create sorted df
#    sorted_df = coverage_curve(new_fold, 'FRD_IND', 'model_pred', new_fold['AUTHZN_AMT'])
#    #remove model_pred
#    new_fold=new_fold.drop('model_pred',axis=1)
#    # produce chart
#    plt.plot(sorted_df['Trxn_Cumulative'], sorted_df['Fraud_Cumulative'], color=color, label='ROC Round %d' % (fold_n[i_num]))
#    plt.title('Coverage Curve with Adversarial Learning')
#    plt.legend(loc="lower right")
##save plot
#plt.savefig('coverage_adv_learn_2.png',bbox_inches='tight')
#
##remove plot
#plt.clf()   
#    

###################