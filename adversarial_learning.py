# -*- coding: utf-8 -*-

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

colnames = ['AUTH_ID','ACCT_ID_TOKEN','FRD_IND','ACCT_ACTVN_DT','ACCT_AVL_CASH_BEFORE_AMT','ACCT_AVL_MONEY_BEFORE_AMT','ACCT_CL_AMT','ACCT_CURR_BAL','ACCT_MULTICARD_IND','ACCT_OPEN_DT','ACCT_PROD_CD','ACCT_TYPE_CD','ADR_VFCN_FRMT_CD','ADR_VFCN_RESPNS_CD','APPRD_AUTHZN_CNT','APPRD_CASH_AUTHZN_CNT','ARQC_RSLT_CD','AUTHZN_ACCT_STAT_CD','AUTHZN_AMT','AUTHZN_CATG_CD','AUTHZN_CHAR_CD','AUTHZN_OPSET_ID','AUTHZN_ORIG_SRC_ID','AUTHZN_OUTSTD_AMT','AUTHZN_OUTSTD_CASH_AMT','AUTHZN_RQST_PROC_CD','AUTHZN_RQST_PROC_DT','AUTHZN_RQST_PROC_TM','AUTHZN_RQST_TYPE_CD','AUTHZN_TRMNL_PIN_CAPBLT_NUM','AVG_DLY_AUTHZN_AMT','CARD_VFCN_2_RESPNS_CD','CARD_VFCN_2_VLDTN_DUR','CARD_VFCN_MSMT_REAS_CD','CARD_VFCN_PRESNC_CD','CARD_VFCN_RESPNS_CD','CARD_VFCN2_VLDTN_CD','CDHLDR_PRES_CD','CRCY_CNVRSN_RT','ELCTR_CMRC_IND_CD','HOME_PHN_NUM_CHNG_DUR','HOTEL_STAY_CAR_RENTL_DUR','LAST_ADR_CHNG_DUR','LAST_PLSTC_RQST_REAS_CD','MRCH_CATG_CD','MRCH_CNTRY_CD','NEW_USER_ADDED_DUR','PHN_CHNG_SNC_APPN_IND','PIN_BLK_CD','PIN_VLDTN_IND','PLSTC_ACTVN_DT','PLSTC_ACTVN_REQD_IND','PLSTC_FRST_USE_TS','PLSTC_ISU_DUR','PLSTC_PREV_CURR_CD','PLSTC_RQST_TS','POS_COND_CD','POS_ENTRY_MTHD_CD','RCURG_AUTHZN_IND','RVRSL_IND','SENDR_RSIDNL_CNTRY_CD','SRC_CRCY_CD','SRC_CRCY_DCML_PSN_NUM','TRMNL_ATTNDNC_CD','TRMNL_CAPBLT_CD','TRMNL_CLASFN_CD','TRMNL_ID','TRMNL_PIN_CAPBLT_CD','DISTANCE_FROM_HOME']
random.seed(1345)

iteration_num = 0

listFNR=[]
#split each fold into training and testing sets
best_strat_list=[]
AUC_list=[]
model_list=[]

iter_num = 1

allsynthetic_sets = []
all_fold_oot=[]

for i in range(1,11):
    #read in the data file
    df1=pd.read_csv('training_part_0'+str(i)+'_of_10.txt',delimiter='|',header=None, names=colnames)
    #df1=pd.read_csv(str('/Users/frankiezeager/Documents/Graduate School/Capstone/'+'training_part_0'+str(i)+'_of_10.txt'),delimiter='|',header=None, names=colnames)
    #df1=df1.sample(n=100000)
    #df1=pd.read_csv('/Users/frankiezeager/Documents/Graduate School/Capstone/training_part_01_of_10.txt',delimiter='|',header=None, names=colnames)
    #sort values by date
    df1 = df1.sort_values(['AUTHZN_RQST_PROC_DT'])
    
    #only select needed columns
    #2-Fraud index (whether or not is fraud)
    #7-ACCT_CUR_BAL
    #14-Total Num. Authorizations
    #18-authorization amount
    #23-authorization outstanding amount
    #30-Avg. Daily Authorization Amount
    #44-Merchant Category ID (convert to factor)
    #53-Plastic Issue Duration
    #57- Point of Sale Entry Method (convert to factor)
    #58 - Recurring Charge (Relabel as 0/1)
    #68 - Distance from Home
    t_ind = [2, 7, 14, 18, 23, 30, 44, 53, 57, 58, 68]
    
    #cols = pd.DataFrame(j.iloc[:,t_ind])
    df1 = pd.DataFrame(df1.iloc[:,t_ind])
    
    #take out any NaN's from the data set
    df1=df1.dropna(axis=0)
    
    #convert these two columns from Y and N to 1 and 0,
    df1['FRD_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    df1['RCURG_AUTHZN_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    fraud_list=df1['FRD_IND']
    df1=df1.drop('FRD_IND',axis=1)
    df1['FRD_IND']=fraud_list

    col_list=df1.columns.values.tolist()
   #take out any NaN's from the data set
    df1=df1.dropna(axis=0)
    
    #convert column type to numeric
    df1['RCURG_AUTHZN_IND'] = df1['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
    df1['FRD_IND'] = df1['FRD_IND'].convert_objects(convert_numeric=True)
    
    if i!=1:
        #find percent fraud in current df
        df_pct_fraud=sum(df1.FRD_IND==1)/len(df1)
        #find difference between this and the 15% fraud desired
        pct_fraud_needed=.15-df_pct_fraud
        #find number of fraud transactions needed
        num_fraud_trans=pct_fraud_needed*len(df1)
        num_fraud_trans=num_fraud_trans.astype(int)
        #finding the fraudulent transactions in synthetic data
        fraud_trans = syntheticdata[syntheticdata.iloc[:,-1] == 1]
        #sampling the fraud transactions to include amount needed
        add_fraud = fraud_trans.sample(n=num_fraud_trans, replace=True)
        #adding fraud transactions back to df1
        df1=pd.concat([df1,add_fraud],axis=0)

    #split into training, 'testing' (finding the adversarial best strategy data frame), out of time (validation set)
    #60% training, 20% test, 20% validation
    
    #alternative method to find 60-20-20 train, test, validate but according to time order
#    def train_test_validate_split(df, train_percent=.6, validate_percent=.2):
#        m = len(df)
#        train_end = int(train_percent * m)
#        validate_end = int(validate_percent * m) + train_end
#        train = df.iloc[:train_end]
#        test = df.iloc[train_end:validate_end]
#        out_of_time = df.iloc[validate_end:]
#        return train, test, out_of_time
#    
#    train, test, out_of_time = train_test_validate_split(df1)

    train,intermediate_set=train_test_split(df1,train_size=.6,test_size=.4,random_state=1575)		
    test, out_of_time=train_test_split(intermediate_set, train_size=.5,test_size=.5,random_state=1575)	
    
    #delete intermediate set
    del intermediate_set
    
    
     #filling any NAs 
    train=train.fillna(method='ffill')
    test=test.fillna(method='ffill')
    out_of_time=out_of_time.fillna(method='ffill')
    
    #convert out of time data frame variables
    out_of_time.FRD_IND.map(dict(Y=1,N=0))
    out_of_time.RCURG_AUTHZN_IND.map(dict(Y=1,N=0))
    
    #filling any NAs 
    out_of_time=out_of_time.fillna(method='ffill')
    
    
    #convert column type to numeric
    #df1['RCURG_AUTHZN_IND'] = df1['RCURG_AUTHZN_IND'].apply(pd.to_numeric)
    #df1['FRD_IND'] = df1['FRD_IND'].apply(pd.to_numeric)
    out_of_time['RCURG_AUTHZN_IND'] = out_of_time['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
    out_of_time['FRD_IND'] = out_of_time['FRD_IND'].convert_objects(convert_numeric=True)
    all_fold_oot.append(out_of_time)
    
    test = test.sort_values(['AUTHZN_AMT']) #sorting by transaction amount

#training the classifier
#7-ACCT_CUR_BAL
#14-Total Num. Authorizations
#30-Avg. Daily Authorization Amount
#44-Merchant Category ID (convert to factor)
#57- Point of Sale Entry Method (convert to factor)
#58 - Recurring Charge (Relabel as 0/1)
#68 - Distance from Home
    
    train_cols = train.drop('FRD_IND', axis=1) #columns for training
    
    
    #Logistic Regression
    logit = LogisticRegression(class_weight='balanced')
    
    mod = logit.fit(train_cols, train['FRD_IND'])
    model_list.append(mod)
    testcol = test.drop('FRD_IND',axis=1)
    #testcol=testcol.fillna(method='ffill')
    testcol.fillna(method='bfill',inplace=True)
    
    #testcol[np.isnan(testcol)] = np.median(testcol[~np.isnan(testcol)])
    mod_test = mod.predict(testcol)

###############################################################################

    #find false negative rate
    cmfull=confusion_matrix(test['FRD_IND'],mod_test)
    listFNR.append(cmfull[0][1])
    fpr, tpr, thresholds = metrics.roc_curve(test['FRD_IND'], mod_test, pos_label=2)

    #find auc score
    print(roc_auc_score(test['FRD_IND'],mod_test ))
    AUC_list.append(roc_auc_score(test['FRD_IND'],mod_test ))

#########Gaussian Mixture Model to Determine Strategies#############

    #subset df to include only pertinent (adversarial-controlled) continuous vars
    strat_ind = [0, 1, 2, 3, 4, 6, 9] 
    strategy_df= pd.DataFrame(test.iloc[:,strat_ind]) 


    #find best number of strategies:
    lowest_bic = np.infty
    bic = []
          # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=3,covariance_type='full')
    gmm.fit(strategy_df)

    #assign each transaction a strategy
    strat_assign=gmm.predict(strategy_df)

    #attach back to data frame
    test['Strategy Number'] = strat_assign
              
    all_batches = []
    for t, B_t in test.groupby('Strategy Number'):
        #B_t['batch_num'] = t
        #check for fraud transactions
        if B_t['FRD_IND'].sum()> 0:
            all_batches.append(B_t)

    fn_rate = []
    for j in all_batches:
        #cols=j.drop(j.columns[-1:-3], axis=1) 
        cols = j.drop(labels='Strategy Number',axis=1)
        cols = cols.drop(labels='FRD_IND',axis=1)
        #cols=j.iloc[:,:-1] 
        #cols=cols.iloc[:,:-1]
        cols.fillna(method='bfill',inplace=True)
        col_response = j.iloc[:,-2]
        pred = mod.predict(cols)
        cm = confusion_matrix(col_response, pred)
        FNR = cm[0][1]
        fn_rate.append(FNR)

    best_strat = fn_rate.index(max(fn_rate))
    best_strat_list.append(best_strat)
    best_fold = all_batches[best_strat]

    #Implement SMOTE (add 'good' fraud to the dataset)
    test_cols = best_fold.drop(labels='Strategy Number',axis=1)
    test_cols = test_cols.drop(labels='FRD_IND',axis=1)
    smote = SMOTE(ratio=0.5, kind='regular',random_state=1345)
    smox, smoy = smote.fit_sample(test_cols, best_fold.FRD_IND)
    smox = pd.DataFrame(smox)
    smoy = pd.DataFrame(smoy)
    syntheticdata = pd.concat((smox,smoy), axis=1)
    syntheticdata.columns=col_list
    #delete data frame to make more space in memory
    #del df1


#output files
val=1
for file in all_fold_oot:
    file.to_csv('adv_learning_test_'+str(val)+'.csv')
    val=val+1

#output models
#for mod in range(3):
#    joblib.dump(model_list[mod], 'adv_learning_model'+(str(mod))+'.pkl')
joblib.dump(model_list, 'adv_learning_models.pkl',compress=True)   

######################################## MODEL STAYS THE SAME, ADVERSARY CHANGES (no adv learning) ###############################################################################################################
#make sure the seed is set
random.seed(1345)      
        
iteration_num = 0

listFNR=[]
#split each fold into training and testing sets
best_strat_list=[]
AUC_list=[]

iter_num = 1

lw = 2

allsynthetic_sets = []
all_fold_oot=[]

for i in range(1,11):
    #read in the data file
    df1=pd.read_csv('/home/ec2-user/adversarial_learning/'+'training_part_0'+str(i)+'_of_10.txt',delimiter='|',header=None, names=colnames)
    #df1=pd.read_csv(str('/Users/frankiezeager/Documents/Graduate School/Capstone/'+'training_part_0'+str(i)+'_of_10.txt'),delimiter='|',header=None, names=colnames)
    
    #take out any NaN's from the data set
    df1=df1.dropna(axis=0)
    
    #sort values by date
    df1 = df1.sort_values(['AUTHZN_RQST_PROC_DT'])
    
    #only select needed columns
    #2-Fraud index (whether or not is fraud)
    #7-ACCT_CUR_BAL
    #14-Total Num. Authorizations
    #18-authorization amount
    #23-authorization outstanding amount
    #30-Avg. Daily Authorization Amount
    #44-Merchant Category ID (convert to factor)
    #53-Plastic Issue Duration
    #57- Point of Sale Entry Method (convert to factor)
    #58 - Recurring Charge (Relabel as 0/1)
    #68 - Distance from Home
    t_ind = [2, 7, 14, 18, 23, 30, 44, 53, 57, 58, 68]
    
    #cols = pd.DataFrame(j.iloc[:,t_ind])
    df1 = pd.DataFrame(df1.iloc[:,t_ind])
    #convert these two columns from Y and N to 1 and 0,
    df1['FRD_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    df1['RCURG_AUTHZN_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    fraud_list=df1['FRD_IND']
    df1=df1.drop('FRD_IND',axis=1)
    df1['FRD_IND']=fraud_list

    col_list=df1.columns.values.tolist()

#    #convert column type to numeric
    #df1['RCURG_AUTHZN_IND'] = df1['RCURG_AUTHZN_IND'].apply(pd.to_numeric)
    #df1['FRD_IND'] = df1['FRD_IND'].apply(pd.to_numeric)
    df1['RCURG_AUTHZN_IND'] = df1['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
    df1['FRD_IND'] = df1['FRD_IND'].convert_objects(convert_numeric=True)
    
    
    if i!=1:
        #find percent fraud in current df
        df_pct_fraud=sum(df1.FRD_IND==1)/len(df1)
        #find difference between this and the 15% fraud desired
        pct_fraud_needed=.15-df_pct_fraud
        #find number of fraud transactions needed
        num_fraud_trans=pct_fraud_needed*len(df1)
        num_fraud_trans=num_fraud_trans.astype(int)
        #finding the fraudulent transactions in synthetic data
        fraud_trans = syntheticdata[syntheticdata.iloc[:,-1] == 1]
        #sampling the fraud transactions to include amount needed
        add_fraud = fraud_trans.sample(n=num_fraud_trans, replace=True)
        #adding fraud transactions back to df1
        df1=pd.concat([df1,add_fraud],axis=0)

 
    
    #split into training, 'testing' (finding the adversarial best strategy data frame), out of time (validation set)
    #60% training, 20% test, 20% validation

    #alternative method to find 60-20-20 train, test, validate but according to time order
#    def train_test_validate_split(df, train_percent=.6, validate_percent=.2, seed=1345):
#        m = len(df)
#        train_end = int(train_percent * m)
#        validate_end = int(validate_percent * m) + train_end
#        train = df.iloc[:train_end]
#        test = df.iloc[train_end:validate_end]
#        out_of_time = df.iloc[validate_end:]
#        return train, test, out_of_time
#    
#    train, test, out_of_time = train_test_validate_split(df1)
    
    train,intermediate_set=train_test_split(df1,train_size=.6,test_size=.4,random_state=1575)		
    test, out_of_time=train_test_split(intermediate_set, train_size=.5,test_size=.5,random_state=1575)	
    #delete intermediate set
    del intermediate_set

    train=train.fillna(method='ffill')
    test=test.fillna(method='ffill')
    out_of_time=out_of_time.fillna(method='ffill')

    
    #convert out of time data frame variables
    out_of_time.FRD_IND.map(dict(Y=1,N=0))
    out_of_time.RCURG_AUTHZN_IND.map(dict(Y=1,N=0))
    #out_of_time['FRD_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    #out_of_time['RCURG_AUTHZN_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    
    #convert column type to numeric
    out_of_time['RCURG_AUTHZN_IND'] = out_of_time['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
    out_of_time['FRD_IND'] = out_of_time['FRD_IND'].convert_objects(convert_numeric=True)
    all_fold_oot.append(out_of_time)
    
    test = test.sort_values(['AUTHZN_AMT']) #sorting by transaction amount
#training the classifier
#7-ACCT_CUR_BAL
#14-Total Num. Authorizations
#30-Avg. Daily Authorization Amount
#44-Merchant Category ID (convert to factor)
#57- Point of Sale Entry Method (convert to factor)
#58 - Recurring Charge (Relabel as 0/1)
#68 - Distance from Home

    
    train_cols = train.drop('FRD_IND', axis=1) #columns for training
    
    #Logistic Regression
    #logit = LogisticRegression(class_weight='balanced')
    
    #use the first model every time
    mod = model_list[0]
    #model_list.append(mod)
    testcol = test.drop('FRD_IND',axis=1)
    testcol.fillna(method='bfill',inplace=True)
    mod_test = mod.predict(testcol)

###############################################################################


    #find false negative rate
    cmfull=confusion_matrix(test['FRD_IND'],mod_test)
    listFNR.append(cmfull[0][1])
    fpr, tpr, thresholds = metrics.roc_curve(test['FRD_IND'], mod_test, pos_label=2)

    #find auc score
    print(roc_auc_score(test['FRD_IND'],mod_test ))
    AUC_list.append(roc_auc_score(test['FRD_IND'],mod_test ))

#########Gaussian Mixture Model to Determine Strategies#############

    #subset df to include only pertinent (adversarial-controlled) continuous vars
    strat_ind = [0, 1, 2, 3, 4, 6, 9] 
    strategy_df= pd.DataFrame(test.iloc[:,strat_ind]) 


    #find best number of strategies:
    lowest_bic = np.infty
    bic = []
          # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=3,covariance_type='full')
    gmm.fit(strategy_df)

    #assign each transaction a strategy
    strat_assign=gmm.predict(strategy_df)

    #attach back to data frame
    test['Strategy Number'] = strat_assign
              
    all_batches = []
    for t, B_t in test.groupby('Strategy Number'):
        #B_t['batch_num'] = t
        #check for fraud transactions
        if B_t['FRD_IND'].sum()> 0:
            all_batches.append(B_t)

    fn_rate = []
    for j in all_batches:
        #cols=j.drop(j.columns[-1:-3], axis=1) 
        cols = j.drop(labels='Strategy Number',axis=1)
        cols = cols.drop(labels='FRD_IND',axis=1)
        cols.fillna(method='bfill',inplace=True)
        #cols=j.iloc[:,:-1] 
        #cols=cols.iloc[:,:-1]
        col_response = j.iloc[:,-2]
        pred = mod.predict(cols)
        cm = confusion_matrix(col_response, pred)
        FNR = cm[0][1]
        fn_rate.append(FNR)

    best_strat = fn_rate.index(max(fn_rate))
    best_strat_list.append(best_strat)
    best_fold = all_batches[best_strat]

    #Implement SMOTE (add 'good' fraud into the dataset)
    test_cols = best_fold.drop(labels='Strategy Number',axis=1)
    test_cols = test_cols.drop(labels='FRD_IND',axis=1)
    smote = SMOTE(ratio=0.5, kind='regular',random_state=1345)
    smox, smoy = smote.fit_sample(test_cols, best_fold.FRD_IND)
    smox = pd.DataFrame(smox)
    smoy = pd.DataFrame(smoy)
    syntheticdata = pd.concat((smox,smoy), axis=1)
    syntheticdata.columns=col_list
    #delete data frame to make more space in memory
    del df1
   


#output files
val=1
for file in all_fold_oot:
    file.to_csv('no_learning_test_'+str(val)+'.csv')
    val=val+1

#output models
joblib.dump(model_list[0], 'no_learning_model.pkl',compress=True)


