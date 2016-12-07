
# coding: utf-8

# Generation of new fraudulent transactions

import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.ensemble import VotingClassifier
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
import statsmodels.api as sm #package for logistic regression



colnames = ['AUTH_ID','ACCT_ID_TOKEN','FRD_IND','ACCT_ACTVN_DT','ACCT_AVL_CASH_BEFORE_AMT','ACCT_AVL_MONEY_BEFORE_AMT','ACCT_CL_AMT','ACCT_CURR_BAL','ACCT_MULTICARD_IND','ACCT_OPEN_DT','ACCT_PROD_CD','ACCT_TYPE_CD','ADR_VFCN_FRMT_CD','ADR_VFCN_RESPNS_CD','APPRD_AUTHZN_CNT','APPRD_CASH_AUTHZN_CNT','ARQC_RSLT_CD','AUTHZN_ACCT_STAT_CD','AUTHZN_AMT','AUTHZN_CATG_CD','AUTHZN_CHAR_CD','AUTHZN_OPSET_ID','AUTHZN_ORIG_SRC_ID','AUTHZN_OUTSTD_AMT','AUTHZN_OUTSTD_CASH_AMT','AUTHZN_RQST_PROC_CD','AUTHZN_RQST_PROC_DT','AUTHZN_RQST_PROC_TM','AUTHZN_RQST_TYPE_CD','AUTHZN_TRMNL_PIN_CAPBLT_NUM','AVG_DLY_AUTHZN_AMT','CARD_VFCN_2_RESPNS_CD','CARD_VFCN_2_VLDTN_DUR','CARD_VFCN_MSMT_REAS_CD','CARD_VFCN_PRESNC_CD','CARD_VFCN_RESPNS_CD','CARD_VFCN2_VLDTN_CD','CDHLDR_PRES_CD','CRCY_CNVRSN_RT','ELCTR_CMRC_IND_CD','HOME_PHN_NUM_CHNG_DUR','HOTEL_STAY_CAR_RENTL_DUR','LAST_ADR_CHNG_DUR','LAST_PLSTC_RQST_REAS_CD','MRCH_CATG_CD','MRCH_CNTRY_CD','NEW_USER_ADDED_DUR','PHN_CHNG_SNC_APPN_IND','PIN_BLK_CD','PIN_VLDTN_IND','PLSTC_ACTVN_DT','PLSTC_ACTVN_REQD_IND','PLSTC_FRST_USE_TS','PLSTC_ISU_DUR','PLSTC_PREV_CURR_CD','PLSTC_RQST_TS','POS_COND_CD','POS_ENTRY_MTHD_CD','RCURG_AUTHZN_IND','RVRSL_IND','SENDR_RSIDNL_CNTRY_CD','SRC_CRCY_CD','SRC_CRCY_DCML_PSN_NUM','TRMNL_ATTNDNC_CD','TRMNL_CAPBLT_CD','TRMNL_CLASFN_CD','TRMNL_ID','TRMNL_PIN_CAPBLT_CD','DISTANCE_FROM_HOME']
df = pd.read_csv('/Users/nathanfogal/Downloads/training_part_10_of_10.txt', delimiter='|',header=None, names=colnames)
df1 = df.sample(n=1000000)
df1.dtypes



df1['FRD_IND'].replace(['Y', 'N'], [1, 0], inplace=True)

df1['RCURG_AUTHZN_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
df1['RCURG_AUTHZN_IND'] = df1['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
df1['MRCH_CATG_CD'] = df1['MRCH_CATG_CD'].astype('category')
df1['POS_ENTRY_MTHD_CD'] = df1['POS_ENTRY_MTHD_CD'].astype('category')

#only select needed columns
t_ind = [2, 7, 14, 18, 26, 30, 44, 57, 58, 68]

#cols = pd.DataFrame(j.iloc[:,t_ind])
df1 = pd.DataFrame(df1.iloc[:,t_ind]) 


df1_MRCHCODE = pd.get_dummies(df1['MRCH_CATG_CD']) #converting to dummy variables
df1_POSENTRY = pd.get_dummies(df1['POS_ENTRY_MTHD_CD']) #converting to dummy variables

#join the dummy variables to the main data frame
df1 = pd.concat([df1, df1_MRCHCODE], axis=1)

#join the dummy variables to the main data frame
df1 = pd.concat([df1, df1_POSENTRY], axis=1)

#drop the original categorical variables (MRCH_CATG_CD) and (POS_ENTRY_MTHD_CD)
df1.drop(['MRCH_CATG_CD', 'POS_ENTRY_MTHD_CD', 'RCURG_AUTHZN_IND'],inplace=True,axis=1)




#create an out-of-time final test set
df2, out_of_time_test = train_test_split(df1, train_size = 0.9)

#sort by date (this sample goes from 1/1/2013 to 8/31/2013)
df2 = df2.sort_values(['AUTHZN_RQST_PROC_DT'])
df2=df2.drop('AUTHZN_RQST_PROC_DT',axis=1)
fraud_list=df2['FRD_IND']
df2=df2.drop('FRD_IND',axis=1)
df2['FRD_IND']=fraud_list
out_of_time_test=out_of_time_test.drop('AUTHZN_RQST_PROC_DT',axis=1)


col_list=df2.columns.values.tolist()

#creation of folds
fold_size = math.floor(df2.shape[0]/3)
folds = []
for f, F_t in df2.groupby(np.arange(len(df2)) // fold_size):
    
    if (len(F_t)==fold_size):
        folds.append(F_t)

iteration_num = 0

listFNR=[]
#split each fold into training and testing sets
best_strat_list=[]
AUC_list=[]
model_list=[]
for i in folds:
    fold_loc = i
    train, test = train_test_split(fold_loc, train_size = 0.5)
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
    logit = LogisticRegression(class_weight='auto')

    mod = logit.fit(train_cols, train['FRD_IND'])
    model_list.append(mod)
    testcol = test.drop('FRD_IND',axis=1)
    
    mod_test = mod.predict(testcol)
    mod_test_prob= mod.predict_proba(testcol)
    #bins = np.linspace(0, 1, 1000)
    #fold_loc_prob= fold_loc
    #fold_loc_prob["prob"]=mod_test_prob[0]
    #digitized = np.digitize(, bins)
    cmfull=confusion_matrix(test['FRD_IND'],mod_test)
    
    listFNR.append(cmfull[0][1])
    
    fpr, tpr, thresholds = metrics.roc_curve(test['FRD_IND'], mod_test, pos_label=2)
    
    print(roc_auc_score(test['FRD_IND'],mod_test ))
    AUC_list.append(roc_auc_score(test['FRD_IND'],mod_test ))
    
    #batches
    chunk_size = math.floor(test.shape[0]/3) #3 transaction strategies 
    chunk_size
    all_batches = []
    for t, B_t in test.groupby(np.arange(len(test)) // chunk_size):
        #B_t['batch_num'] = t
        #check for fraud transactions
        if (sum(B_t['FRD_IND']) != 0):
            all_batches.append(B_t)
    
    fn_rate = []
    for j in all_batches:
        cols=j.iloc[:,:-1]
        col_response = j.iloc[:,-1]
        pred = mod.predict(cols)
        cm = confusion_matrix(col_response, pred)
        FNR = cm[0][1]
        fn_rate.append(FNR)        
     
    best_strat = fn_rate.index(max(fn_rate))
    best_strat_list.append(best_strat)
    best_fold = all_batches[best_strat]         
            
    
    
    #Implement SMOTE
    #test_cols = test.drop("Class", axis = 1)
    test_cols = best_fold.drop('FRD_IND',axis=1)
    #test_cols = test_cols.values
    smote = SMOTE(ratio='auto', kind='regular')
    smox, smoy = smote.fit_sample(test_cols, best_fold.FRD_IND)
    smox = pd.DataFrame(smox)
    smoy = pd.DataFrame(smoy)
    syntheticdata = pd.concat((smox,smoy), axis=1)
    syntheticdata.columns=col_list
    #testing the synthetic data sets on the trained logistic regression classifie
    
    #index of the data chunk with the largest FNR (best strategy)
    
    #take all the fraudulent transactions of best strategy batch and add it to the next fold
    
    fraud_trans = best_fold[best_fold.iloc[:,-1] == 1]
    #append the fraud transactions to fold+1
    if (iteration_num+1)<len(folds):
        folds[iteration_num+1]=pd.concat([folds[iteration_num+1], fraud_trans], axis=0)
        iteration_num = iteration_num+ 1
        print("iter")

        
        
        
        
        
############################################################################
#out of time tests
        
#applying smote to out of time data as adversary would

for l in best_strat_list:
    out_of_time_test= out_of_time_test.sort_values(['AUTHZN_AMT'])
    chunk_size2 = math.floor(out_of_time_test.shape[0]/3) #3 transaction strategies 
    
    all_batches2 = []
    for q, B_t2 in out_of_time_test.groupby(np.arange(len(out_of_time_test) )// chunk_size2):
        #B_t['batch_num'] = t
        #check for fraud transactions
        if (sum(B_t2['FRD_IND']) != 0):
            all_batches2.append(B_t2)
    best_region = all_batches2[l]
    #Implement SMOTE
    #test_cols = test.drop("Class", axis = 1)
    test_cols2 = best_region.drop('FRD_IND',axis=1)
    smote2 = SMOTE(ratio='auto', kind='regular')
    smox2, smoy2 = smote.fit_sample(test_cols2, best_region.FRD_IND)
    smox2 = pd.DataFrame(smox2)
    smoy2 = pd.DataFrame(smoy2)
    syntheticdata2 = pd.concat((smox2,smoy2), axis=1)
    best_region=best_region[best_region['FRD_IND']==1]
    best_region.columns=col_list
    out_of_time_test=pd.concat([out_of_time_test,best_region],axis=0)
    print("iter")
       

    #testing the synthetic data sets on the trained logistic regression classifie
    
    
    
    
    
#Test update out of time list 
out_of_time_train=out_of_time_test.drop('FRD_IND',axis=1)    
mod_test2 = mod.predict(out_of_time_train)

cmfull=confusion_matrix(out_of_time_test['FRD_IND'],mod_test2)
fpr, tpr, thresholds = metrics.roc_curve(out_of_time_test['FRD_IND'], mod_test2, pos_label=2)
print("The FNR is:", cmfull[0][1])
print("The Outside of Time Sample AUC score is:", roc_auc_score(out_of_time_test['FRD_IND'],mod_test2 ))

    
#############################################################
#model doesnt change






for i in folds:
    fold_loc = i
    train, test = train_test_split(fold_loc, train_size = 0.5)
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

    testcol = test.drop('FRD_IND',axis=1)
    
    mod_test = mod_l.predict(testcol)
    mod_test_prob= mod.predict_proba(testcol)
    #bins = np.linspace(0, 1, 1000)
    #fold_loc_prob= fold_loc
    #fold_loc_prob["prob"]=mod_test_prob[0]
    #digitized = np.digitize(, bins)
    cmfull=confusion_matrix(test['FRD_IND'],mod_test)
    
    listFNR.append(cmfull[0][1])
    
    fpr, tpr, thresholds = metrics.roc_curve(test['FRD_IND'], mod_test, pos_label=2)
    
    print(roc_auc_score(test['FRD_IND'],mod_test ))
    AUC_list.append(roc_auc_score(test['FRD_IND'],mod_test ))
    
    #batches
    chunk_size = math.floor(test.shape[0]/3) #3 transaction strategies 
    chunk_size
    all_batches = []
    for t, B_t in test.groupby(np.arange(len(test)) // chunk_size):
        #B_t['batch_num'] = t
        #check for fraud transactions
        if (sum(B_t['FRD_IND']) != 0):
            all_batches.append(B_t)
    
    fn_rate = []
    for j in all_batches:
        cols=j.iloc[:,:-1]
        col_response = j.iloc[:,-1]
        pred = mod.predict(cols)
        cm = confusion_matrix(col_response, pred)
        FNR = cm[0][1]
        fn_rate.append(FNR)        
     
    best_strat = fn_rate.index(max(fn_rate))
    best_strat_list.append(best_strat)
    best_fold = all_batches[best_strat]         
            
    
    
    #Implement SMOTE
    #test_cols = test.drop("Class", axis = 1)
    test_cols = best_fold.drop('FRD_IND',axis=1)
    #test_cols = test_cols.values
    smote = SMOTE(ratio='auto', kind='regular')
    smox, smoy = smote.fit_sample(test_cols, best_fold.FRD_IND)
    smox = pd.DataFrame(smox)
    smoy = pd.DataFrame(smoy)
    syntheticdata = pd.concat((smox,smoy), axis=1)
    syntheticdata.columns=col_list
    #testing the synthetic data sets on the trained logistic regression classifie
    
    #index of the data chunk with the largest FNR (best strategy)
    
    #take all the fraudulent transactions of best strategy batch and add it to the next fold
    
    fraud_trans = best_fold[best_fold.iloc[:,-1] == 1]
    #append the fraud transactions to fold+1
    if (iteration_num+1)<len(folds):
        folds[iteration_num+1]=pd.concat([folds[iteration_num+1], fraud_trans], axis=0)
        iteration_num = iteration_num+ 1
        print("iter")

        
        
        
        
        
############################################################################
#out of time tests
        
#applying smote to out of time data as adversary would

for l in best_strat_list:
    out_of_time_test= out_of_time_test.sort_values(['AUTHZN_AMT'])
    chunk_size2 = math.floor(out_of_time_test.shape[0]/3) #3 transaction strategies 
    
    all_batches2 = []
    for q, B_t2 in out_of_time_test.groupby(np.arange(len(out_of_time_test) )// chunk_size2):
        #B_t['batch_num'] = t
        #check for fraud transactions
        if (sum(B_t2['FRD_IND']) != 0):
            all_batches2.append(B_t2)
    best_region = all_batches2[l]
    #Implement SMOTE
    #test_cols = test.drop("Class", axis = 1)
    test_cols2 = best_region.drop('FRD_IND',axis=1)
    smote2 = SMOTE(ratio='auto', kind='regular')
    smox2, smoy2 = smote.fit_sample(test_cols2, best_region.FRD_IND)
    smox2 = pd.DataFrame(smox2)
    smoy2 = pd.DataFrame(smoy2)
    syntheticdata2 = pd.concat((smox2,smoy2), axis=1)
    best_region=best_region[best_region['FRD_IND']==1]
    best_region.columns=col_list
    out_of_time_test=pd.concat([out_of_time_test,best_region],axis=0)
    print("iter")
       

    #testing the synthetic data sets on the trained logistic regression classifie
    
    
    
    
    
#Test update out of time list 
out_of_time_train=out_of_time_test.drop('FRD_IND',axis=1)    
mod_test2 = mod.predict(out_of_time_train)

cmfull=confusion_matrix(out_of_time_test['FRD_IND'],mod_test2)
fpr, tpr, thresholds = metrics.roc_curve(out_of_time_test['FRD_IND'], mod_test2, pos_label=2)
print("The FNR is:", cmfull[0][1])
print("The Outside of Time Sample AUC score is:", roc_auc_score(out_of_time_test['FRD_IND'],mod_test2 ))

    




