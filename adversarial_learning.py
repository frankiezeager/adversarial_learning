# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.externals import joblib
from cleaning_functions import initial_clean, prepare_train_test_validate
from gaussain import find_and_assign_strat,find_best_strat,implement_smote
from add_fraud import add_fraud
#this is only needed to output everything locally on AWS
plt.switch_backend('agg')

#set column names
colnames = ['AUTH_ID','ACCT_ID_TOKEN','FRD_IND','ACCT_ACTVN_DT','ACCT_AVL_CASH_BEFORE_AMT','ACCT_AVL_MONEY_BEFORE_AMT','ACCT_CL_AMT','ACCT_CURR_BAL','ACCT_MULTICARD_IND','ACCT_OPEN_DT','ACCT_PROD_CD','ACCT_TYPE_CD','ADR_VFCN_FRMT_CD','ADR_VFCN_RESPNS_CD','APPRD_AUTHZN_CNT','APPRD_CASH_AUTHZN_CNT','ARQC_RSLT_CD','AUTHZN_ACCT_STAT_CD','AUTHZN_AMT','AUTHZN_CATG_CD','AUTHZN_CHAR_CD','AUTHZN_OPSET_ID','AUTHZN_ORIG_SRC_ID','AUTHZN_OUTSTD_AMT','AUTHZN_OUTSTD_CASH_AMT','AUTHZN_RQST_PROC_CD','AUTHZN_RQST_PROC_DT','AUTHZN_RQST_PROC_TM','AUTHZN_RQST_TYPE_CD','AUTHZN_TRMNL_PIN_CAPBLT_NUM','AVG_DLY_AUTHZN_AMT','CARD_VFCN_2_RESPNS_CD','CARD_VFCN_2_VLDTN_DUR','CARD_VFCN_MSMT_REAS_CD','CARD_VFCN_PRESNC_CD','CARD_VFCN_RESPNS_CD','CARD_VFCN2_VLDTN_CD','CDHLDR_PRES_CD','CRCY_CNVRSN_RT','ELCTR_CMRC_IND_CD','HOME_PHN_NUM_CHNG_DUR','HOTEL_STAY_CAR_RENTL_DUR','LAST_ADR_CHNG_DUR','LAST_PLSTC_RQST_REAS_CD','MRCH_CATG_CD','MRCH_CNTRY_CD','NEW_USER_ADDED_DUR','PHN_CHNG_SNC_APPN_IND','PIN_BLK_CD','PIN_VLDTN_IND','PLSTC_ACTVN_DT','PLSTC_ACTVN_REQD_IND','PLSTC_FRST_USE_TS','PLSTC_ISU_DUR','PLSTC_PREV_CURR_CD','PLSTC_RQST_TS','POS_COND_CD','POS_ENTRY_MTHD_CD','RCURG_AUTHZN_IND','RVRSL_IND','SENDR_RSIDNL_CNTRY_CD','SRC_CRCY_CD','SRC_CRCY_DCML_PSN_NUM','TRMNL_ATTNDNC_CD','TRMNL_CAPBLT_CD','TRMNL_CLASFN_CD','TRMNL_ID','TRMNL_PIN_CAPBLT_CD','DISTANCE_FROM_HOME']

random.seed(1445)

#initialize lists
iteration_num = 0
listFNR=[]
best_strat_list=[]
AUC_list=[]
model_list=[]
iter_num = 1
allsynthetic_sets = []
all_fold_oot=[]

for i in range(1,11): #number of total files
    #read in the data file
    df1=pd.read_csv('training_part_0{}_of_10.txt'.format(i),delimiter='|',header=None, names=colnames)
    df1=initial_clean(df1) #clean data frame

    if i!=1:
        df1=add_fraud(df1,syntheticdata) #add the 'best fraud' from the previous round of the game

    #split into training, 'testing' (finding the adversarial best strategy data frame), out of time (validation set)
    #60% training, 20% test, 20% validation
    train,train_cols,test,testcol, out_of_time=prepare_train_test_validate(df1)
    all_fold_oot.append(out_of_time)

    #Logistic Regression
    logit = LogisticRegression(class_weight='balanced')

    mod = logit.fit(train_cols, train['FRD_IND'])

    #keep the models in a list to access later
    model_list.append(mod)

    #use the model to predict
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
    test=find_and_assign_strat(test,6) #find 6 strategies for this dataset using GMM, and assign each row a strategy

    all_batches = []

    for t, B_t in test.groupby('Strategy Number'):
        all_batches.append(B_t) #group strategies into batches

    best_fold=find_best_strat(all_batches)

    #Implement SMOTE (add 'good' fraud to the dataset)
    syntheticdata,col_list=implement_smote(best_fold)



#output files to be used in curves.py
val=1
for file in all_fold_oot:
    file.to_csv('adv_learning_test_{}.csv'.format(val))
    val=val+1

#output models to be used in curves.py
joblib.dump(model_list, 'adv_learning_models.pkl',compress=True)
