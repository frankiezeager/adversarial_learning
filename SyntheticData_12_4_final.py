# -*- coding: utf-8 -*-


#first filtering by Merchant Category Code and then associating each of the GMMs with a particular code
#do this for several of the codes

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
#import statsmodels.api as sm #package for logistic regression
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D
import itertools
from scipy import linalg
import matplotlib as mpl
from sklearn.mixture import GaussianMixture
from numpy import linspace,hstack
from pylab import plot,show,hist
import boto3


colnames = ['AUTH_ID','ACCT_ID_TOKEN','FRD_IND','ACCT_ACTVN_DT','ACCT_AVL_CASH_BEFORE_AMT','ACCT_AVL_MONEY_BEFORE_AMT','ACCT_CL_AMT','ACCT_CURR_BAL','ACCT_MULTICARD_IND','ACCT_OPEN_DT','ACCT_PROD_CD','ACCT_TYPE_CD','ADR_VFCN_FRMT_CD','ADR_VFCN_RESPNS_CD','APPRD_AUTHZN_CNT','APPRD_CASH_AUTHZN_CNT','ARQC_RSLT_CD','AUTHZN_ACCT_STAT_CD','AUTHZN_AMT','AUTHZN_CATG_CD','AUTHZN_CHAR_CD','AUTHZN_OPSET_ID','AUTHZN_ORIG_SRC_ID','AUTHZN_OUTSTD_AMT','AUTHZN_OUTSTD_CASH_AMT','AUTHZN_RQST_PROC_CD','AUTHZN_RQST_PROC_DT','AUTHZN_RQST_PROC_TM','AUTHZN_RQST_TYPE_CD','AUTHZN_TRMNL_PIN_CAPBLT_NUM','AVG_DLY_AUTHZN_AMT','CARD_VFCN_2_RESPNS_CD','CARD_VFCN_2_VLDTN_DUR','CARD_VFCN_MSMT_REAS_CD','CARD_VFCN_PRESNC_CD','CARD_VFCN_RESPNS_CD','CARD_VFCN2_VLDTN_CD','CDHLDR_PRES_CD','CRCY_CNVRSN_RT','ELCTR_CMRC_IND_CD','HOME_PHN_NUM_CHNG_DUR','HOTEL_STAY_CAR_RENTL_DUR','LAST_ADR_CHNG_DUR','LAST_PLSTC_RQST_REAS_CD','MRCH_CATG_CD','MRCH_CNTRY_CD','NEW_USER_ADDED_DUR','PHN_CHNG_SNC_APPN_IND','PIN_BLK_CD','PIN_VLDTN_IND','PLSTC_ACTVN_DT','PLSTC_ACTVN_REQD_IND','PLSTC_FRST_USE_TS','PLSTC_ISU_DUR','PLSTC_PREV_CURR_CD','PLSTC_RQST_TS','POS_COND_CD','POS_ENTRY_MTHD_CD','RCURG_AUTHZN_IND','RVRSL_IND','SENDR_RSIDNL_CNTRY_CD','SRC_CRCY_CD','SRC_CRCY_DCML_PSN_NUM','TRMNL_ATTNDNC_CD','TRMNL_CAPBLT_CD','TRMNL_CLASFN_CD','TRMNL_ID','TRMNL_PIN_CAPBLT_CD','DISTANCE_FROM_HOME']
df = pd.read_csv('/Users/frankiezeager/Documents/Graduate School/Capstone/training_part_10_of_10.txt', delimiter='|',header=None, names=colnames)

####### FOR AWS #########
#read data from s3
s3=boto3.client('s3')
obj=s3.get_object(Bucket='')
#combine together

#first sort everything by date
df1 = df1.sort_values(['AUTHZN_RQST_PROC_DT'])

df1['FRD_IND'].replace(['Y', 'N'], [1, 0], inplace=True)

df1['RCURG_AUTHZN_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
df1['RCURG_AUTHZN_IND'] = df1['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
df1['MRCH_CATG_CD'] = df1['MRCH_CATG_CD'].astype('category')
df1['POS_ENTRY_MTHD_CD'] = df1['POS_ENTRY_MTHD_CD'].astype('category')

#only select needed columns
t_ind = [2, 7, 14, 18, 23, 30, 44, 53, 57, 58, 68]

#cols = pd.DataFrame(j.iloc[:,t_ind])
df1 = pd.DataFrame(df1.iloc[:,t_ind])

df1_MRCHCODE = pd.get_dummies(df1['MRCH_CATG_CD']) #converting to dummy variables
df1_POSENTRY = pd.get_dummies(df1['POS_ENTRY_MTHD_CD']) #converting to dummy variables

#join the dummy variables to the main data frame
df1 = pd.concat([df1, df1_MRCHCODE], axis=1)

#join the dummy variables to the main data frame
df1 = pd.concat([df1, df1_POSENTRY], axis=1)

#drop the original categorical variables (MRCH_CATG_CD) and (POS_ENTRY_MTHD_CD) and (RCURG_AUTHZN_IND)
df1.drop(['MRCH_CATG_CD', 'POS_ENTRY_MTHD_CD', 'RCURG_AUTHZN_IND'],inplace=True,axis=1)


fraud_list=df1['FRD_IND']
df1=df1.drop('FRD_IND',axis=1)
df1['FRD_IND']=fraud_list

#df1 = df1.sort_values(['AUTHZN_RQST_PROC_DT'])
#df1=df1.drop('AUTHZN_RQST_PROC_DT',axis=1)
#create an out-of-time final test set



#############################################################

#df2, out_of_time_test = train_test_split(df1, train_size = 0.9)

col_list=df1.columns.values.tolist()

#creation of folds
fold_size = math.floor(df1.shape[0]/3)
folds = []
for f, F_t in df1.groupby(np.arange(len(df1)) // fold_size):
    if (len(F_t)==fold_size):
        folds.append(F_t)

iteration_num = 0

listFNR=[]
#split each fold into training and testing sets
best_strat_list=[]
AUC_list=[]
model_list=[]

iter_num = 1

colors = cycle(['cyan', 'indigo', 'seagreen'])
lw = 2

allsynthetic_sets = []


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

#####output coverage curve csv (predicted probabilities)######################
    prob_test=mod.predict_proba(testcol)
    prob_test=pd.DataFrame(prob_test)
    #real truth values
    prob_target=test.FRD_IND
    #predicted truth values
    prob_pred_truth=pd.DataFrame(mod_test)
    prob_df=pd.concat([prob_test.reset_index(),prob_target.reset_index(),prob_pred_truth.reset_index()],axis=1)
    prob_col_list=['index1','prob_0','prob_1','index2','truth_val','index3','pred_truth_val']
    prob_df.columns=prob_col_list
    prob_df=prob_df.drop(['index1','index2','index3'],axis=1)
    path='/Users/frankiezeager/Documents/Graduate School/Capstone/' #set path
    prob_df.to_csv(path+'both_learn_coverage_'+str(iteration_num)+'.csv')
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
    strat_ind = [0, 2, 3, 5, 6]
    #strat_ind = [18,53,68,7,23] 
    strategy_df= pd.DataFrame(test.iloc[:,strat_ind]) #incorrect subsetting of dataframe, did not include these in the original df!!!


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
        cols = j.drop(j.columns[[529, 530]], axis=1)
        #cols=j.iloc[:,:-1] 
        #cols=cols.iloc[:,:-1]
        col_response = j.iloc[:,-2]
        ###############################ERROR: #too many columns#########
        pred = mod.predict(cols)
        cm = confusion_matrix(col_response, pred)
        FNR = cm[0][1]
        fn_rate.append(FNR)

    best_strat = fn_rate.index(max(fn_rate))
    best_strat_list.append(best_strat)
    best_fold = all_batches[best_strat]

    #Implement SMOTE
    #test_cols = test.drop("Class", axis = 1)
    test_cols = best_fold.drop(best_fold.columns[[529, 530]],axis=1)
    #test_cols = test_cols.values
    if (iteration_num+1)<len(folds):
        fraud_next_fold=sum(folds[iteration_num+1].FRD_IND==1)
        new_fraud_fold=fold_size*.002 #specified next fold fraud ratio
        fraud_needed=math.ceil(new_fraud_fold-fraud_next_fold)
    smote = SMOTE(ratio=0.5, kind='regular')
    smox, smoy = smote.fit_sample(test_cols, best_fold.FRD_IND)
    smox = pd.DataFrame(smox)
    smoy = pd.DataFrame(smoy)
    syntheticdata = pd.concat((smox,smoy), axis=1)
    syntheticdata.columns=col_list
    
    #take all of the synthetic data as an out-of-time sample
    allsynthetic_sets.append(syntheticdata)

    #take all the fraudulent transactions of best strategy batch and add it to the next fold
    fraud_trans = syntheticdata[syntheticdata.iloc[:,-1] == 1]
    #append the fraud transactions to fold+1
    if (iteration_num+1)<len(folds):
        fraud_trans = fraud_trans.sample(n=fraud_needed)
        folds[iteration_num+1]=pd.concat([folds[iteration_num+1], fraud_trans], axis=0)
        iteration_num = iteration_num+ 1
        print(iteration_num)



#plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#plt.xlim([-0.05, 1.05])
#plt.ylim([-0.05, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Test')
#plt.legend(loc="lower right")
#plt.show()


#having three out-of-time sets from synthetic data
i_num = 1
for l, color, z in zip(allsynthetic_sets, colors, model_list):
    syntheticdata_test=l.drop('FRD_IND',axis=1)
    #mod_test2 = z.predict(syntheticdata_test)
    mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
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
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i_num, aucscore))
    i_num += 1

#adding ROC curve code
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Test')
plt.legend(loc="lower right")
plt.show()

############################################ MODEL STAYS THE SAME #######################################
firstmod = model_list[0] #original model
i_num = 1
for l, color in zip(allsynthetic_sets, colors):
    syntheticdata_test=l.drop('FRD_IND',axis=1)
    #mod_test2 = z.predict(syntheticdata_test)
    mod_test3 = firstmod.predict_proba(syntheticdata_test)[:,1]
    #cmfull=confusion_matrix(l['FRD_IND'],mod_test2)
    fpr, tpr, _ = roc_curve(l['FRD_IND'], mod_test3)
    #fpr, tpr, thresholds = roc_curve(l['FRD_IND'], mod_test3, pos_label=2)
    #print("The FNR is:", cmfull[0][1])
    print("The Outside of Time Sample AUC score is:", roc_auc_score(l['FRD_IND'],mod_test3 ))
    #aucscore = roc_auc_score(l['FRD_IND'],mod_test2 )
    #getting predicted probablilites for fraud
    #mod_test3 = z.predict_proba(syntheticdata_test)[:,1]
    #fpr1, tpr1, _ = roc_curve(l['FRD_IND'], mod_test3)
    aucscore = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color, label='ROC fold %d (area = %0.2f)' % (i_num, aucscore))
    i_num += 1

#adding ROC curve code
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Test')
plt.legend(loc="lower right")
plt.show()


