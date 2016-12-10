import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

#insert your path where you saved the csv files from the SyntheticData.py
path="/Applications/Graduate School/Fall 2016/Capstone/code/"
os.chdir(path)

names=[]

#for all the folds in synthetic data (printed in different csvs)
csv_list=glob.glob(path+'both_learn_coverage_*.csv')

df_list=[]
#note: this will take a long time
for file in csv_list:
    #read csvs
    name=pd.read_csv(file)
    #sort by descending probability of being fraud
    name.sort_values(['prob_1'])
    #find the percent of fraud caught
    for i in name.truth_val:
        if np.cumsum(i) !=0:
            name['pct_fraud_caught']=(np.cumsum(name.pred_truth_val))/i
        else:
            name['pct_fraud_caught']=1
    name.head(10)
    name.tail(10)
    df_list.append(name)
    #replace infs with 1 (100%) (since this indicates 0 fraud in total)
    #name.replace([np.inf, -np.inf], 1,inplace=True)
    #name.pct_fraud_caught.fillna(1)
    #plot the coverage curve
    
for name in df_list:
    plt.plot(name.prob_1,name.pct_fraud_caught)
    plt.ylabel('Percentage of Fraud Caught')
    plt.xlabel('Score Band (Most Risky to Least Risky)')
    plt.show()
    


