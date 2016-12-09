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
for file in csv_list:
    name='probabilities'+str(file)
    #read csvs
    name=pd.read_csv(file)
    #sort by descending probability of being fraud
    name.sort_values(['prob_1'],ascending=False)
    #find the percent of fraud caught
    name.pct_fraud_caught=(np.cumsum(name.pred_truth_val))/(np.cumsum(name.truth_val))
    print(name.pct_fraud_caught)
    #plot the coverage curve
    #plt.plot(name.prob_1,name.pct_fraud_caught)
    #plt.ylabel('Percentage of Fraud Caught')
    #plt.xlabel('Score Band (Most Risky to Least Risky)')
    #plt.show()
    #names.append(name)


