import sys, os
import pandas as pd
import matplotlib as plt
import numpy as np

#insert your path where you saved the csv files from the SyntheticData.py
path="/Applications/Graduate School/Fall 2016/Capstone/code/"
os.chdir(path)

names=[]

#for all the folds in synthetic data (printed in different csvs)
for file in list(ls ('[0-9]*.csv')):
    name='probabilities'+str(file)
    #read csvs
    name=pd.read_csv(path+file)
    #sort by descending probability of being fraud
    name.sort_values(['prob_1'],ascending=False)
    #find the percent of fraud caught
    name.pct_fraud_caught=name.
    #plot thecoverage curve
    #plt.plot(name.'prob_1')
    #names.append(name)


