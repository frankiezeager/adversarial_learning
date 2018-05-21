def add_fraud(df,syntheticdata):
    """A function to add fraudulent data that utilizes the best strategy calculated from the last round of the game
    Input: cleaned data frame and the fraudulent data from the previous round of the game
    Output: a data frame with 5 percent fraud
    """
#find percent fraud in current df
    df_pct_fraud=sum(df.FRD_IND==1)/len(df)

    #find difference between this and the 5% fraud desired
    pct_fraud_needed=.05-df_pct_fraud

    #find number of fraud transactions needed
    num_fraud_trans=math.floor(pct_fraud_needed*len(df))

    #finding the fraudulent transactions in synthetic data
    fraud_trans = syntheticdata[syntheticdata.iloc[:,-1] == 1]

    #sampling the fraud transactions to include amount needed
    add_fraud = fraud_trans.sample(n=num_fraud_trans, replace=True,random_state=1575)

    #adding fraud transactions back to df
    df=pd.concat([df,add_fraud],axis=0)

    return df
