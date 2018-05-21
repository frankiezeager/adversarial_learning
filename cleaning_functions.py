def initial_clean(df):
    """
    A function to clean and prepare the data for the adversarial learning algorithm
        organized by date, a label of 1/0 for fraud, and removing NAs

    Input: one data frame of data with fraud/not fraud labels
    Output: cleaned data frame, sorted by date with relevant columns
    """

    #make sure the df sorted by date
    df = df.sort_values(['AUTHZN_RQST_PROC_DT'])

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

    #subset relevant variables
    df = pd.DataFrame(df.iloc[:,t_ind])

    #take out any NaN's from the data set
    df=df.dropna(axis=0)

    #convert these two columns from Y and N to 1 and 0
    df['FRD_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    df['RCURG_AUTHZN_IND'].replace(['Y', 'N'], [1, 0], inplace=True)
    fraud_list=df['FRD_IND']
    df=df.drop('FRD_IND',axis=1)
    df['FRD_IND']=fraud_list

    #make a list of columns to use later
    col_list=df.columns.values.tolist()
   #take out any NaN's from the data set
    df=df.dropna(axis=0)

    #convert column type to numeric
    df['RCURG_AUTHZN_IND'] = df['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
    df['FRD_IND'] = df['FRD_IND'].convert_objects(convert_numeric=True)

    return df

def prepare_train_test_validate(df):
        """A function to create the train test and validation sets for the model
        Input: data frame with added fraud
        Output: train, test, and an out of time validation set
        """
        train,intermediate_set=train_test_split(df,train_size=.6,test_size=.4,random_state=1575)
        test, out_of_time=train_test_split(intermediate_set, train_size=.5,test_size=.5,random_state=1575)

        #delete intermediate set
        del intermediate_set


         #filling any NAs  (slicing the data frame sometimes inputs NAs)
        train=train.fillna(method='ffill')
        test=test.fillna(method='ffill')
        out_of_time=out_of_time.fillna(method='ffill')

        #convert out of time data frame variables
        out_of_time.FRD_IND.map(dict(Y=1,N=0))
        out_of_time.RCURG_AUTHZN_IND.map(dict(Y=1,N=0))

        #filling any NAs
        out_of_time=out_of_time.fillna(method='ffill')


        #convert column type to numeric
        out_of_time['RCURG_AUTHZN_IND'] = out_of_time['RCURG_AUTHZN_IND'].convert_objects(convert_numeric=True)
        out_of_time['FRD_IND'] = out_of_time['FRD_IND'].convert_objects(convert_numeric=True)

    #training the classifier (using the following variables)
        #7-ACCT_CUR_BAL
        #14-Total Num. Authorizations
        #30-Avg. Daily Authorization Amount
        #44-Merchant Category ID (convert to factor)
        #57- Point of Sale Entry Method (convert to factor)
        #58 - Recurring Charge (Relabel as 0/1)
        #68 - Distance from Home

        train_cols = train.drop('FRD_IND', axis=1) #columns for training

        testcol = test.drop('FRD_IND',axis=1)
        testcol.fillna(method='bfill',inplace=True)

        return train, train_cols, test, testcol, out_of_time
