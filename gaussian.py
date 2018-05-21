
def find_and_assign_strat(test,n_components):
    """ A function to find strategies using a Gaussian Mixture Model and assign a strategy to each transaction
    Input: Test dataset, number of components for the GMM
    Output: Test dataset that includes a strategy assignment for each row
    """
    #subset df to include only pertinent (adversarial-controlled) continuous variables
    #account current balance, authorization amount, authorization outstanding amount
    #and plastic issue duration

    strat_ind =[0,2,3,6]
    strategy_df= pd.DataFrame(test.iloc[:,strat_ind])


    # Fit a 6 component Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,covariance_type='full')
    gmm.fit(strategy_df)

    #assign each transaction a strategy
    strat_assign=gmm.predict(strategy_df)

    #attach back to data frame
    test['Strategy Number'] = strat_assign

    return test

def find_best_strat(all_batches):
    """
    A function to find the best (the one with the highest false negative rate)
    strategies from the individual batches separated by strategy

    Input: All_batches: a list of data frames organized by strategy
    Output: best_fold: data that utilized the best strategy
    """

    fn_rate = []

    #find batch with highest false negative rate (this will be our best strategy)
    for j in all_batches:
        cols = j.drop(labels='Strategy Number',axis=1)
        cols = cols.drop(labels='FRD_IND',axis=1)
        cols.fillna(method='bfill',inplace=True)
        col_response = j.iloc[:,-2]
        pred = mod.predict(cols)
        cm = confusion_matrix(col_response, pred)
        FNR = cm[0][1]
        #append false negative rate to a list of false negatives
        fn_rate.append(FNR)
    #find strategy with best false negative rate
    best_strat = fn_rate.index(max(fn_rate))
    #append this df to the best strategy list of data frames
    best_strat_list.append(best_strat)
    print("The Best Strategy is: ", best_strat)
    best_fold = all_batches[best_strat]

    return best_fold

def implement_smote(best_fold):
    """ A function to implement smote on the data that utilizes the best strategy
    Input: best_fold : data that utilizes the best strategy
    Output: syntheticdata: synthesized data that is similar to the best strategy
    """
    test_cols = best_fold.drop(labels='Strategy Number',axis=1)
    test_cols = test_cols.drop(labels='FRD_IND',axis=1)
    smote = SMOTE(ratio='auto', kind='regular')
    smox, smoy = smote.fit_sample(test_cols, best_fold.FRD_IND)
    smox = pd.DataFrame(smox)
    smoy = pd.DataFrame(smoy)
    syntheticdata = pd.concat((smox,smoy), axis=1)
    syntheticdata.columns=col_list
    
    return syntheticdata, col_list
