#Feature Engineering using the Merchant Category Code Variable

library(RcppRoll)
library(zoo)
library(lubridate)
library(data.table)
library(dplyr)
library(plyr)

df <- fread('training_part_10_of_10.txt')
df1 <- subset(df, select=c('V27', 'V2', 'V19', 'V45'))
colnames(df1) <- c('AUTHZN_RQST_PROC_DT', 'ACCT_ID_TOKEN', 'AUTHZN_AMT', 'MRCH_CATG_CD')
df1$AUTHZN_RQST_PROC_DT <- as.Date(df1$AUTHZN_RQST_PROC_DT) #convert the character to a date
df2 <- subset(df, select=c('V27', 'V2', 'V19', 'V45')) #authorization request date, the authorization amount, etc.
df2<-df2[sample(nrow(df2),1000000),]
colnames(df2) <- c('AUTHZN_RQST_PROC_DT', 'ACCT_ID_TOKEN', 'AUTHZN_AMT', 'MRCH_CATG_CD')
sapply(df2, typeof) #checking the type of column
df2$AUTHZN_RQST_PROC_DT <- as.Date(df2$AUTHZN_RQST_PROC_DT) #convert the character to a date

df2$AUTHZN_RQST_PROC_DT <- sort(df2$AUTHZN_RQST_PROC_DT, decreasing = FALSE) #sorting the date column in ascending fashion

merchant_category_code <- unique(df2$MRCH_CATG_CD) #getting the unique merchant category codes, ex: 5542

#calculate probability of fraud by MCC
#processing data frame
percent_fraud <- ddply(df2, )













fuel_code <- df2[df2$MRCH_CATG_CD == "5542"] #looking for 5542 (Gas stations)

#Convert your data frame to data table and set column V1 as key.
fuel_code <-data.table(fuel_code,key='AUTHZN_RQST_PROC_DT')
#remove the $1 amounts from the rows
fuel_code <- fuel_code[fuel_code$AUTHZN_AMT != 1.0]
#Calculate mean for each column .SD means subset of your data table <- daily frequency
fuel_avg <- fuel_code[,lapply(.SD,mean),by=list(AUTHZN_RQST_PROC_DT, MRCH_CATG_CD, ACCT_ID_TOKEN)] #grouping by the the date and the merchant category code
#counting the daily frequency for the gas station
fuel_freq <- fuel_code[,lapply(.SD,length),by=list(AUTHZN_RQST_PROC_DT, MRCH_CATG_CD, ACCT_ID_TOKEN)]
#average transaction amount/day for a gas station
fuel_agg <- fuel_code[,lapply(.SD,sum),by=list(AUTHZN_RQST_PROC_DT, MRCH_CATG_CD, ACCT_ID_TOKEN)]

#WEEKLY FREQUENCY
#tapply(fuel_agg$V19, week(fuel_agg$V27), mean)
fuel_freq$Week <- as.Date(cut(fuel_freq$AUTHZN_RQST_PROC_DT,breaks = "week",start.on.monday = FALSE))
weekly.frequency <- aggregate(data.frame(frequency = fuel_freq$AUTHZN_AMT), list(week = fuel_freq$Week), sum)
weekly.frequency$week <- as.Date(weekly.frequency$week)

#MONTHLY FREQUENCY
fuel_freq$Month <- as.Date(cut(fuel_freq$AUTHZN_RQST_PROC_DT,breaks = "month",start.on.monday = FALSE))
monthly.frequency <- aggregate(data.frame(frequency = fuel_freq$AUTHZN_AMT), list(month = fuel_freq$Month), sum)

#QUARTER FREQUENCY
fuel_freq$Quarter <- as.Date(cut(fuel_freq$AUTHZN_RQST_PROC_DT,breaks = "quarter",start.on.monday = FALSE))
quarter.frequency <- aggregate(data.frame(frequency = fuel_freq$AUTHZN_AMT), list(month = fuel_freq$Quarter), sum)

#AVERAGE TRANSACTION AMOUNT
#Weekly average transaction amount
fuel_agg$Week <- as.Date(cut(fuel_agg$AUTHZN_RQST_PROC_DT,breaks = "week",start.on.monday = FALSE))
weekly.amount <- aggregate(data.frame(avg_amount = fuel_agg$AUTHZN_AMT), list(week = fuel_agg$Week), mean)

#Monthly average transaction amount
fuel_agg$Month <- as.Date(cut(fuel_agg$AUTHZN_RQST_PROC_DT,breaks = "month",start.on.monday = FALSE))
monthly.amount <- aggregate(data.frame(avg_amount = fuel_agg$AUTHZN_AMT), list(month = fuel_agg$Month), mean)

#quarterly average transaction amount
fuel_agg$Quarter <- as.Date(cut(fuel_agg$AUTHZN_RQST_PROC_DT,breaks = "quarter",start.on.monday = FALSE))
quarter.amount <- aggregate(data.frame(avg_amount = fuel_agg$AUTHZN_AMT), list(month = fuel_agg$Quarter), mean)

# Merchant Category Codes (MCC) Of Interest
# Automated Fuel Dispenser: 5542
# Grocery stores: 5411
# Passenger Railways: 4112
# Drugs, Drug Proprietors, Druggist's Sundries: 5122
# Package Stores (Beer, Wine, Liquor): 5921
# Electronic Sales: 5732

#################ROLLING WINDOW CALCULATIONS##################
df3 <- subset(df, select=c('V27', 'V2', 'V19', 'V45'))
colnames(df3) <- c('AUTHZN_RQST_PROC_DT', 'ACCT_ID_TOKEN', 'AUTHZN_AMT', 'MRCH_CATG_CD') #need to add the Account ID Token 
df3$AUTHZN_RQST_PROC_DT <- as.Date(df3$AUTHZN_RQST_PROC_DT) #convert the character to a date
#setkey(df3, "AUTHZN_RQST_PROC_DT", "MRCH_CATG_CD")

df4 <- df3

#Number merchant type over month - Total number of transactions with same merchant during past 30 days
#daily_merch_freq <- df3[,lapply(.SD,length),by=list(AUTHZN_RQST_PROC_DT, MRCH_CATG_CD)]

#daily_merch_freq[, Roll.Tot.Amt := roll_sumr(AUTHZN_AMT, 30), by=list(ACCT_ID_TOKEN, MRCH_CATG_CD)]

#try using zoo package

df4[, Roll.Tot.Amt := roll_meanr(AUTHZN_AMT, 30, partial=TRUE), by=c('ACCT_ID_TOKEN', 'MRCH_CATG_CD')]

#Amount merchant type over month - Average amount per day spent over a 30 day period on all transactions up to this one on the same merchant type as this transaction
daily_avg_amt <- df3[,lapply(.SD,sum),by=list(AUTHZN_RQST_PROC_DT, MRCH_CATG_CD)]

daily_avg_amt[, Roll.Avg.Amt := roll_meanr(AUTHZN_AMT, 30), by=list(ACCT_ID_TOKEN, MRCH_CATG_CD)]


####Next steps
#create other derived/velocity attributes such as:
#looking at different time frames for number of transactions and authorization amount filtering by MCC
#try to create variables looking at the timestamp of the authorization request (column V28, AUTHZN_RQST_PROC_TM)
#look at country code of merchant (http://www.web-merchant.co.uk/select/countries.html)
#category type of authorization (AUTHZN_CATG_CD)

#get the means for these variables

