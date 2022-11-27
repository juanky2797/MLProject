
#import libraries

import datetime

import matplotlib
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import random
import seaborn as sns



pd.set_option('display.max_columns', None)

#read dataset
df = pd.read_csv('data/application_including_boundary_data.csv',encoding='latin-1', low_memory=False)

#When we see some columns of the data set we can see that there are columns that are not very useful to use as features


#So we can drop these columns in the following way



#Above, we defined a list that contains the names of all the columns we want to drop. Next, we call the drop() function on our object passing in the inplace parameter as True and the axis parameter as 1. This tells Pandas that we want the changes to be made directly in our object and that it should look for the values to be dropped in the columns of the object.


#print(df['OBJECTID'].is_unique)


#So far, we have removed unnecessary columns and changed the index of our DataFrame to something more sensible. In this section, we will clean specific columns and get them to a uniform format to get a better understanding of the dataset and enforce consistency.



#print(df.loc[50548])
to_drop = ['ApplicantForename','ApplicantSurname','ApplicantAddress','OBJECTID', 'AppealRefNumber', 'ApplicationNumber',
           'DevelopmentDescription','GrantDate','ExpiryDate','AppealStatus','AppealDecision','AppealDecision',
           'AppealDecision', 'FIRequestDate','FIRecDate','LinkAppDetails','ETL_DATE','SiteId','ORIG_FID',
           'DecisionDueDate','WithdrawnDate','DecisionDate','AppealSubmittedDate','AppealDecisionDate','ITMEasting','ITMEasting','DevelopmentPostcode','ApplicationStatus','ApplicationType','ITMNorthing','LandUseCode','Cities_and_Suburbs']

df_dropped = df.drop(to_drop,axis=1)

df_dropped = df_dropped[df_dropped.ReceivedDate != '3185/10/01 00:00:00+00'] # this entry throws an error when we convert to datetime.

# creating a datetime column
df_dropped['date']=pd.to_datetime(df_dropped['ReceivedDate'], format='%Y/%m/%d %H:%M:%S+00')

# removing unnecessary receivedDate column
#df_dropped = df_dropped.drop(['data'],axis=1)

# filter by dates
df_filtered = df_dropped[(df_dropped['date'] > '2017-01-01') & (df_dropped['date'] < '2022-10-01')]

# Replacing nan values with 'BLANK' string (i can't seem to get it to otherwise remove the nan values)
#df_filtered[['Decision']] = df_filtered[['Decision']].replace(float('nan'),'BLANK')
#print(df_filtered.shape)
df_filtered[['Metropolitian_Areas']] = df_filtered[['Metropolitian_Areas']].replace('character(0)',np.nan)
df_filtered[['Key_Growth_Areas']] = df_filtered[['Key_Growth_Areas']].replace('character(0)',np.nan)
#df_filtered[['Cities_and_Suburbs']] = df_filtered[['Cities_and_Suburbs']].replace('character(0)',np.nan)




#remove NaN and null values
df_filtered = df_filtered.dropna(subset=['Decision','AreaofSite','NumResidentialUnits','OneOffHouse','FloorArea','OneOffKPI','Metropolitian_Areas','Key_Growth_Areas',])
#print(df_filtered.shape)


#print(df_filtered)

# Removing rows with decison values that are in conclusive
df_filtered = df_filtered[df_filtered['Decision'] != 'BLANK']
df_filtered = df_filtered[df_filtered['Decision'] != 'DECLARED EXEMPT']
df_filtered = df_filtered[df_filtered['Decision'] != 'DECLARED NOT EXEMPT']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT PERMISSION & REFUSE PERMISSION    ']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT PERMISSION & REFUSE PERMISSION']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT PERMISSION & REFUSE PERMISSION              ']
df_filtered = df_filtered[df_filtered['Decision'] != 'REQUEST AI EXT OF TIME']
df_filtered = df_filtered[df_filtered['Decision'] != 'REFUSE PERMISSION & GRANT RETENTION               ']
df_filtered = df_filtered[df_filtered['Decision'] != 'Grant Retention & Refuse Retention                ']
df_filtered = df_filtered[df_filtered['Decision'] != 'DECLARED EXEMPT & DECLARED NOT EXEMPT']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT RETENTION & REFUSE PERMISSION']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT TIME EXT (3 months) AI                      ']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT PERMISSION & REFUSE RETENTION']
df_filtered = df_filtered[df_filtered['Decision'] != 'Grant Permission & Refuse Retention               ']
df_filtered = df_filtered[df_filtered['Decision'] != 'REPORT RETURNED TO AN BORD PLEANALA']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT RETENTION PERM & REFUSE RETENTION ']
df_filtered = df_filtered[df_filtered['Decision'] != 'Ext Dur no ai recd 1mth WD                        ']
df_filtered = df_filtered[df_filtered['Decision'] != 'IS Exempted Development                           ']
df_filtered = df_filtered[df_filtered['Decision'] != 'PAC REPORT & FILE CLOSED                ']
df_filtered = df_filtered[df_filtered['Decision'] != 'Precluded under 34 (12)(b) from Making a Decision ']
df_filtered = df_filtered[df_filtered['Decision'] != 'Referred to An Bord Pleanala for Determination']
df_filtered = df_filtered[df_filtered['Decision'] != 'Returned Application under Section 37(5)          ']
df_filtered = df_filtered[df_filtered['Decision'] != 'REPORT RETURNED TO AN BORD PLEANALA']
df_filtered = df_filtered[df_filtered['Decision'] != 'Grant Perm & Retention and Refuse Retent          ']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT RETENTION & REFUSE RETENTION']
df_filtered = df_filtered[df_filtered['Decision'] != 'GRANT PERMISSION & GRANT OUTLINE PERM.          ']
df_filtered = df_filtered[df_filtered['Decision'] != 'SPLIT DECISION(PERMISSION & REFUSAL)']
df_filtered = df_filtered[df_filtered['Decision'] != 'Split decision']
df_filtered = df_filtered[df_filtered['Decision'] != 'SPLIT DECISION(RETENTION PERMISSION)']

df_filtered[['Decision']] = df_filtered[['Decision']].replace('APPLICATION WITHDRAWN','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('WITHDRAWN ARTICLE 33 (NO SUB)','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('WITHDRAW THE APPLICATION','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARED WITHDRAWN','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('WITHDRAW THE APPLICATION          ','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('App withdrawn as no AI recd in 6 months           ','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE APPLICATION WITHDRAWN           ','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('WITHDRAWN ARTICLE 33 (SUBSECTION 4)','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('WITHDRAWN AT ABP STAGE','Withdrawn')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('App. withdrawn as no C.AI recd in 6 mths          ','Withdrawn')

df_filtered[['Decision']] = df_filtered[['Decision']].replace('ADDITIONAL INFORMATION','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REQUEST ADDITIONAL INFORMATION                    ','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REQUEST ADDITIONAL INFORMATION','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REQUEST ADDITIONAL INFORMATION          ','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CLARIFICATION OF ADDITIONAL INFORMATION','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.            ','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CLARIFICATION OF FURTHER INFORMATION    ','request additional information')

df_filtered[['Decision']] = df_filtered[['Decision']].replace('ADDITIONAL INFORMATION','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REQUEST ADDITIONAL INFORMATION                    ','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REQUEST ADDITIONAL INFORMATION','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REQUEST ADDITIONAL INFORMATION          ','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CLARIFICATION OF ADDITIONAL INFORMATION','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.            ','request additional information')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CLARIFICATION OF FURTHER INFORMATION    ','request additional information')

df_filtered[['Decision']] = df_filtered[['Decision']].replace('APPLICATION DECLARED INVALID','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID APPLICATION DUE TO SITE NOTICE            ','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID APPLICATION','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID - SITE NOTICE','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE INVALID (SITE NOTICE)           ','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE INVALID (SITE NOTICE)','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE APPLICATION INVALID             ','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CANNOT DETERMINE','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE APPLICATION INVALID','invalid')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Invalid Application (site notice)','invalid')

df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSED','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION                                 ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION                       ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('R','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE RETENTION PERMISSION','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION FOR RETENTION                   ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION FOR RETENTION','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION FOR RETENTION         ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION & REFUSE RETENTION              ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION & REFUSE RETENTION','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE FURTHER EXT. OF DURATION OF PERM.','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE OUTLINE PERMISSION                         ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE OUTLINE PERMISSION               ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE EXT. OF DURATION OF PERMISSION   ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE 3 MONTH EXTENSION                          ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE APPROVAL','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERM. CONSEQUENT TO OUTLINE PERM.          ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('PART 8 REJECTED BY COUNCIL','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE CERTIFICATE of EXEMPTION','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION & REFUSE OUTLINE PERM.','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE PERMISSION & REFUSE OUTLINE PERM.          ','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE EXT. OF DURATION OF PERMISSION','Refused')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('REFUSE OUTLINE PERMISSION','Refused')

df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION                                  ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION                        ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT RETENTION PERMISSION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION AND RETENTION PERMISSION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION FOR RETENTION                    ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION FOR RETENTION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION & GRANT RETENTION                ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION FOR RETENTION          ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION & GRANT RETENTION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT EXTENSION OF DURATION OF PERMISSION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT EXTENSION OF DURATION OF PERM.    ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERMISSION & GRANT RETENTION      ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT EXTENSION OF DURATION OF PERM.              ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('PART 8 APPROVED BY COUNCIL','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT OUTLINE PERMISSION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('APPROVAL FOR THE PROPOSAL TO PROCEED    ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECISION QUASHED BY HIGH COURT','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT APPROVAL','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT OUTDOOR EVENT LICENCE','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT OUTLINE PERM. & REFUSE OUTLINE PERM.','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PER. REFU. PERM. GRAT. RETENTION            ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Grant Perm. & Grant Perm. conq. to O.P.           ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT OUTLINE PERMISSION                          ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT PERM. CONSEQUENT TO OUTLINE PERM.           ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT FURTHER EXT. OF DURATION OF PERMISSION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('GRANT OUTLINE PERMISSION                ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CONDITIONAL','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Conditional Permission','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Granted (Conditional)','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('C','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('UNCONDITIONAL','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Unconditional Permission','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Granted (Unconditional)','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('APPLICATION DECLARED INVALID','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID APPLICATION DUE TO SITE NOTICE            ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID APPLICATION','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID - SITE NOTICE','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE INVALID (SITE NOTICE)           ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('INVALID PLANNING APPLICATION                      ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE INVALID (SITE NOTICE)','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE APPLICATION INVALID             ','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('CANNOT DETERMINE','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('DECLARE APPLICATION INVALID','Granted')
df_filtered[['Decision']] = df_filtered[['Decision']].replace('Invalid Application (site notice)','Granted')

#print(df_filtered.isnull().sum())


print(df_filtered.shape)


#Investigate all the elements within each Feature

g = sns.pairplot(df_filtered)

for column in df_filtered:
    unique_values = np.unique(df_filtered[column])
    nr_values = len(unique_values)
    if nr_values < 10:
        print("The number of values for feature {} is: {} -- {}".format(column,nr_values,unique_values))
    else:
        print("The number of values for feature {} is: {}".format(column,nr_values))

plt.show()






#Clean Decision Column
#Clean LandUseCode Column
#Clean NumResidentialUnits
#Add keyGrowthArea column
#Add metropolitanArea column
#Add suburbsAndCities column
#Juan
#Clean ApplicationType column
#Clean AreaofSite column
#Clean FloorArea column
#Clean OneOffHouse column
#Filter the csv only applications after 2017/01/01

