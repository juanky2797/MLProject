
#import libraries

import datetime

import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import random

pd.set_option('display.max_columns', None)

#read dataset
df = pd.read_csv('data/application_including_boundary_data.csv',encoding='latin-1', low_memory=False)

#When we see some columns of the data set we can see that there are columns that are not very useful to use as features

#So we can drop these columns in the following way
to_drop = ['Unnamed: 0',
           'DevelopmentDescription',
           'ApplicationNumber',
           'ApplicationNumber',
           'DevelopmentAddress',
           'DevelopmentPostcode',
           'ITMEasting',
           'ITMNorthing',
           'ApplicationStatus',
           'ApplicantForename',
           'ApplicantSurname',
           'ApplicantAddress',
           'LandUseCode',
           'ReceivedDate',
           'WithdrawnDate',
           'DecisionDueDate',
           'GrantDate',
           'ExpiryDate',
           'AppealRefNumber',
           'AppealStatus',
           'AppealDecision',
           'AppealDecisionDate',
           'AppealSubmittedDate',
           'FIRequestDate',
           'FIRecDate',
           'LinkAppDetails',
           'ETL_DATE',
           'OneOffKPI',
           'SiteId',
           'ORIG_FID']

df.drop(to_drop, inplace=True, axis=1)

df.columns = df.columns.str.replace('ï..X', 'Longitude')
df.columns = df.columns.str.replace('Y', 'Latitude')

#Above, we defined a list that contains the names of all the columns we want to drop. Next, we call the drop() function on our object passing in the inplace parameter as True and the axis parameter as 1. This tells Pandas that we want the changes to be made directly in our object and that it should look for the values to be dropped in the columns of the object.

df = df.set_index('OBJECTID')

#So far, we have removed unnecessary columns and changed the index of our DataFrame to something more sensible. In this section, we will clean specific columns and get them to a uniform format to get a better understanding of the dataset and enforce consistency.

# filter by dates
df['DecisionDate']=pd.to_datetime(df['DecisionDate'], format='%Y/%m/%d %H:%M:%S+00')
df = df[(df['DecisionDate'] > '2017-01-01') & (df['DecisionDate'] < '2022-10-01')]

# Cleaning
df.loc[df['NumResidentialUnits'] == 0, 'OneOffHouse'] = 'NA'
df.loc[df['NumResidentialUnits'] == 1, 'OneOffHouse'] = 'Yes'
df.loc[df['NumResidentialUnits'] > 1, 'OneOffHouse'] = 'No'
df.loc[df['OneOffHouse'] == 'Y', 'OneOffHouse'] = 'Yes'
df.loc[df['Key_Growth_Areas'] == 'character(0)', 'Key_Growth_Areas'] = 'Outside_Key_Growth_Areas'
df.loc[df['Cities_and_Suburbs'] == 'character(0)', 'Cities_and_Suburbs'] = 'Outside_Cities_and_Suburbs'
df.loc[df['Metropolitian_Areas'] == 'character(0)', 'Metropolitian_Areas'] = 'Outside_Metropolitian_Areas'

# Delete NaN and null values
df = df.replace(float('nan'),None)
df = df.dropna(subset=['Decision','AreaofSite','NumResidentialUnits','OneOffHouse','FloorArea','ApplicationType'])

# Removing rows with decison values that are in conclusive
df = df[df['Decision'] != 'DECLARED EXEMPT']
df = df[df['Decision'] != 'DECLARED NOT EXEMPT']
df = df[df['Decision'] != 'GRANT PERMISSION & REFUSE PERMISSION    ']
df = df[df['Decision'] != 'GRANT PERMISSION & REFUSE PERMISSION']
df = df[df['Decision'] != 'GRANT PERMISSION & REFUSE PERMISSION              ']
df = df[df['Decision'] != 'REQUEST AI EXT OF TIME']
df = df[df['Decision'] != 'REFUSE PERMISSION & GRANT RETENTION               ']
df = df[df['Decision'] != 'Grant Retention & Refuse Retention                ']
df = df[df['Decision'] != 'DECLARED EXEMPT & DECLARED NOT EXEMPT']
df = df[df['Decision'] != 'GRANT RETENTION & REFUSE PERMISSION']
df = df[df['Decision'] != 'GRANT TIME EXT (3 months) AI                      ']
df = df[df['Decision'] != 'GRANT PERMISSION & REFUSE RETENTION']
df = df[df['Decision'] != 'Grant Permission & Refuse Retention               ']
df = df[df['Decision'] != 'REPORT RETURNED TO AN BORD PLEANALA']
df = df[df['Decision'] != 'GRANT RETENTION PERM & REFUSE RETENTION ']
df = df[df['Decision'] != 'Ext Dur no ai recd 1mth WD                        ']
df = df[df['Decision'] != 'IS Exempted Development                           ']
df = df[df['Decision'] != 'PAC REPORT & FILE CLOSED                ']
df = df[df['Decision'] != 'Precluded under 34 (12)(b) from Making a Decision ']
df = df[df['Decision'] != 'Referred to An Bord Pleanala for Determination']
df = df[df['Decision'] != 'Returned Application under Section 37(5)          ']
df = df[df['Decision'] != 'REPORT RETURNED TO AN BORD PLEANALA']
df = df[df['Decision'] != 'Grant Perm & Retention and Refuse Retent          ']
df = df[df['Decision'] != 'GRANT RETENTION & REFUSE RETENTION']
df = df[df['Decision'] != 'GRANT PERMISSION & GRANT OUTLINE PERM.          ']
df = df[df['Decision'] != 'SPLIT DECISION(PERMISSION & REFUSAL)']
df = df[df['Decision'] != 'Split decision']
df = df[df['Decision'] != 'SPLIT DECISION(RETENTION PERMISSION)']

# Making decision classes consistent
df[['Decision']] = df[['Decision']].replace('APPLICATION WITHDRAWN','Withdrawn')
df[['Decision']] = df[['Decision']].replace('WITHDRAWN ARTICLE 33 (NO SUB)','Withdrawn')
df[['Decision']] = df[['Decision']].replace('WITHDRAW THE APPLICATION','Withdrawn')
df[['Decision']] = df[['Decision']].replace('DECLARED WITHDRAWN','Withdrawn')
df[['Decision']] = df[['Decision']].replace('WITHDRAW THE APPLICATION          ','Withdrawn')
df[['Decision']] = df[['Decision']].replace('App withdrawn as no AI recd in 6 months           ','Withdrawn')
df[['Decision']] = df[['Decision']].replace('DECLARE APPLICATION WITHDRAWN           ','Withdrawn')
df[['Decision']] = df[['Decision']].replace('WITHDRAWN ARTICLE 33 (SUBSECTION 4)','Withdrawn')
df[['Decision']] = df[['Decision']].replace('WITHDRAWN AT ABP STAGE','Withdrawn')
df[['Decision']] = df[['Decision']].replace('App. withdrawn as no C.AI recd in 6 mths          ','Withdrawn')

df[['Decision']] = df[['Decision']].replace('ADDITIONAL INFORMATION','Request additional information')
df[['Decision']] = df[['Decision']].replace('REQUEST ADDITIONAL INFORMATION                    ','Request additional information')
df[['Decision']] = df[['Decision']].replace('REQUEST ADDITIONAL INFORMATION','Request additional information')
df[['Decision']] = df[['Decision']].replace('REQUEST ADDITIONAL INFORMATION          ','Request additional information')
df[['Decision']] = df[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.','Request additional information')
df[['Decision']] = df[['Decision']].replace('CLARIFICATION OF ADDITIONAL INFORMATION','Request additional information')
df[['Decision']] = df[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.            ','Request additional information')
df[['Decision']] = df[['Decision']].replace('CLARIFICATION OF FURTHER INFORMATION    ','Request additional information')
df[['Decision']] = df[['Decision']].replace('ADDITIONAL INFORMATION','Request additional information')
df[['Decision']] = df[['Decision']].replace('REQUEST ADDITIONAL INFORMATION                    ','Request additional information')
df[['Decision']] = df[['Decision']].replace('REQUEST ADDITIONAL INFORMATION','Request additional information')
df[['Decision']] = df[['Decision']].replace('REQUEST ADDITIONAL INFORMATION          ','Request additional information')
df[['Decision']] = df[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.','Request additional information')
df[['Decision']] = df[['Decision']].replace('CLARIFICATION OF ADDITIONAL INFORMATION','Request additional information')
df[['Decision']] = df[['Decision']].replace('SEEK CLARIFICATION OF ADDITIONAL INFO.            ','Request additional information')
df[['Decision']] = df[['Decision']].replace('CLARIFICATION OF FURTHER INFORMATION    ','Request additional information')

df[['Decision']] = df[['Decision']].replace('APPLICATION DECLARED INVALID','Invalid')
df[['Decision']] = df[['Decision']].replace('INVALID APPLICATION DUE TO SITE NOTICE            ','Invalid')
df[['Decision']] = df[['Decision']].replace('INVALID APPLICATION','Invalid')
df[['Decision']] = df[['Decision']].replace('INVALID - SITE NOTICE','Invalid')
df[['Decision']] = df[['Decision']].replace('DECLARE INVALID (SITE NOTICE)           ','Invalid')
df[['Decision']] = df[['Decision']].replace('DECLARE INVALID (SITE NOTICE)','Invalid')
df[['Decision']] = df[['Decision']].replace('DECLARE APPLICATION INVALID             ','Invalid')
df[['Decision']] = df[['Decision']].replace('CANNOT DETERMINE','Invalid')
df[['Decision']] = df[['Decision']].replace('DECLARE APPLICATION INVALID','Invalid')
df[['Decision']] = df[['Decision']].replace('Invalid Application (site notice)','Invalid')

df[['Decision']] = df[['Decision']].replace('REFUSED','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION                                 ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION                       ','Refused')
df[['Decision']] = df[['Decision']].replace('R','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE RETENTION PERMISSION','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION FOR RETENTION                   ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION FOR RETENTION','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION FOR RETENTION         ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION & REFUSE RETENTION              ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION & REFUSE RETENTION','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE FURTHER EXT. OF DURATION OF PERM.','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE OUTLINE PERMISSION                         ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE OUTLINE PERMISSION               ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE EXT. OF DURATION OF PERMISSION   ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE 3 MONTH EXTENSION                          ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE APPROVAL','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERM. CONSEQUENT TO OUTLINE PERM.          ','Refused')
df[['Decision']] = df[['Decision']].replace('PART 8 REJECTED BY COUNCIL','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE CERTIFICATE of EXEMPTION','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION & REFUSE OUTLINE PERM.','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE PERMISSION & REFUSE OUTLINE PERM.          ','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE EXT. OF DURATION OF PERMISSION','Refused')
df[['Decision']] = df[['Decision']].replace('REFUSE OUTLINE PERMISSION','Refused')

df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION                                  ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION                        ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT RETENTION PERMISSION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION AND RETENTION PERMISSION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION FOR RETENTION                    ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION FOR RETENTION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION & GRANT RETENTION                ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION FOR RETENTION          ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION & GRANT RETENTION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT EXTENSION OF DURATION OF PERMISSION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT EXTENSION OF DURATION OF PERM.    ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION & GRANT RETENTION      ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT EXTENSION OF DURATION OF PERM.              ','Granted')
df[['Decision']] = df[['Decision']].replace('PART 8 APPROVED BY COUNCIL','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT OUTLINE PERMISSION','Granted')
df[['Decision']] = df[['Decision']].replace('APPROVAL FOR THE PROPOSAL TO PROCEED    ','Granted')
df[['Decision']] = df[['Decision']].replace('DECISION QUASHED BY HIGH COURT','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT APPROVAL','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT OUTDOOR EVENT LICENCE','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT OUTLINE PERM. & REFUSE OUTLINE PERM.','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PER. REFU. PERM. GRAT. RETENTION            ','Granted')
df[['Decision']] = df[['Decision']].replace('Grant Perm. & Grant Perm. conq. to O.P.           ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT OUTLINE PERMISSION                          ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT PERM. CONSEQUENT TO OUTLINE PERM.           ','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT FURTHER EXT. OF DURATION OF PERMISSION','Granted')
df[['Decision']] = df[['Decision']].replace('GRANT OUTLINE PERMISSION                ','Granted')
df[['Decision']] = df[['Decision']].replace('CONDITIONAL','Granted')
df[['Decision']] = df[['Decision']].replace('Conditional Permission','Granted')
df[['Decision']] = df[['Decision']].replace('Granted (Conditional)','Granted')
df[['Decision']] = df[['Decision']].replace('C','Granted')
df[['Decision']] = df[['Decision']].replace('UNCONDITIONAL','Granted')
df[['Decision']] = df[['Decision']].replace('Unconditional Permission','Granted')
df[['Decision']] = df[['Decision']].replace('Granted (Unconditional)','Granted')
df[['Decision']] = df[['Decision']].replace('APPLICATION DECLARED INVALID','Granted')
df[['Decision']] = df[['Decision']].replace('INVALID APPLICATION DUE TO SITE NOTICE            ','Granted')
df[['Decision']] = df[['Decision']].replace('INVALID APPLICATION','Granted')
df[['Decision']] = df[['Decision']].replace('INVALID - SITE NOTICE','Granted')
df[['Decision']] = df[['Decision']].replace('DECLARE INVALID (SITE NOTICE)           ','Granted')
df[['Decision']] = df[['Decision']].replace('INVALID PLANNING APPLICATION                      ','Granted')
df[['Decision']] = df[['Decision']].replace('DECLARE INVALID (SITE NOTICE)','Granted')
df[['Decision']] = df[['Decision']].replace('DECLARE APPLICATION INVALID             ','Granted')
df[['Decision']] = df[['Decision']].replace('CANNOT DETERMINE','Granted')
df[['Decision']] = df[['Decision']].replace('DECLARE APPLICATION INVALID','Granted')
df[['Decision']] = df[['Decision']].replace('Invalid Application (site notice)','Granted')

# Making Application Type classes consistent
df[['ApplicationType']] = df[['ApplicationType']].replace('APPROVAL','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('C','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('EXTENSION OF DURATION','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('O','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('OUTLINE PERMISSION','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('Outline Permission','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('P','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('PERMISSION','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('PERMISSION CONSEQUENT','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('Perm. following Grant of Outline Perm.','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('Permission and Outline Permission','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('Permission and Retention','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('R', 'Retention')
df[['ApplicationType']] = df[['ApplicationType']].replace('RETENTION', 'Retention')
df[['ApplicationType']] = df[['ApplicationType']].replace('SDZ Application','Permission')
df[['ApplicationType']] = df[['ApplicationType']].replace('TEMPORARY PERMISSION','Permission')

df = df[df['Decision'] != 'Withdrawn']
df = df[df['Decision'] != 'Invalid']
df = df[df['Decision'] != 'Request additional information']
df[['Decision']] = df[['Decision']].replace('Refused',0)
df[['Decision']] = df[['Decision']].replace('Granted',1)

# df.to_csv('data/cleaned_output.csv')