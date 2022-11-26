
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
to_drop = [
           'ApplicationNumber',
           'DevelopmentDescription',
           'ApplicationNumber',
           'DevelopmentDescription',
           'DevelopmentAddress',
           'DevelopmentPostcode',
           'ITMEasting',
           'ITMNorthing',
           'ApplicationStatus',
           'ApplicantForename',
           'ApplicantSurname',
           'ApplicantAddress',
           'LandUseCode',
           'WithdrawnDate',
           'DecisionDate',
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

#Above, we defined a list that contains the names of all the columns we want to drop. Next, we call the drop() function on our object passing in the inplace parameter as True and the axis parameter as 1. This tells Pandas that we want the changes to be made directly in our object and that it should look for the values to be dropped in the columns of the object.

df = df.set_index('OBJECTID')

#print(df['OBJECTID'].is_unique)


#So far, we have removed unnecessary columns and changed the index of our DataFrame to something more sensible. In this section, we will clean specific columns and get them to a uniform format to get a better understanding of the dataset and enforce consistency.



#print(df.loc[50548])

#date = datetime.date(df['ReceivedDate'])

print(df['ReceivedDate'].dtype)

#Parse RecievedDate data to datetime
df['ReceivedDate'] = pd.to_datetime(df['ReceivedDate'], errors='coerce')

#Filter data between 2017 and 2022
df = df[(df['ReceivedDate'] > '2017-01-01') & (df['ReceivedDate'] < '2022-10-01')]


#Exploratory Data Analysis
# Checking for null values

#print(df.shape)

#delete NaN and null values
df = df.dropna(subset=['Decision','AreaofSite','NumResidentialUnits','OneOffHouse','FloorArea','ApplicationType'])

print(df)

#print(df.isnull().sum())

#print(df)





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

