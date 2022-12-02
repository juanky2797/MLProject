
#import libraries

import datetime
from imblearn.over_sampling import SMOTE
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from random import randint
from IPython.display import Image
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import random
from sklearn.linear_model import LogisticRegression
from sklearn import metrics



# -------------------------- DATA PREPROCESSING ----------------------------


pd.set_option('display.max_columns', None)

#read dataset
df = pd.read_csv('data/application_including_boundary_data.csv',encoding='latin-1', low_memory=False,index_col=False)

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
           'ORIG_FID','OBJECTID']

df.drop(to_drop, inplace=True, axis=1)

df.columns = df.columns.str.replace('ï..X', 'Longitude')
df.columns = df.columns.str.replace('Y', 'Latitude')

#Above, we defined a list that contains the names of all the columns we want to drop. Next, we call the drop() function on our object passing in the inplace parameter as True and the axis parameter as 1. This tells Pandas that we want the changes to be made directly in our object and that it should look for the values to be dropped in the columns of the object.

#df = df.set_index('OBJECTID')

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
df = df.dropna(how = 'any',subset=['Decision','AreaofSite','NumResidentialUnits','OneOffHouse','FloorArea','ApplicationType'])

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
df[['Decision']] = df[['Decision']].replace('GRANT PERMISSION FOLLOWING OUTLINE',1)


df[['OneOffHouse']] = df[['OneOffHouse']].replace('Yes',1)
df[['OneOffHouse']] = df[['OneOffHouse']].replace('No',0)
df[['OneOffHouse']] = df[['OneOffHouse']].replace('NA',randint(0, 1))





#g = sns.pairplot(df.sample(50),hue='Decision')

# Investigate the distribution of y
# here we investigate if the dataset is imbalance or not
#sns.countplot(x='Decision', data=df, palette='Set3')

#making categorical variables into numeric representation
df = pd.get_dummies(df, columns=['PlanningAuthority','ApplicationType','Key_Growth_Areas','Cities_and_Suburbs','Metropolitian_Areas'])

too_drop = ['Decision','DecisionDate']

# Split the data into x & y
X = df.drop(too_drop, axis=1).values
y = df['Decision']
y = df.iloc[:, 2]


y = y.astype(int)


#print(X.shape)
#print(y.shape)


dt = DecisionTreeClassifier(random_state=15, criterion='entropy', max_depth=10)

dt.fit(X,y)


fi_col = []
fi = []


for i,column in enumerate(df.drop(too_drop, axis=1)):
   # print('The feature importance for {} is: {}'.format(column,dt.feature_importances_[i]))
    fi_col.append(column)
    fi.append(dt.feature_importances_[i])

#creating dataset
fi_df = zip(fi_col, fi)
fi_df = pd.DataFrame(fi_df, columns=['Feature', 'Feature Importance'])


#ordering the data
fi_df = fi_df.sort_values('Feature Importance', ascending= False).reset_index()

columns_to_keep = fi_df['Feature'][0:22]


#Split the data into X and y

X = df[columns_to_keep].values
y = df['Decision']
y = df.iloc[:, 2]





#print(X.shape)
#print(y.shape)


#Hold-out validation

smote = SMOTE(random_state=888)


#First one
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)


#Second one
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.1, random_state=15)


#Oversampling using SMOTE
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


#print(y_resampled.value_counts())

print(X_train.shape)
#print(X_test.shape)
#print(X_valid.shape)

print(y_train.shape)
#print(y_test.shape)
#print(y_valid.shape)




#TRAINING LOGISTIC REGRESION

log_reg = LogisticRegression(random_state=10, solver='lbfgs',max_iter=1000)

log_reg.fit(X_resampled, y_resampled)


#Predict - Predict class labels for samples in X

y_pred = log_reg.predict(X_resampled)

#print(y_resampled.value_counts())


 # Predict_proba - Probability Estimates
pred_proba = log_reg.predict_proba(X_resampled)
#print(log_reg.predict_proba(X_resampled))
#print(y_pred)

#coef - Coefficient of the features in the decision function

#print(log_reg.coef_)




## EVALUATING THE MODEL

#accurracy on Train
#print('The training accurracy is: ', log_reg.score(X_train,y_train))
#Accurracy on Test
#print('The test accurracy is: ', log_reg.score(X_test,y_test))


#Classification Report
print(classification_report(y_resampled,y_pred))

#Confusion Matrix function
def plot_confusion_matrix(cm, classes=None, title='Confusion Matrix'):
    """Plots a confusion matrix"""

    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0,vmax=1, annot=True,annot_kws={'size':25})
    else:
        sns.heatmap(cm,vmin=0,vmax=1.)
        plt.title(title)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


#Visualizing cm
cm = confusion_matrix(y_resampled,y_pred)
cm_norm = cm / cm.sum(axis=1).reshape(-1,1)

#print(log_reg.classes_)

plot_confusion_matrix(cm=cm_norm,classes=log_reg.classes_, title='Confusion Matrix')

# Calculating False Positives (FP), False Negatives (FN), True Positives (TP) & True Negatives (TN)

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)


#Sensitivty, hit rate, recall, or true positive rate
TPR = TP / (TP + FN)
#print('The true Positive Rate is: ', TPR)

#Precision or positive predictive value
PPV = TP / (TP + FP)
#print('The Precision is: ', PPV)

#False positive rate or False alarm rate
FPR = FP / (FP+TN)
#print('The False positive rate is: ', FPR)

#False negative rate or Miss rate
FNR = FN / (FN+TP)
#print('The False Negative Rate is: ', FNR)


##Total avarages :
#print("")
#print('The average TPR is:', TPR.sum()/2)
#print('The average Precision is:', PPV.sum()/2)
#print('The average False Positive rate is:', FPR.sum()/2)
#print('The average False Negative Rate is:', FNR.sum()/2)


#Running Log Loss on training
#print('The log Loss on Training is:', log_loss(y_resampled,pred_proba))



#Hyper Parameter Tuning

#Creating a range for C values
np.geomspace(1e-5, 1e5, num=20)

fig, ay, = plt.subplots()

#ploting it
ay.plot(np.geomspace(1e-5, 1e5, num=20)) # uniformly distributed in Log space
ay.plot(np.linspace(1e-5,1e5, num=20)) # uniformly distributed in linear space, instead of Log space

#Looping over the parameters

C_list = np.geomspace(1e-5, 1e5, num=20)
CA = []
Logaritmic_Loss = []

for c in C_list:
    log_reg2 = LogisticRegression(random_state=10, solver='lbfgs', C=c,max_iter=10000)
    log_reg2.fit(X_resampled, y_resampled)
    score = log_reg2.score(X_resampled,y_resampled)
    CA.append(score)
   # print("The classification accurracy of C parameter {} is: {}:".format(c,score))
    pred_proba_t = log_reg2.predict_proba(X_resampled)
    log_loss2 = log_loss(y_resampled,pred_proba_t)
    Logaritmic_Loss.append(log_loss2)
    #print("The logg Loss of C parameter {} is {}:".format(c,log_loss2))
    #print("")


# putting the outcomes in a Table

#Reshaping
CA2 = np.array(CA).reshape(20,)
Logaritmic_Loss2 = np.array(Logaritmic_Loss).reshape(20,)

#zip
outcomes = zip(C_list, CA2, Logaritmic_Loss2)

#df
df_outcomes = pd.DataFrame(outcomes, columns=["C_List", 'CA2','Logarithmic_Loss2'])


#Ordering the data (sort values)
df_outcomes = df_outcomes.sort_values("Logarithmic_Loss2", ascending=True)

print(df_outcomes)



## Applying K-fold cross validation:


from sklearn.model_selection import KFold
kf = KFold(n_splits= 5, random_state = 0, shuffle=True)

#Logistic Regresion Cross validation:
Log_reg3 = LogisticRegressionCV(cv=kf,random_state=15, Cs=C_list)
Log_reg3.fit(X_resampled,y_resampled)
#print("The CA is:", Log_reg3.score(X_resampled, y_resampled))
pred_proba_t = Log_reg3.predict_proba(X_resampled)
log_loss3 = log_loss(y_resampled,pred_proba_t)
#print("The logistic Loss is: ", log_loss3)

#print("The optimal C parameter is: ", Log_reg3.C_)





#Training a Dummy Classifier

dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_resampled,y_resampled)
score = dummy_clf.score(X_test,y_test)

pred_proba_t = dummy_clf.predict_proba(X_resampled)
log_loss2 = log_loss(y_resampled,pred_proba_t)
Logaritmic_Loss.append(log_loss2)

#print("Testing accurracy:", score)
#print("Log Loss:", log_loss2)



# FINAL MODEL WITH SELECTED PARAMETERS

log_reg3 = LogisticRegression(random_state=10,solver='lbfgs',C=0.048329)
log_reg3.fit(X_resampled, y_resampled)
score = log_reg3.score(X_resampled, y_resampled)

y_pred_logistic = log_reg3.predict(X_resampled)




#------- KNN CLASSIFIER -----------


mean_score = []
std_score = []

#for K in range(2,10):
 #   clf = KNeighborsClassifier(n_neighbors=K)
  #  clf.fit(X_resampled,y_resampled)
   # scores = cross_val_score(clf, X_train, y_train, cv=5)
    #mean_score.append(scores.mean()*100)
    #std_score.append(scores.std()*100)
    #y_pred = clf.predict(X_resampled)
    #print("Score",round(scores.mean(),3))
    #print("Score",round((y_pred==y_test.values).sum()/y_pred.shape[0],3))


clf = KNeighborsClassifier(n_neighbors=10)
clf.fit(X_resampled,y_resampled)
y_pred_knn = clf.predict(X_resampled)


conf_matrix_knn = metrics.confusion_matrix(y_resampled,y_pred_knn)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix_knn, display_labels=[True, False])

dummy = DummyClassifier(strategy='most_frequent').fit(X_resampled, y_resampled)
ydummy1 = dummy.predict(X_resampled)

dummy = DummyRegressor(strategy='mean').fit(X_resampled, y_resampled)
ydummy2 = dummy.predict(X_resampled)



result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])
fpr, tpr, _ = roc_curve(y_resampled,y_pred_logistic)
auc = roc_auc_score(y_resampled,y_pred_logistic)
result_table = result_table.append({'classifiers':'logistics',
                                        'fpr':fpr,
                                        'tpr':tpr,
                                        'auc':auc}, ignore_index=True)


fpr, tpr, _ = roc_curve(y_resampled,y_pred_knn)
auc = roc_auc_score(y_resampled,y_pred_knn)
result_table = result_table.append({'classifiers':'knn',
                                       'fpr':fpr,
                                       'tpr':tpr,
                                       'auc':auc}, ignore_index=True)

fig, ayz, = plt.subplots()

for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'],
             result_table.loc[i]['tpr'],
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc'],result_table.loc[i]['classifiers']))



plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("False Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size': 13}, loc='lower right')




plt.show()



#df.to_csv('data/cleaned_output.csv')