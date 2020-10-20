#importing important libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.modelselection as sl
from sklearn.linearmodel import LogisticRegression
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
employment={'Salaried':0 , 'Self employed':1 , ' ':1}

#importing dataset
dataset=pd.readcsv ( 'CreditCardData.csv')
dataset=dataset.dropna()

#creating dummy variable for column EmploymentType
dataset.EmploymentType=[employment[number] for number in dataset.EmploymentType]
loanDefaulter=dataset[['loandefault']]
factors=dataset[['disbursedamount' , 'assetcost' , 'EmploymentType' , 'PRI.CURRENT.BALANCE' , 'PRI.SANCTIONED.AMOUNT' , 'PRIMARY.INSTAL.AMT']]

#creating training and testing set and scaling the data
factorsTrain , factorsTest , loanTrain , loanTest=sl.traintestsplit ( factors , loanDefaulter , testsize=0.2 , shuffle=True)
scaler=StandardScaler()
factorsTrain=scaler.fittransform ( factorsTrain)
factorsTest=scaler.fittransform ( factorsTest)

#creating regression object
LogisticRegressor=LogisticRegression ( )
LogisticRegressor.fit ( factorsTrain , loanTrain)
predictedResult=LogisticRegressor.predict ( factorsTest)
print ( 'Confusion matrix using solver: ')
print ( metrics.confusionmatrix ( loanTest , predictedResult))
print ( 'Accuracy using solver: ')
print ( 'Accuracy:' , metrics.accuracy_score ( loanTest , predictedResult))