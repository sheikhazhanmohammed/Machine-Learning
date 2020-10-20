#importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.modelselection as sl
from sklearn import linearmodel
from sklearn.metrics import meansquarederror , r2score

#loading dataset and storing in a variable named dataset
dataset=pd.readcsv ( 'Climate.csv')

#Creating primary array of temperature and rainfall
Temperature=dataset[['MeanTemperatureMinimum']]
Rainfall=dataset[['MeanRainfallMM']]

#Splitting the dataset
Temptrain , Temptest , Raintrain , Raintest=sl.traintestsplit ( Temperature , Rainfall , testsize=0.2 , shuffle=True)

#Creating regression object
regr=linearmodel.LinearRegression ( fitintercept=True , normalize=False)

#Training the model
regr.fit ( Temptrain , Raintrain)

#Using the model to predict values from Test Dataset
Rainpred=regr.predict ( Temptest)

#Printing mean square error and r2 score
print ( 'Mean squared error: ' , meansquarederror ( Raintest , Rainpred) )
print ( 'R2 Score: ' , r2score ( Raintest , Rainpred) )

#Plotting the dataset
plt.scatter ( Temptest , Raintest , color='black')
plt.scatter ( Temptest , Rainpred , color='blue')

# naming the x axis
plt.xlabel ( 'Mean Temperature Maximum')
# naming the y axis
plt.ylabel ( 'Mean Rainfall in mm')