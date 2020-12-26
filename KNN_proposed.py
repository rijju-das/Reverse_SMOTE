# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 03:14:12 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix
path=r"C:\Users\Lenovo\NewDropbox\Dropbox\phd2019nitsilchar\code\datasets\Ecoli3\\"
df=pd.read_csv(path+"file.csv")
#print(df,"\n",df.shape)
y = df.iloc[:, -1].values
X = df.iloc[:, :-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=None) 
  
knn = KNeighborsClassifier(n_neighbors=7) 
#print(X)  
knn.fit(X_train,y_train) 
#  
## Predict on dataset which model has not seen before 
y_pred=knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
