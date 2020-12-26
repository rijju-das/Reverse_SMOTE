# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 07:16:22 2020

@author: Lenovo
"""

# roc curve and auc.3253
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score, auc
from matplotlib import pyplot
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from statistics import mean 
from scipy import interp
import matplotlib.pylab as plt
from sklearn.utils import shuffle

def ROC_graph_create(df, classifier):
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    
    cv = KFold(n_splits=5, random_state=None, shuffle=False)
    tprs = []
    base_fpr = np.linspace(0, 1, 101)
    auc=[]
    for train,test in cv.split(X,y):
        classifier.fit(X[train],y[train])
        y_score = classifier.predict_proba(X[test])
        fpr, tpr, _ = roc_curve(y[test], y_score[:, 1])
        auc.append(roc_auc_score(y[test],y_score[:,1]))
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)
    print(mean(auc))   
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    return base_fpr, mean_tprs

def main():
    pathe=r"C:\Users\Lenovo\NewDropbox\Dropbox\phd2019nitsilchar\code\datasets\ecoli3\\"
    pathp=r"C:\Users\Lenovo\NewDropbox\Dropbox\phd2019nitsilchar\code\datasets\pima\\"
    pathb=r"C:\Users\Lenovo\NewDropbox\Dropbox\phd2019nitsilchar\code\datasets\PageBlock\\"
    
    plt.figure(figsize=(8, 8))
    df=pd.read_csv(pathp+"file_proposed_pima.csv")
    df.columns = [*df.columns[:-1], 'Class']
    df['Class'] = df['Class'].apply({'positive':0, 'negative':1}.get)
    df=shuffle(df)
    base_fpr, mean_tprs=ROC_graph_create(df,KNeighborsClassifier(n_neighbors=10))
    plt.plot(base_fpr, mean_tprs, 'g',label="Proposed")

    df=pd.read_csv(pathp+"file_Boderline1_pima.csv")
    df.columns = [*df.columns[:-1], 'Class']
    df['Class'] = df['Class'].apply({'positive':0, 'negative':1}.get)
    df=shuffle(df)
    base_fpr, mean_tprs=ROC_graph_create(df,KNeighborsClassifier(n_neighbors=10))
    plt.plot(base_fpr, mean_tprs, 'r', label="Boderline1")
    
    df=pd.read_csv(pathp+"file_ADASYN_pima.csv")
    df.columns = [*df.columns[:-1], 'Class']
    df['Class'] = df['Class'].apply({'positive':0, 'negative':1}.get)
    df=shuffle(df)
#    print(df)
    base_fpr, mean_tprs=ROC_graph_create(df,KNeighborsClassifier(n_neighbors=10))
    plt.plot(base_fpr, mean_tprs, 'b', label="ADASYN")

    df=pd.read_csv(pathp+"file_Safelevel_SMOTE_pima.csv")
    df.columns = [*df.columns[:-1], 'Class']
    df['Class'] = df['Class'].apply({'positive':0, 'negative':1}.get)
    df=shuffle(df)
    base_fpr, mean_tprs=ROC_graph_create(df,KNeighborsClassifier(n_neighbors=10))
    plt.plot(base_fpr, mean_tprs, 'y', label="Safe level SMOTE")
    
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig("ROC_pima")
    plt.show()
    
if __name__=="__main__":
    main()