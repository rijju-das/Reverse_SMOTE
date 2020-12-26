# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:26:18 2020

@author: Lenovo
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA as sklearnPCA

path=r"C:\Users\Lenovo\NewDropbox\Dropbox\phd2019nitsilchar\code\datasets\pima\\"
df=pd.read_csv(path+'pima.dat')
df_minority = df.loc[df["Class"]=="positive"]
df_majority = df.loc[df["Class"]=="negative"]

y = df['Class']          # Split off classifications
X = df.iloc[:,:-1] # Split off features
print("Size of majority dataset",df.loc[df["Class"]=="positive"].shape)
print("Size of majority dataset",df.loc[df["Class"]=="negative"].shape)
X_norm = (X - X.min())/(X.max() - X.min())
print(X_norm.shape)
pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
print(transformed[y=='positive'])
plt.scatter(transformed[y=='positive'][0], transformed[y=='positive'][1], label='Minority', c='red')
plt.scatter(transformed[y=='negative'][0], transformed[y=='negative'][1], label='Majority', c='blue')

plt.legend()
plt.show()
