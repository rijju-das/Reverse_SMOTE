# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 05:46:12 2020

@author: Lenovo
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 18:52:29 2020

@author: riju
"""
import pandas as pd
import random
import math

def fileread_min(df,col_names):
    items=[]
    features=col_names[:-1]
    for lines in df.values:
        line=lines.tolist()
#        print(line)
        itemFeatures={}
        for j in range(len(features)):
            f=features[j]
            v=float(line[j])
            itemFeatures[f]=v
        items.append(itemFeatures)
#    print(items)
    return items
def fileread_maj(df,col_names):
    items=[]
    features=col_names[:-1]
    for lines in df.values:
        line=lines.tolist()
        itemFeatures={"Class" : line[-1]}
        for j in range(len(features)):
            f=features[j]
            v=float(line[j])
            itemFeatures[f]=v
        items.append(itemFeatures)
#    print(items)
    return items
def fetch_value(key,df,col_names):
    r_items=[]
    r_features=col_names[:]
    for line in df.values:
        if key==line[0]:
            r_itemFeatures={}
            for j in range(len(r_features)):
                f=r_features[j]
                if(f=="Class"):
                    v=line[j]
                else:
                    v=float(line[j])
                r_itemFeatures[f]=v
            r_items.append(r_itemFeatures)
    return r_items  
def Reverse_NN(rItem, k, Items):
    if(k>len(Items)):
        return "k larger than list length"
    n_neighbors=[]
    for item in Items:
        distance1=EuclideanDistance(rItem, item)

        n_neighbors=UpdateNeighbors(n_neighbors, item, distance1,k)
  
        
def Class_count(nItem, k, Items):
    if(k>len(Items)):
        return "k larger than list length"
    neighbors=[]
    for item in Items:
        distance1=EuclideanDistance(nItem, item)
#        print(distance)
        neighbors=UpdateNeighbors(neighbors, item, distance1,k)
#    print(neighbors)
    count=ClaculateNeighborsClass_count(neighbors,k)
#    print(count)
    return neighbors, count
#    return FindMax(count)
def EuclideanDistance(x,y):
    S=0
    for key in x.keys():
        if(key!="Index" and key!="Class"):
            S+=math.pow(x[key]-y[key],2)
    return math.sqrt(S)

def UpdateNeighbors(neighbors, item, distance1, k):
    if(len(neighbors)<k):
        neighbors.append([distance1,item["Index"],item["Class"]])
        neighbors=sorted(neighbors)
    else:
        if neighbors[-1][0] > distance1:
            neighbors[-1]=[distance1,item["Index"],item['Class']]
            neighbors= sorted(neighbors)
    return neighbors
def ClaculateNeighborsClass_count(neighbors, k):
    count={}
    count[1]=0
    count[0]=0
    for i in range(k):
        
        if(neighbors[i][-1] in count):
            count[neighbors[i][-1]] += 1
        #print(count)
    return count
def FindMax(countList):
    maximum = -1;
    classification =""
    for key in countList.keys():
        if(countList[key]>maximum):
            maximum = countList[key]
            classification = key
    return classification

def get_SYN(x,d):
    for key in x.keys():
        if(key!="Index" and key!="Class"):
            x[key]=x[key]+d
#            print(x)
    return x
def majority_minority_count(df):
    df_minority = df.loc[df["Class"]==1]
    df_majority = df.loc[df["Class"]==0]
    print("Size of minority= ",df_minority.shape)
    print("Size of majority= ", df_majority.shape)
    print("Total Size= ",df.shape)
def reverse_smote(df):
    df.insert(0,"Index",range(0,0+len(df)))
    #Step1: Divide the dataset(df) in two classes : df_minority and df_majority
    df_minority = df.loc[df["Class"]==1]
    df_majority = df.loc[df["Class"]==0]
    items = fileread_maj(df,df.columns)
    items_minority = fileread_min(df_minority,df_minority.columns)
#    print(items_minority)
    NN={} #to contain list of neighbors(index:[[distance,index,class]])
    NN_list={} #to contain count of neighbors (index: {positive:positive_count,negative:negative_count})
    for nItem in items_minority:
        NN[nItem["Index"]], NN_list[nItem["Index"]]=(Class_count(nItem, 10, items))
    POTENT={} #to contain rate of the datapoints who have more number of neighbors from positive class than negative(index,rate)
#    print(NN_list)
    for nItem in items_minority:
        if NN_list[nItem["Index"]][1]>NN_list[nItem["Index"]][0]:
            if(NN_list[nItem["Index"]][0] !=0):
                rate=math.ceil(NN_list[nItem["Index"]][1]/NN_list[nItem["Index"]][0])
                POTENT[nItem["Index"]]=rate
            else:
                rate=NN_list[nItem["Index"]][1]
                POTENT[nItem["Index"]]=rate
#    print(POTENT)
    RNN={}
    for x,y in POTENT.items():
        tmp=[]
        for rx,ry in NN.items():
#            print(ry)
            for z in ry:
#                print(z[0])
                if x == z[1] and z[0]!=0.0:
                    tmp.append(rx)
        RNN[x]=tmp
    XRR={}
    for x, y in POTENT.items():
        XR=[]
        m=min(len(RNN[x]),y)
        for i in range(0,m):

            XR.append(RNN[x][i])
        XRR[x]=XR
    t1=[]
    SYN=[]
    for x, y in POTENT.items():
        t1=fetch_value(x,df,df.columns)
        t2=[]
#        print(t1)
        for i in range(0,len(XRR[x])):
            t2=fetch_value(XRR[x][i],df,df.columns)
            d=EuclideanDistance(t1[0],t2[0])
#            d=distance.euclidean(t1[0],t2[0])
#            print(d)
            g=random.uniform(0,.0001)
#            print(d*g)
            SYN.append(get_SYN(t1[0],d*g))
#    print(SYN)
    dfObj=pd.DataFrame(SYN)
    df_minority=df_minority.append(dfObj)
    df_syn=pd.concat([df_majority,df_minority],ignore_index=True)
    df_syn=df_syn.sort_values(by=['Index'])
    df_syn=df_syn.drop(columns=['Index'])
    return(df_syn)
def oversampler_fun(df,oversampler):
    X=df.iloc[:,:-1].values
    y=df.iloc[:,-1].values
    X_samp, y_samp= oversampler.sample(X, y)
    df=pd.concat([pd.DataFrame(X_samp),pd.DataFrame(y_samp)],axis=1)
    return df    




def classifier_result(df):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25,random_state=0)
    
#    from sklearn.model_selection import cross_validate
#    scoring = ['accuracy', 'precision', 'recall', 'f1']
#    clf = SVC(kernel='linear', C=1, random_state=0)
#    scores = cross_validate(clf, X_train, y_train, cv=5,
#                        scoring=scoring, return_train_score=False)
#    print(scores)
    temp=[]
    
#    print("\nResults of Support Vector Machine:")
    modelsvc = SVC(random_state = 0, gamma =0.8, kernel ='rbf')
    modelsvc.fit(X_train, y_train)
    y_pred = modelsvc.predict(X_test)
    # print(roc_auc_score(y_pred, probs)) 
    temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])      
#    print("accuracy : {:.6f} ".format(accuracy_score(y_test, y_pred)))
#    print("Precision : {:.6f} ".format(precision_score(y_test, y_pred, average='weighted')))
#    print("Recall : {:.6f} ".format(recall_score(y_test, y_pred, average='weighted')))
#    print("F-measure : {:.6f} ".format(f1_score(y_test, y_pred, average='weighted')))

#    print("\nResults of Random Forest:")
    modelrf = RandomForestClassifier(n_estimators=5, random_state = 0)
    modelrf.fit(X_train, y_train)
    y_pred = modelrf.predict(X_test)
    # print(roc_auc_score(y_pred, probs))  
    temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])     
#    print("accuracy : {:.6f} ".format(accuracy_score(y_test, y_pred)))
#    print("Precision : {:.6f} ".format(precision_score(y_test, y_pred, average='weighted')))
#    print("Recall : {:.6f} ".format(recall_score(y_test, y_pred, average='weighted')))
#    print("F-measure : {:.6f} ".format(f1_score(y_test, y_pred, average='weighted')))
    
#    print("\nResults of Logistic Regression:")
    modellr = LogisticRegression(solver='lbfgs')
    modellr.fit(X_train, y_train)
    y_pred = modellr.predict(X_test)
    temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])      
#    print("accuracy : {:.6f} ".format(accuracy_score(y_test, y_pred)))
#    print("Precision : {:.6f} ".format(precision_score(y_test, y_pred, average='weighted')))
#    print("Recall : {:.6f} ".format(recall_score(y_test, y_pred, average='weighted')))
#    print("F-measure : {:.6f} ".format(f1_score(y_test, y_pred, average='weighted')))
    
#    print("\nResults of Decision Tree:")
    modeldt = DecisionTreeClassifier()
    modeldt.fit(X_train, y_train)
    y_pred = modeldt.predict(X_test)
    temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])       
#    print("accuracy : {:.6f} ".format(accuracy_score(y_test, y_pred)))
#    print("Precision : {:.6f} ".format(precision_score(y_test, y_pred, average='weighted')))
#    print("Recall : {:.6f} ".format(recall_score(y_test, y_pred, average='weighted')))
#    print("F-measure : {:.6f} ".format(f1_score(y_test, y_pred, average='weighted')))

#    print("\nResults of Gradient Boosting:")
    modelgb = GradientBoostingClassifier(random_state = 0)
    modelgb.fit(X_train, y_train)
    y_pred = modelgb.predict(X_test)
    temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])      
#    print("accuracy : {:.6f} ".format(accuracy_score(y_test, y_pred)))
#    print("Precision : {:.6f} ".format(precision_score(y_test, y_pred, average='weighted')))
#    print("Recall : {:.6f} ".format(recall_score(y_test, y_pred, average='weighted')))
#    print("F-measure : {:.6f} ".format(f1_score(y_test, y_pred, average='weighted')))

#    print("\nResults of XGBoost:")
    modelxg = XGBClassifier()
    modelxg.fit(X_train, y_train)
    y_pred = modelxg.predict(X_test)
    temp.append([accuracy_score(y_test, y_pred),precision_score(y_test, y_pred, average='weighted'),recall_score(y_test, y_pred, average='weighted'),f1_score(y_test, y_pred, average='weighted')])      
#    print("accuracy : {:.6f} ".format(accuracy_score(y_test, y_pred)))
#    print("Precision : {:.6f} ".format(precision_score(y_test, y_pred, average='weighted')))
#    print("Recall : {:.6f} ".format(recall_score(y_test, y_pred, average='weighted')))
#    print("F-measure : {:.6f} ".format(f1_score(y_test, y_pred, average='weighted')))
    
    
    df_temp=pd.DataFrame(temp,columns=["Accuracy","Precision","Recall","F-measure"],index=["Support Vector Machine","Random Forest","Logistic Regression","Decision Tree","Gradient Boosting", "XGBoost"])
    return(df_temp)