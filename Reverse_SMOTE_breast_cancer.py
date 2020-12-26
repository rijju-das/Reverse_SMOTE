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
        if(key!="id_no" and key!="Class"):
            S+=math.pow(x[key]-y[key],2)
    return math.sqrt(S)

def UpdateNeighbors(neighbors, item, distance1, k):
    if(len(neighbors)<k):
        neighbors.append([distance1,item["id_no"],item["Class"]])
        neighbors=sorted(neighbors)
    else:
        if neighbors[-1][0] > distance1:
            neighbors[-1]=[distance1,item["id_no"],item['Class']]
            neighbors= sorted(neighbors)
    return neighbors
def ClaculateNeighborsClass_count(neighbors, k):
    count={}
    count[4]=0
    count[2]=0
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
        if(key!="id_no" and key!="Class"):
            x[key]=round(x[key]+d)
#            print(x)
    return x
def pre_process(df):
    for line in df.values:
        c=0
        for l in line:
            if(l=='?'):
                c=1
        if(c==1):
            df=df.drop(df[df["id_no"]==line[0]].index)
    return df
def main():
    col_names=["id_no", "Clupm_thickness", "Cell_size", "Cell_shape", "Adhesion", "Epi_Cell_size", "Barce_Nucleoli","Chromatin","Normal_Nucleoli", "Mitoses","Class"]
    path=r"C:\Users\Lenovo\NewDropbox\Dropbox\phd2019nitsilchar\code\datasets\breast_cancer\\"
    df=pd.read_csv(path+"breast_cancer.txt", names= col_names)
    df=pre_process(df)
#    df.insert(0,"Index",range(0,0+len(df)))
#    Step1: Divide the dataset(df) in two classes : df_minority and df_majority
    df_minority = df.loc[df["Class"]==4]
    df_majority = df.loc[df["Class"]==2]
    items = fileread_maj(df,df.columns)
    print("Size of minority(Positive) class before applying oversampling\n", df_minority.shape)
    print("Size of majority dataset",df_majority.shape)
    items_minority = fileread_min(df_minority,df_minority.columns)
#    print(items_minority)
    NN={} #to contain list of neighbors(index:[[distance,index,class]])
    NN_list={} #to contain count of neighbors (index: {positive:positive_count,negative:negative_count})
    for nItem in items_minority:
        NN[nItem["id_no"]], NN_list[nItem["id_no"]]=(Class_count(nItem, 5, items))
    POTENT={} #to contain rate of the datapoints who have more number of neighbors from positive class than negative(index,rate)
#    print(NN_list)
    c=0
    for nItem in items_minority:
        
        if NN_list[nItem["id_no"]][4]>NN_list[nItem["id_no"]][2]:
            if(NN_list[nItem["id_no"]][2] !=0):
                rate=math.ceil(NN_list[nItem["id_no"]][4]/NN_list[nItem["id_no"]][2])
                POTENT[nItem["id_no"]]=rate
            else:
                c=c+1
                rate=0
                POTENT[nItem["id_no"]]=rate
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
            g=random.uniform(0,.0001)
#            print(d*g)
            SYN.append(get_SYN(t1[0],d*g))
#    print(SYN)
    dfObj=pd.DataFrame(SYN)
    df_minority=df_minority.append(dfObj)
#    print("Size of majority class after applying oversampling\n", df_majority.shape)
    print("Size of minority(Positive) class after applying oversampling\n", df_minority.shape)
    df_syn=pd.concat([df_majority,df_minority],ignore_index=True)
    df_syn=df_syn.sort_values(by=['id_no'])
#    print(df_syn)
    df_syn.to_csv(path+"file_proposed_breast_cancer.csv",index=False)
    
    print("Size of dataset",df.shape)
    print("Size of dataset",df_syn.shape)
if __name__== "__main__":
    main()
