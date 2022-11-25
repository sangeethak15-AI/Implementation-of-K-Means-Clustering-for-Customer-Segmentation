# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data set and find the number of null data.
3. Import KMeans from sklearn.clusters library package.
4. Find the y_pred.
5. Plot the clusters in graph.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Sangeetha.K
RegisterNumber:212221230085  
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.cluster import KMeans
wcss=[]

for i in range (1,11):
    kmeans=KMeans(n_clusters = i,init="k-means++")
    kmeans.fit(data.iloc[:,3:])
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")

km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])

y_pred=km.predict(data.iloc[:,3:])
y_pred

data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segmets")
```

## Output:


![m1](https://user-images.githubusercontent.com/93992063/204021456-aee19b1c-ad67-43ae-8dda-b8d18dc8d708.png)

![m2](https://user-images.githubusercontent.com/93992063/204021469-213b7377-59d4-4c1a-9965-504a11f768f6.png)

![3](https://user-images.githubusercontent.com/93992063/204021549-a4b060e9-e485-4b0e-9d35-0a870771b880.png)

![4](https://user-images.githubusercontent.com/93992063/204021561-c09a1097-5cc0-4423-8cb8-4d82a0ad2de6.png)


![6](https://user-images.githubusercontent.com/93992063/204021649-861bc79b-a125-4ee9-8874-f6f59ddee502.png)


![7](https://user-images.githubusercontent.com/93992063/204021624-a3bfb0d8-3a16-4537-a558-51b9cd9672e8.png)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
