import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


#Data collection
#loading the data from csv file to a Pandas DataFrame
customer_data = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/Oasis Infobytes Customer Clustering/ifood_df.csv')
#info about our data
print(customer_data.head())
print(customer_data.shape)
print(customer_data.info())
print(customer_data.isnull().sum())

#Main features that we want to select: Income (0th column), MntTotal (36th column)
X = customer_data.iloc[:,[0,36]].values
print(X)

#choosing the number of clusters using WCSS - Within Cluster Sum of Squares
#finding wcss value for different no of clusters.
# we need less wcss valued clusters
wcss=[]

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plottimg elbow graph to choose kth value(optimum no of clusters)
sns.set()
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#optimum no of clusters = 4
#training the k-means Clustering model
kmeans = KMeans(
    n_clusters = 4, init = 'k-means++', random_state=42
)

#return a label for each data point
Y = kmeans.fit_predict(X)
print(Y)

#visualizing all clusters
#plotting all the clusters and their centroids
plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green', label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red', label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow', label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='blue', label='Cluster 4')
#centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='black',label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Income')
plt.ylabel('Total amount spent')
plt.show()
