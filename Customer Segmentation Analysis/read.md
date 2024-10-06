# Step 1: Import Libraries
First, we need to bring in necessary libraries for data handling, visualization, and clustering:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```
# Step 2: Data Collection
We load our data from a CSV file into a pandas DataFrame and check its structure and any missing values:

```
customer_data = pd.read_csv('C:/Users/tanya/OneDrive/Desktop/Oasis Infobytes Customer Clustering/ifood_df.csv')
print(customer_data.head())  # Print first few rows
print(customer_data.shape)   # Print shape of the dataset
print(customer_data.info())  # Print info about the dataset
print(customer_data.isnull().sum())  # Check for any missing values
```

# Step 3: Select Relevant Features
We select the columns we need for clustering. Here, we choose 'Income' (0th column) and 'MntTotal' (36th column):
```
X = customer_data.iloc[:, [0, 36]].values
print(X)
```
# Step 4: Choose the Number of Clusters
We use the Elbow Method to determine the optimal number of clusters by plotting the Within-Cluster Sum of Squares (WCSS) for different numbers of clusters:
```
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

sns.set()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```
# Step 5: Train the K-Means Model
Based on the elbow graph, we choose the optimal number of clusters (in this case, 4), and train our K-Means clustering model:
```
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
Y = kmeans.fit_predict(X)
print(Y)
```
# Step 6: Visualize the Clusters
Finally, we plot the clusters and their centroids to visualize the customer groups:
```
plt.figure(figsize=(8, 8))
plt.scatter(X[Y == 0, 0], X[Y == 0, 1], s=50, c='green', label='Cluster 1')
plt.scatter(X[Y == 1, 0], X[Y == 1, 1], s=50, c='red', label='Cluster 2')
plt.scatter(X[Y == 2, 0], X[Y == 2, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[Y == 3, 0], X[Y == 3, 1], s=50, c='blue', label='Cluster 4')

#Plotting the centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', label='Centroids')

plt.title('Customer Groups')
plt.xlabel('Income')
plt.ylabel('Total amount spent')
plt.legend()
plt.show()
```
That’s it! You’ve successfully clustered customer data based on their income and spending and visualized the results. Easy peasy! 🚀
