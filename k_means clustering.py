# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 07:21:30 2020

@author: whisp
"""

%reset -f

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pylab import rcParams
rcParams['figure.figsize'] = 7, 5

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values

#using the elbow method to find the optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()  

#Applying kmeans to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

#visualizing the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0 ,1], s = 100 , c = 'red', label = 'carefull')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1 ,1], s = 100 , c = 'blue', label = 'standard')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2 ,1], s = 100 , c = 'green', label = 'tagert_client')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3 ,1], s = 100 , c = 'cyan', label = 'careless')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4 ,1], s = 100 , c = 'magenta', label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('clusters of clients')
plt.xlabel('annual income')
plt.ylabel('spending income')
plt.legend()
plt.show()
