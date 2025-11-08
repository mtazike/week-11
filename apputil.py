# your code here

import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans


# Exercise 1 
def kmeans(X, k):
    km = KMeans(n_clusters=k, random_state=0, n_init='auto')
    km.fit(X)
    centroids = km.cluster_centers_
    labels = km.labels_
    return centroids, labels


# Exercise 2 
diamonds = sns.load_dataset('diamonds')
numeric_diamonds = diamonds.select_dtypes(include=[np.number])

def kmeans_diamonds(n=1000, k=5):
    X = numeric_diamonds.head(n).to_numpy()
    centroids, labels = kmeans(X, k)
    return centroids, labels

# Exersice 3
from time import time

def kmeans_timer(n, k, n_iter=5):
    times = []
    
    for i in range(n_iter):
        start = time()                
        _ = kmeans_diamonds(n, k)     
        elapsed = time() - start      
        times.append(elapsed)
    
    avg_time = np.mean(times)        
    return avg_time