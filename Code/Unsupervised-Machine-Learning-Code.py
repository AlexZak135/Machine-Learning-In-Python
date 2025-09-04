# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 3, 2025

import numpy as np
import os
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

customers = pd.read_csv("Mall-Customers-Data.csv")

model = KMeans(n_clusters = 5)
clusters = model.fit_predict(customers[["Annual Income", "Spending Score"]])

def plot_elbow_curve(df, max_clusters = 10):
    
    inertias = [] 
    
    for k in range(1, max_clusters + 1):
        model = KMeans(n_clusters = k, random_state = 744)
        cluster = model.fit_predict(df)
        inertias.append(round(model.inertia_, 2)) 
        
    plt.figure(figsize = (12, 8))
    plt.plot(range(1, max_clusters + 1), inertias, marker = "o")    
    plt.xticks(ticks = range(1, max_clusters + 1), 
               labels = range(1, max_clusters + 1))
    plt.title("Inertia vs Number of Clusters")
    plt.tight_layout()
    plt.show()
    
    return inertias
  
inertias = plot_elbow_curve(customers[["Annual Income", "Spending Score"]])
print(inertias)

customers = customers[["Annual Income", "Spending Score"]] 
scaler = StandardScaler()
scaler.fit(customers)
customers = scaler.transform(customers)

inertias = plot_elbow_curve(customers)
print(inertias)

customers = pd.read_csv("Mall-Customers-Data.csv")
customers = customers.drop(columns = "CustomerID")
customers["Gender"] = np.where(customers["Gender"] == "Male", 1, 0)

scaler = StandardScaler()
scaler.fit(customers)
customers_scaled = scaler.transform(customers)

inertias = plot_elbow_curve(customers_scaled)
print(inertias)

model = KMeans(n_clusters = 6, random_state = 123)
clusters = model.fit_predict(customers_scaled)
customers["Cluster"] = clusters + 1
print(customers["Cluster"].value_counts())

# 6 of 9