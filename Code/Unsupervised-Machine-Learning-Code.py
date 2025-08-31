# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 1, 2025

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

customers = pd.read_csv("Mall-Customers-Data.csv")[["Age", "Spending Score"]]
centroids = customers.sample(2, random_state = 123)

def fetch_coordinates(df):
    age_centroid_1 = df.iloc[0, 0]
    score_centroid_1 = df.iloc[0, 1]
    age_centroid_2 = df.iloc[1, 0]
    score_centroid_2 = df.iloc[1, 1]
    
    return age_centroid_1, score_centroid_1, age_centroid_2, score_centroid_2

age_centroid_1, score_centroid_1, age_centroid_2, score_centroid_2 = ( 
    fetch_coordinates(centroids)
    )

plt.scatter(customers["Age"], customers["Spending Score"])
plt.scatter(centroids["Age"], centroids["Spending Score"], color = "black", 
            s = 100)
plt.show()

def calculate_distance(df, age_centroid, score_centroid):
    distance = np.sqrt((df.loc["Age"] - age_centroid) ** 2 + 
                       (df.loc["Spending Score"] - score_centroid) ** 2)
    
    return distance

customers["dist_centroid_1"] = customers.apply(
    calculate_distance, args = (age_centroid_1, score_centroid_1), axis = 1
    )
customers["dist_centroid_2"] = customers.apply(
    calculate_distance, args = (age_centroid_2, score_centroid_2), axis = 1
    )

def calculate_distance_assign_clusters(customers, centroids):
    age_centroid_1, score_centroid_1, age_centroid_2, score_centroid_2 = (
        fetch_coordinates(centroids)
        )
    
    customers["dist_centroid_1"] = customers.apply(
        calculate_distance, args = (age_centroid_1, score_centroid_1), axis = 1
        )
    customers["dist_centroid_2"] = customers.apply(
        calculate_distance, args = (age_centroid_2, score_centroid_2), axis = 1
        )
    
    customers["cluster"] = np.where(
        customers["dist_centroid_1"] < customers["dist_centroid_2"], 1, 2
        )
    
    return customers

customers = calculate_distance_assign_clusters(customers, centroids)

sns.scatterplot(x = "Age", y = "Spending Score", hue = "cluster", 
                palette = "tab10", data = customers, s = 50)
sns.scatterplot(x = "Age", y = "Spending Score", color = "black", 
                data = centroids, s = 100)
plt.show()

new_centroids = (
    customers.groupby("cluster")[["Age", "Spending Score"]].mean().reset_index()
    ).drop("cluster", axis = 1)

customers = calculate_distance_assign_clusters(customers, new_centroids)

sns.scatterplot(x = "Age", y = "Spending Score", hue = "cluster", 
                palette = "tab10", data = customers, s = 50)
sns.scatterplot(x = "Age", y = "Spending Score", color = "black", 
                data = new_centroids, s = 100)
plt.show()

customers = customers[["Age", "Spending Score"]]

def create_clusters(df):
    centroids = df.sample(2, random_state = 123)
    df = calculate_distance_assign_clusters(df, centroids)
    new_centroids = (
        df.groupby("cluster")[["Age", "Spending Score"]].mean().reset_index()
        )
    new_centroids.drop("cluster", axis = 1)
    customers = calculate_distance_assign_clusters(df, new_centroids)
    
    return df["cluster"]

clusters = create_clusters(customers)

