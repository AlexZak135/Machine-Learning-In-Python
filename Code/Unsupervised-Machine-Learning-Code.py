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

customers = pd.read_csv("Mall-Customers-Data.csv")[["Annual Income", 
                                                    "Spending Score"]]

def get_centroids(df, k):
    df_sample = df.sample(k, random_state = 123).reset_index(drop = True)
    df_ll = df_sample.values.tolist()
    
    return df_sample, df_ll

centroids, coords = get_centroids(customers, 2)

def calculate_distance(df, centroids_coords):
    names = []
    
    for i, centroid in enumerate(centroids_coords):
        df[f"dist_centroid_{i + 1}"] = (
            np.sqrt((df.iloc[:, 0] - centroid[0]) ** 2 + 
                    (df.iloc[:, 1] - centroid[1]) ** 2)
            )
        
        names.append(f"dist_centroid_{i + 1}")
    
    return df, names

customers, dist_names = calculate_distance(customers, coords)

customers["cluster"] = (
    customers[dist_names].idxmin(axis = 1).str.split("_").str[-1]
    )

sns.scatterplot(x = "Annual Income", y = "Spending Score", hue = "cluster", 
                palette = "tab10", data = customers, s = 50)
sns.scatterplot(x = "Annual Income", y = "Spending Score", color = "black", 
                data = centroids, s = 100)
plt.show()

new_centroids = (
    round(customers.groupby("cluster")[customers.columns[:2]].mean(), 4)
    )

new_coords = new_centroids.values.tolist()

customers = pd.read_csv("Mall-Customers-Data.csv")
customers = customers[["Annual Income", "Spending Score"]]
variables = customers.columns
centroids, coords = get_centroids(customers, 2)

for i in range(100):
    last_coords = coords.copy()
    customers, dist_names = calculate_distance(customers, coords)
    customers["cluster"] = (
        customers[dist_names].idxmin(axis = 1).str.split("_").str[-1]
        )
    centroids = round(customers.groupby("cluster")[variables].mean(), 4)
    coords = centroids.values.tolist()
    
    if coords == last_coords:
        break

print(f"Total Iterations: {i + 1}")
                       
sns.scatterplot(x = "Annual Income", y = "Spending Score", hue = "cluster", 
                palette = "tab10", data = customers, s = 50)
sns.scatterplot(x = "Annual Income", y = "Spending Score", color = "black", 
                data = centroids.reset_index(drop = True), s = 100)
plt.show()

customers = pd.read_csv("Mall-Customers-Data.csv")
customers = customers[["Annual Income", "Spending Score"]]

def kmeans(df, k, n_iterations = 100):
    variables = df.columns
    
    centroids, coords = get_centroids(df, k)

    for i in range(n_iterations):
        last_coords = coords.copy()
        df, dists = calculate_distance(df, coords)
        df["cluster"] = df[dists].idxmin(axis = 1).str.split("_").str[-1]
        centroids = round(df.groupby("cluster")[variables].mean(), 4)
        coords = centroids.values.tolist()

        if last_coords == coords:
      	    break
    
    print(f"Total Iterations: {i + 1}")

    fig, ax = plt.subplots(figsize = (10, 5))
    sns.scatterplot(x = variables[0], y = variables[1], hue = "cluster", 
                    palette = "tab10", data = df, s = 50, ax = ax)
    sns.scatterplot(x = variables[0], y = variables[1], color = "black", 
                    data = centroids, s = 100, ax = ax)


    plt.tight_layout()
    plt.show()

    return df["cluster"]

clusters = kmeans(customers, 2)