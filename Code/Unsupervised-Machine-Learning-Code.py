# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 22, 2025

# Load to import, clean, and wrangle data
import numpy as np
import os
import polars as pl

# Load to visualize data
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# Load to preprocess, perform, and evaluate clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Set the working directory
os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

# Part 1: K-Means

# Section 1.1: Data Preprocessing

# Load the data from the CSV file, filter, create new columns, and drop columns
customers = (
    pl.read_csv("Credit-Cards-Data.csv")
      .with_columns(
          pl.when(pl.col("gender") == "M") 
            .then(1)
            .otherwise(0)
            .alias("male"),
          pl.when(pl.col("education_level").is_in(["Graduate", "Post-Graduate", 
                                                   "Doctorate"]))
            .then(1)
            .otherwise(0)
            .alias("college_degree")  
          )
      .drop("gender", "dependent_count", "education_level", "marital_status", 
            "total_relationship_count")         
    )
 
# Select columns, convert to a Pandas DataFrame, and calculate correlations    
correlations = (
    customers.select("age", "estimated_income", "months_on_book", 
                     "months_inactive_12_mon", "credit_limit", 
                     "total_trans_amount", "total_trans_count", 
                     "avg_utilization_ratio")
             .to_pandas()
             .corr()
             # Reset index, rename a column, and convert to a Polars DataFrame
             .reset_index()
             .rename(columns = {"index": "variable"})            
             .pipe(pl.from_pandas)
    )

# Drop columns and standardize the numeric variables
customers = customers.drop("months_on_book", "total_trans_count")

num_cols = ["age", "estimated_income", "months_inactive_12_mon", "credit_limit", 
            "total_trans_amount", "avg_utilization_ratio"]
scaled = StandardScaler().fit_transform(customers.select(num_cols))
customers_scaled = customers.with_columns([
    pl.Series(col_name, scaled[:, i]) for i, col_name in enumerate(num_cols)
    ])

# Section 1.2: Machine Learning Model

# Define a function to plot an elbow curve to choose the best number of clusters
def plot_elbow_curve(df, max_clusters):
    # Create an empty list
    inertias = [] 
    
    # Loop over k values, fit k-means, and store each inertia in the list
    for k in range(1, max_clusters + 1):
        model = KMeans(n_clusters = k, random_state = 123)
        model.fit(df)
        inertias.append(model.inertia_)
    
    # Create a line graph showing the inertia for each number of clusters
    plt.figure(figsize = (12, 8))
    plt.plot(range(1, max_clusters + 1), inertias, marker = "o") 
    plt.xticks(ticks = range(1, max_clusters + 1), 
               labels = range(1, max_clusters + 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.title("Inertia vs. Number of Clusters")
    plt.tight_layout()
    plt.show()
    
    # Return the list
    return inertias

# Plot the elbow curve
plot_elbow_curve(customers_scaled.drop("customer_id"), 10)

# Create an empty list
silhouette_scores = []

# Loop over k values, fit k-means, and store each silhouette score in the list
for k in range(2, 11):
    labels = (
        KMeans(n_clusters = k, random_state = 123)
        .fit_predict(customers_scaled.drop("customer_id"))
        )
    silhouette_scores.append(silhouette_score(customers_scaled
                                              .drop("customer_id"), labels))

# Print the best number of clusters based on silhouette score             
print(range(2, 11)[np.argmax(silhouette_scores)])

# Assign customers to clusters and print cluster means by feature
customers = customers.with_columns(
    pl.Series("cluster", 
              (KMeans(n_clusters = 4, random_state = 123) 
               .fit_predict(customers_scaled.drop("customer_id"))) + 1)
    )
   
for col in customers.drop("customer_id", "cluster").columns:
    print(customers.group_by("cluster").agg(pl.mean(col)).sort("cluster"))