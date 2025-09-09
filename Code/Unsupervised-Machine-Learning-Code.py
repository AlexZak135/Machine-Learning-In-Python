# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 8, 2025

import os
import polars as pl

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

customers = (
    pl.read_csv("Customer-Segmentation-Data.csv")
      .with_columns(
          pl.when(pl.col("gender") == "M")
            .then(1) 
            .otherwise(0)
            .alias("gender"), 
          pl.when(pl.col("education_level") == "Uneducated")
            .then(0)
            .when(pl.col("education_level") == "High School")
            .then(1)
            .when(pl.col("education_level") == "College")
            .then(2)
            .when(pl.col("education_level") == "Graduate") 
            .then(3) 
            .when(pl.col("education_level") == "Post-Graduate") 
            .then(4) 
            .when(pl.col("education_level") == "Doctorate") 
            .then(5) 
          )
      .to_dummies(columns = "marital_status")
      .drop("marital_status")
    )




################################################################################
customer_id: unique identifier for each customer.
age: customer age in years.
gender: customer gender (M or F).
dependent_count: number of dependents of each customer.
education_level: level of education ("High School", "Graduate", etc.).
marital_status: marital status ("Single", "Married", etc.).
estimated_income: the estimated income for the customer projected by the data science team.
months_on_book: time as a customer in months.
total_relationship_count: number of times the customer contacted the company.
months_inactive_12_mon: number of months the customer did not use the credit card in the last 12 months.
credit_limit: customer's credit limit.
total_trans_amount: the overall amount of money spent on the card by the customer.
total_trans_count: the overall number of times the customer used the card.
avg_utilization_ratio: daily average utilization ratio.
################################################################################
################################################################################
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

fig = plt.figure(figsize = (20, 10))

for i, column in enumerate(["Age", "Annual Income", "Spending Score"]):
    df_plot = customers.groupby("Cluster")[column].mean()
    ax = fig.add_subplot(2, 2, i + 1)
    ax.bar(df_plot.index, df_plot, color = sns.color_palette("Set1"), 
           alpha = 0.6)
    ax.set_title(f"Average {column.title()} per Cluster", alpha = 0.5)
    ax.xaxis.grid(False)
    
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize = (16, 8))
sns.scatterplot(x = "Age", y = "Annual Income", hue = "Cluster", 
                data = customers, palette = "Set1", alpha = 0.4, ax = axs[0][0])
sns.scatterplot(x = "Age", y = "Spending Score", hue = "Cluster", 
                data = customers, palette = "Set1", alpha = 0.4, ax = axs[0][1],
                legend = False)
sns.scatterplot(x = "Annual Income", y = "Spending Score", hue = "Cluster", 
                data = customers, palette = "Set1", alpha = 0.4, ax = axs[1][0])

plt.tight_layout()
plt.show()

plot_df = pd.crosstab(
    index = customers["Cluster"], columns = customers["Gender"], 
    values = customers["Gender"], aggfunc = "size", normalize = "index"
    )

fig, ax = plt.subplots(figsize = (12, 6))
plot_df.plot.bar(stacked = True, ax = ax, alpha = 0.6)
ax.set_title(f"% Gender per Cluster", alpha = 0.5)
ax.set_ylim(0, 1.4)
ax.legend(frameon = False)
ax.xaxis.grid(False)

plt.tight_layout()
plt.show()