# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 20, 2025

import numpy as np
import os
import polars as pl

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

os.chdir("/Users/atz5/Desktop/Machine-Learning-In-Python/Data")

customers = (
    pl.read_csv("Customer-Segmentation-Data.csv")
      .filter(pl.col("marital_status") != "Unknown")
      .to_dummies(["gender", "education_level", "marital_status"]) 
      .rename({"gender_F": "female", 
               "gender_M": "male",
               "education_level_College": "college",
               "education_level_Doctorate": "doctorate",
               "education_level_Graduate": "graduate",
               "education_level_High School": "high school",
               "education_level_Post-Graduate": "post-graduate",
               "education_level_Uneducated": "uneducated",
               "marital_status_Divorced": "divorced", 
               "marital_status_Married": "married", 
               "marital_status_Single": "single"})
    )

correlations = (
    customers.select("age", "dependent_count", "estimated_income", 
                     "months_on_book", "total_relationship_count", 
                     "months_inactive_12_mon", "credit_limit", 
                     "total_trans_amount", "total_trans_count", 
                     "avg_utilization_ratio")
             .to_pandas()
             .corr()
             .reset_index()
             .rename(columns = {"index": "variable"})
    )

customers = customers.drop("months_on_book", "total_trans_count")

num_cols = ["age", "dependent_count", "estimated_income", 
            "total_relationship_count", "months_inactive_12_mon", 
            "credit_limit", "total_trans_amount", "avg_utilization_ratio"]
scaled = StandardScaler().fit_transform(customers.select(num_cols))
customers_scaled = customers.with_columns([
    pl.Series(name, scaled[:, i]) for i, name in enumerate(num_cols)
    ])

def plot_elbow_curve(df, max_clusters):
    inertias = [] 
    
    for k in range(1, max_clusters + 1): 
        model = KMeans(n_clusters = k, random_state = 123)
        model.fit(df)
        inertias.append(model.inertia_)
        
    plt.figure(figsize = (12, 8))
    plt.plot(range(1, max_clusters + 1), inertias, marker = "o") 
    plt.xticks(ticks = range(1, max_clusters + 1), 
               labels = range(1, max_clusters + 1))
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    plt.title("Inertia vs. Number of Clusters")
    plt.tight_layout()
    plt.show()
    
    return inertias

plot_elbow_curve(customers_scaled.drop("customer_id"), 10)

silhouette_scores = []

for k in range(2, 11):
    labels = (
        KMeans(n_clusters = k, random_state = 123)
        .fit_predict(customers_scaled.drop("customer_id"))
        )
    silhouette_scores.append(silhouette_score(customers_scaled
                                              .drop("customer_id"), labels))
            
print(range(2, 11)[np.argmax(silhouette_scores)])

customers = customers.with_columns(
    pl.Series("cluster", 
              (KMeans(n_clusters = 4, random_state = 123) 
               .fit_predict(customers_scaled.drop("customer_id"))) + 1)
    )
 
print(customers.select(pl.col("cluster").value_counts()))   



################################################################################
age: customer age in years.
gender: customer gender (M or F).
dependent_count: number of dependents of each customer.
education_level: level of education ("High School", "Graduate", etc.).
marital_status: marital status ("Single", "Married", etc.).
estimated_income: the estimated income for the customer projected by the data science team.
total_relationship_count: number of times the customer contacted the company.
months_inactive_12_mon: number of months the customer did not use the credit card in the last 12 months.
credit_limit: customer's credit limit.
total_trans_amount: the overall amount of money spent on the card by the customer.
avg_utilization_ratio: daily average utilization ratio.
################################################################################