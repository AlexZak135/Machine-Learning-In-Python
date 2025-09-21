# Title: Unsupervised Machine Learning Module 
# Author: Alexander Zakrzeski
# Date: September 21, 2025

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

# Load the data from the CSV file, filter, create new columns, and drop columns
customers = (
    pl.read_csv("Customer-Segmentation-Data.csv")
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

print(customers.select(pl.col("cluster").value_counts(normalize = True)))   

for col in customers.drop("customer_id", "cluster").columns:
    print(customers.group_by("cluster").agg(pl.mean(col)).sort("cluster"))

















################################################################################
Okay, I want to provide a label to each of my clusters. I own a credit card 
company, and I segmented my customers using k-means. Here is a summary of each
cluster based on the input variables, and please assign a phrase that best fits
each cluster. Also, here is a data dictionary.
Age: customer age in years.
estimated_income: the estimated income for the customer projected by the data 
                   science team.
months_inactive_12_mon: number of months the customer did not use the credit 
                        card in the last 12 months.
total_trans_amount: the overall amount of money spent on the card by the 
                    customer.
avg_utilization_ratio: daily average utilization ratio.
male: 1 for male and 0 for female
college_degree: 1 for college degree and 0 for no degree
married: 1 for married and 0 for not married

Cluster 1:
Mean Age - 47
Mean Estimated Income - 117,329
Mean Months Inactive 12 Months - 2.3 
Mean Credit Limit - 23,560
Mean Total Trans Amount - 5,367 
Mean Avg Utilization Ratio - 0.07
Mean Male (Proportion male and not female) - 0.92 
Mean College Degree (Proportion college degree and not college degree ) - 0.46
Mean Married (Proportion married and not married) - 0.47  
--------------------------------------------------------------------------------
Cluster 2: 
Mean Age - 52
Mean Estimated Income - 54,522
Mean Months Inactive 12 Months - 2.9
Mean Credit Limit - 6,415
Mean Total Trans Amount - 3,371
Mean Avg Utilization Ratio - 0.14
Mean Male (Proportion male and not female) - 0.43  
Mean College Degree (Proportion college degree and not college degree ) - 0.49
Mean Married (Proportion married and not married) - 0.52  
--------------------------------------------------------------------------------
Cluster 3: 
Mean Age - 47
Mean Estimated Income - 43,713
Mean Months Inactive 12 Months - 2.3 
Mean Credit Limit - 2,652
Mean Total Trans Amount - 3,804
Mean Avg Utilization Ratio - 0.65
Mean Male (Proportion male and not female) - 0.24
Mean College Degree (Proportion college degree and not college degree ) - 0.48
Mean Married (Proportion married and not married) - 0.53  
--------------------------------------------------------------------------------
Cluster 4:
Mean Age - 40
Mean Estimated Income - 51,000 
Mean Months Inactive 12 Months - 1.9
Mean Credit Limit - 6,653
Mean Total Trans Amount - 5,265
Mean Avg Utilization Ratio - 0.15
Mean Male (Proportion male and not female) - 0.45
Mean College Degree (Proportion college degree and not college degree ) - 0.48
Mean Married (Proportion married and not married) -  0.47 



################################################################################
age: customer age in years.
male: 1 for male and 0 for female
college_degree: 1 for college degree and 0 for no degree
estimated_income: the estimated income for the customer projected by the data science team.
months_inactive_12_mon: number of months the customer did not use the credit card in the last 12 months.
credit_limit: customer's credit limit.
total_trans_amount: the overall amount of money spent on the card by the customer.
avg_utilization_ratio: daily average utilization ratio.
################################################################################
